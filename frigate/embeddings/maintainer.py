"""Maintain embeddings in SQLite-vec."""

import base64
import datetime
import logging
import os
import re
import threading
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from peewee import DoesNotExist
from playhouse.sqliteq import SqliteQueueDatabase
from dataclasses import dataclass

from frigate.comms.embeddings_updater import EmbeddingsRequestEnum, EmbeddingsResponder
from frigate.comms.event_metadata_updater import (
    EventMetadataSubscriber,
    EventMetadataTypeEnum,
)
from frigate.comms.events_updater import EventEndSubscriber, EventUpdateSubscriber
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.const import (
    CLIPS_DIR,
    FRIGATE_LOCALHOST,
    UPDATE_EVENT_DESCRIPTION,
)
from frigate.data_processing.real_time.api import RealTimeProcessorApi
from frigate.data_processing.real_time.bird_processor import BirdProcessor
from frigate.data_processing.real_time.face_processor import FaceProcessor
from frigate.data_processing.types import DataProcessorMetrics
from frigate.embeddings.lpr.lpr import LicensePlateRecognition
from frigate.events.types import EventTypeEnum
from frigate.genai import get_genai_client
from frigate.models import Event
from frigate.types import TrackedObjectUpdateTypesEnum
from frigate.util.builtin import serialize
from frigate.util.image import SharedMemoryFrameManager, area, calculate_region

from .embeddings import Embeddings

logger = logging.getLogger(__name__)

MAX_THUMBNAILS = 10
WPOD_INPUT_SIZE = 512
WPOD_STRIDE = 16
WPOD_CONF_THRESH = 0.5
LP_TARGET_SIZE = (100, 32)  # Standard license plate size
LP_PADDING = 0.1  # 10% padding for license plate crop
WPOD_MIN_CONFIDENCE = 0.5  # Minimum confidence for WPOD-NET detections
LP_MIN_AREA = 100  # Minimum area for license plate region
LP_MAX_AREA_RATIO = 0.3  # Maximum ratio of plate area to car area

@dataclass
class Detection:
    """Represents a license plate detection with corner points and confidence score.
    
    Attributes:
        points: Array of shape (2,4) containing x,y coordinates of the 4 corner points
        prob: Detection confidence score between 0 and 1
    """
    points: np.ndarray  # Shape (2,4) array of x,y coordinates
    prob: float  # Detection confidence score 0-1

    def tl(self) -> np.ndarray:
        """Get top-left point of detection box.
        
        Returns:
            np.ndarray: [x,y] coordinates of top-left corner
        """
        return np.array([min(self.points[0]), min(self.points[1])])
    
    def br(self) -> np.ndarray:
        """Get bottom-right point of detection box.
        
        Returns:
            np.ndarray: [x,y] coordinates of bottom-right corner
        """
        return np.array([max(self.points[0]), max(self.points[1])])

class EmbeddingMaintainer(threading.Thread):
    """Handle embedding queue and post event updates."""

    def __init__(
        self,
        db: SqliteQueueDatabase,
        config: FrigateConfig,
        metrics: DataProcessorMetrics,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(name="embeddings_maintainer")
        self.config = config
        self.metrics = metrics
        self.embeddings = Embeddings(config, db, metrics)

        # Check if we need to re-index events
        if config.semantic_search.reindex:
            self.embeddings.reindex()

        self.event_subscriber = EventUpdateSubscriber()
        self.event_end_subscriber = EventEndSubscriber()
        self.event_metadata_subscriber = EventMetadataSubscriber(
            EventMetadataTypeEnum.regenerate_description
        )
        self.embeddings_responder = EmbeddingsResponder()
        self.frame_manager = SharedMemoryFrameManager()
        self.processors: list[RealTimeProcessorApi] = []

        if self.config.face_recognition.enabled:
            self.processors.append(FaceProcessor(self.config, metrics))

        if self.config.classification.bird.enabled:
            self.processors.append(BirdProcessor(self.config, metrics))

        # create communication for updating event descriptions
        self.requestor = InterProcessRequestor()
        self.stop_event = stop_event
        self.tracked_events: dict[str, list[any]] = {}
        self.genai_client = get_genai_client(config)

        # set license plate recognition conditions
        self.lpr_config = self.config.lpr
        self.requires_license_plate_detection = (
            "license_plate" not in self.config.objects.all_objects
        )
        self.detected_license_plates: dict[str, dict[str, any]] = {}

        if self.lpr_config.enabled:
            self.license_plate_recognition = LicensePlateRecognition(
                self.lpr_config, self.requestor, self.embeddings
            )

    def run(self) -> None:
        """Maintain a SQLite-vec database for semantic search."""
        while not self.stop_event.is_set():
            self._process_requests()
            self._process_updates()
            self._process_finalized()
            self._process_event_metadata()

        self.event_subscriber.stop()
        self.event_end_subscriber.stop()
        self.event_metadata_subscriber.stop()
        self.embeddings_responder.stop()
        self.requestor.stop()
        logger.info("Exiting embeddings maintenance...")

    def _process_requests(self) -> None:
        """Process embeddings requests"""

        def _handle_request(topic: str, data: dict[str, any]) -> str:
            try:
                if topic == EmbeddingsRequestEnum.embed_description.value:
                    return serialize(
                        self.embeddings.embed_description(
                            data["id"], data["description"]
                        ),
                        pack=False,
                    )
                elif topic == EmbeddingsRequestEnum.embed_thumbnail.value:
                    thumbnail = base64.b64decode(data["thumbnail"])
                    return serialize(
                        self.embeddings.embed_thumbnail(data["id"], thumbnail),
                        pack=False,
                    )
                elif topic == EmbeddingsRequestEnum.generate_search.value:
                    return serialize(
                        self.embeddings.embed_description("", data, upsert=False),
                        pack=False,
                    )
                else:
                    for processor in self.processors:
                        resp = processor.handle_request(topic, data)

                        if resp is not None:
                            return resp
            except Exception as e:
                logger.error(f"Unable to handle embeddings request {e}")

        self.embeddings_responder.check_for_request(_handle_request)

    def _process_updates(self) -> None:
        """Process event updates"""
        update = self.event_subscriber.check_for_update(timeout=0.01)

        if update is None:
            return

        source_type, _, camera, frame_name, data = update

        if not camera or source_type != EventTypeEnum.tracked_object:
            return

        camera_config = self.config.cameras[camera]

        # no need to process updated objects if face recognition, lpr, genai are disabled
        if (
            not camera_config.genai.enabled
            and not self.lpr_config.enabled
            and len(self.processors) == 0
        ):
            return

        # Create our own thumbnail based on the bounding box and the frame time
        try:
            yuv_frame = self.frame_manager.get(
                frame_name, camera_config.frame_shape_yuv
            )
        except FileNotFoundError:
            pass

        if yuv_frame is None:
            logger.debug(
                "Unable to process object update because frame is unavailable."
            )
            return

        for processor in self.processors:
            processor.process_frame(data, yuv_frame)

        if self.lpr_config.enabled:
            start = datetime.datetime.now().timestamp()
            processed = self._process_license_plate(data, yuv_frame)

            if processed:
                duration = datetime.datetime.now().timestamp() - start
                self.metrics.alpr_pps.value = (
                    self.metrics.alpr_pps.value * 9 + duration
                ) / 10

        # no need to save our own thumbnails if genai is not enabled
        # or if the object has become stationary
        if self.genai_client is not None and not data["stationary"]:
            if data["id"] not in self.tracked_events:
                self.tracked_events[data["id"]] = []

            data["thumbnail"] = self._create_thumbnail(yuv_frame, data["box"])

            # Limit the number of thumbnails saved
            if len(self.tracked_events[data["id"]]) >= MAX_THUMBNAILS:
                # Always keep the first thumbnail for the event
                self.tracked_events[data["id"]].pop(1)

            self.tracked_events[data["id"]].append(data)

        self.frame_manager.close(frame_name)

    def _process_finalized(self) -> None:
        """Process the end of an event."""
        while True:
            ended = self.event_end_subscriber.check_for_update(timeout=0.01)

            if ended == None:
                break

            event_id, camera, updated_db = ended
            camera_config = self.config.cameras[camera]

            for processor in self.processors:
                processor.expire_object(event_id)

            if event_id in self.detected_license_plates:
                self.detected_license_plates.pop(event_id)

            if updated_db:
                try:
                    event: Event = Event.get(Event.id == event_id)
                except DoesNotExist:
                    continue

                # Skip the event if not an object
                if event.data.get("type") != "object":
                    continue

                # Extract valid thumbnail
                thumbnail = base64.b64decode(event.thumbnail)

                # Embed the thumbnail
                self._embed_thumbnail(event_id, thumbnail)

                if (
                    camera_config.genai.enabled
                    and self.genai_client is not None
                    and event.data.get("description") is None
                    and (
                        not camera_config.genai.objects
                        or event.label in camera_config.genai.objects
                    )
                    and (
                        not camera_config.genai.required_zones
                        or set(event.zones) & set(camera_config.genai.required_zones)
                    )
                ):
                    if event.has_snapshot and camera_config.genai.use_snapshot:
                        with open(
                            os.path.join(CLIPS_DIR, f"{event.camera}-{event.id}.jpg"),
                            "rb",
                        ) as image_file:
                            snapshot_image = image_file.read()

                            img = cv2.imdecode(
                                np.frombuffer(snapshot_image, dtype=np.int8),
                                cv2.IMREAD_COLOR,
                            )

                            # crop snapshot based on region before sending off to genai
                            height, width = img.shape[:2]
                            x1_rel, y1_rel, width_rel, height_rel = event.data["region"]

                            x1, y1 = int(x1_rel * width), int(y1_rel * height)
                            cropped_image = img[
                                y1 : y1 + int(height_rel * height),
                                x1 : x1 + int(width_rel * width),
                            ]

                            _, buffer = cv2.imencode(".jpg", cropped_image)
                            snapshot_image = buffer.tobytes()

                    num_thumbnails = len(self.tracked_events.get(event_id, []))

                    embed_image = (
                        [snapshot_image]
                        if event.has_snapshot and camera_config.genai.use_snapshot
                        else (
                            [
                                data["thumbnail"]
                                for data in self.tracked_events[event_id]
                            ]
                            if num_thumbnails > 0
                            else [thumbnail]
                        )
                    )

                    if camera_config.genai.debug_save_thumbnails and num_thumbnails > 0:
                        logger.debug(
                            f"Saving {num_thumbnails} thumbnails for event {event.id}"
                        )

                        Path(
                            os.path.join(CLIPS_DIR, f"genai-requests/{event.id}")
                        ).mkdir(parents=True, exist_ok=True)

                        for idx, data in enumerate(self.tracked_events[event_id], 1):
                            jpg_bytes: bytes = data["thumbnail"]

                            if jpg_bytes is None:
                                logger.warning(
                                    f"Unable to save thumbnail {idx} for {event.id}."
                                )
                            else:
                                with open(
                                    os.path.join(
                                        CLIPS_DIR,
                                        f"genai-requests/{event.id}/{idx}.jpg",
                                    ),
                                    "wb",
                                ) as j:
                                    j.write(jpg_bytes)

                    # Generate the description. Call happens in a thread since it is network bound.
                    threading.Thread(
                        target=self._embed_description,
                        name=f"_embed_description_{event.id}",
                        daemon=True,
                        args=(
                            event,
                            embed_image,
                        ),
                    ).start()

            # Delete tracked events based on the event_id
            if event_id in self.tracked_events:
                del self.tracked_events[event_id]

    def _process_event_metadata(self):
        # Check for regenerate description requests
        (topic, event_id, source) = self.event_metadata_subscriber.check_for_update(
            timeout=0.01
        )

        if topic is None:
            return

        if event_id:
            self.handle_regenerate_description(event_id, source)

    def _detect_license_plate(self, input: np.ndarray) -> Optional[tuple[tuple[int, int, int, int], np.ndarray, float]]:
        """Detect license plate using WPOD-NET model.
        
        Args:
            input: Input BGR image as numpy array
            
        Returns:
            Optional tuple containing:
                - Box coordinates as (x1,y1,x2,y2)
                - Affine transform matrix
                - Detection confidence score
            Returns None if no plate is detected
            
        Raises:
            ValueError: If input image is invalid
        """
        if input is None or not isinstance(input, np.ndarray):
            raise ValueError("Invalid input image")
        
        if input.size == 0 or len(input.shape) != 3:
            raise ValueError("Invalid input image dimensions")
        
        # Check if model is ready before proceeding
        if not self.embeddings.lp_detector_model or not self.embeddings.lp_detector_model.runner:
            logger.debug("License plate detector model not loaded or still initializing")
            return None

        try:
            # Get original dimensions
            height, width = input.shape[:2]
            min_dim = min(height, width)
            
            # Skip if image is too small
            if min_dim < WPOD_INPUT_SIZE // 4:
                logger.debug("Input image too small for detection")
                return None
            
            # Calculate resize factor
            factor = float(WPOD_INPUT_SIZE)/min_dim
            w = int(width * factor)
            h = int(height * factor)
            
            # Pad to multiple of stride
            w += (WPOD_STRIDE - w % WPOD_STRIDE) if w % WPOD_STRIDE != 0 else 0
            h += (WPOD_STRIDE - h % WPOD_STRIDE) if h % WPOD_STRIDE != 0 else 0
            
            # Resize and normalize
            resized = cv2.resize(input, (w, h))
            img = resized.astype('float32')/255.
            img = np.expand_dims(img, axis=0)
            
            # Run inference
            try:
                Y = self.embeddings.lp_detector_model([img])[0]
            except Exception as e:
                logger.error(f"Error running WPOD-NET inference: {e}")
                return None
            
            if Y.size == 0:
                logger.debug("No detections from WPOD-NET model") 
                return None
            
            # Get probabilities and affine parameters
            Probs = Y[..., 0]
            Affines = Y[..., 2:]
            
            # Find high confidence detections
            points = np.where(Probs > WPOD_MIN_CONFIDENCE)
            
            if len(points[0]) == 0:
                logger.debug(f"No detections above confidence threshold {WPOD_MIN_CONFIDENCE}")
                return None

            # Create list of detections
            detections = []
            for i in range(len(points[0])):
                y, x = points[0][i], points[1][i]
                prob = Probs[y, x]
                affine = Affines[y, x].reshape(2, 3)
                
                # Ensure positive scale factors
                affine[0, 0] = max(affine[0, 0], 0.)
                affine[1, 1] = max(affine[1, 1], 0.)
                
                pts = self._get_plate_points(affine, (x, y), (w, h))
                if pts is not None:
                    # Calculate detection area
                    hull = cv2.convexHull(pts.T.astype(np.float32))
                    area = cv2.contourArea(hull)
                    
                    # Filter by minimum area
                    if area >= LP_MIN_AREA:
                        detections.append(Detection(pts, prob))
                
            # Apply NMS
            if detections:
                selected = self._nms(detections)
                if selected:
                    # Get highest confidence detection after NMS
                    best_det = selected[0]
                    pts = best_det.points
                    
                    # Get the affine transform
                    y, x = points[0][0], points[1][0]
                    affine = Affines[y, x].reshape(2, 3)
                    
                    # Convert to rectangle
                    x1 = int(min(pts[0]))
                    y1 = int(min(pts[1]))
                    x2 = int(max(pts[0]))
                    y2 = int(max(pts[1]))
                    
                    # Scale back to original size
                    x1 = int(x1 * width / w)
                    y1 = int(y1 * height / h)
                    x2 = int(x2 * width / w)
                    y2 = int(y2 * height / h)
                    
                    # Validate detection area ratio
                    det_area = (x2 - x1) * (y2 - y1)
                    image_area = width * height
                    if det_area / image_area > LP_MAX_AREA_RATIO:
                        logger.debug("Detection area too large relative to image")
                        return None
                    
                    duration = datetime.datetime.now().timestamp() - start
                    self.metrics.lpd_fps.value = (
                        self.metrics.lpd_fps.value * 9 + duration
                    ) / 10

                    return ((x1, y1, x2, y2), affine, best_det.prob)
        
        except Exception as e:
            logger.error(f"Error during license plate detection: {e}")
            return None

    def _get_plate_points(self, affine: np.ndarray, center: tuple[int, int], size: tuple[int, int]) -> Optional[np.ndarray]:
        """Convert affine transform to plate corner points.
        Uses perspective transform for better accuracy.
        
        Args:
            affine: 2x3 affine transform matrix
            center: (x,y) center point
            size: (width,height) of input image
            
        Returns:
            4x2 array of corner points or None if invalid
        """
        try:
            # Validate inputs
            if not isinstance(affine, np.ndarray):
                logger.error("Invalid affine matrix type")
                return None
            
            if affine.shape != (2,3):
                logger.error("Invalid affine matrix shape") 
                return None
            
            if not isinstance(center, tuple) or len(center) != 2:
                logger.error("Invalid center point")
                return None
            
            if not isinstance(size, tuple) or len(size) != 2:
                logger.error("Invalid size")
                return None

            net_stride = WPOD_STRIDE
            side = ((208. + 40.)/2.)/net_stride  # 7.75
            vxx = vyy = 0.5  # alpha
            
            # Base rectangle
            base = np.matrix([
                [-vxx, -vyy, 1.],
                [vxx, -vyy, 1.],
                [vxx, vyy, 1.],
                [-vxx, vyy, 1.]
            ]).T
            
            # Apply affine transform
            pts = np.array(affine * base)
            
            # Scale and translate points
            mn = np.array([float(center[0]) + 0.5, float(center[1]) + 0.5])
            pts = pts * side + mn.reshape((2, 1))
            
            # Normalize to image size
            w, h = size
            pts[0] = pts[0] * w / (w//net_stride)
            pts[1] = pts[1] * h / (h//net_stride)
            
            # Convert to homogeneous coordinates
            pts_h = np.vstack([pts, np.ones(4)])
            
            # Get target rectangle points
            w_target, h_target = 100, 32  # Standard LP size
            t_pts = self._get_rect_points(0, 0, w_target, h_target)
            
            # Find perspective transform
            H = self._find_transform_matrix(pts_h, t_pts)
            if H is None:
                return None
            
            # Validate output points
            if not np.all(np.isfinite(pts)):
                logger.error("Invalid plate points (non-finite values)")
                return None
            
            return pts
        
        except Exception as e:
            logger.error(f"Error getting plate points: {e}")
            return None

    def _find_transform_matrix(self, pts: np.ndarray, t_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Find perspective transform matrix H that maps pts to t_pts.
        """
        A = np.zeros((8, 9))
        for i in range(4):
            xi = pts[:, i]
            xil = t_pts[:, i]
            
            A[i*2, 3:6] = -xil[2] * xi
            A[i*2, 6:] = xil[1] * xi
            A[i*2+1, :3] = xil[2] * xi
            A[i*2+1, 6:] = -xil[0] * xi
        
        try:
            _, _, V = np.linalg.svd(A)
            H = V[-1, :].reshape((3, 3))
            return H
        except np.linalg.LinAlgError:
            return None

    def _get_rect_points(self, tlx: int, tly: int, brx: int, bry: int) -> np.ndarray:
        """
        Get rectangle points in homogeneous coordinates.
        """
        return np.matrix([
            [tlx, brx, brx, tlx],
            [tly, tly, bry, bry],
            [1., 1., 1., 1.]
        ], dtype=float)

    def _process_license_plate(self, obj_data: dict[str, any], frame: np.ndarray) -> bool:
        """Process license plate detection and recognition for an object.
        
        Args:
            obj_data: Object detection data dictionary
            frame: Input frame as numpy array
            
        Returns:
            bool: True if processing completed successfully
        """
        # Add input validation
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid frame input")
            return False
        
        if frame.size == 0:
            logger.error("Empty frame")
            return False
        
        if len(frame.shape) != 3:
            logger.error("Invalid frame dimensions") 
            return False

        id = obj_data["id"]

        # Validate inputs
        if not isinstance(frame, np.ndarray):
            logger.error("Invalid frame type")
            return False
        
        if "box" not in obj_data:
            logger.debug("No box coordinates in object data")
            return False

        # Only process car objects
        if obj_data.get("label") != "car":
            logger.debug("Not processing license plate for non car object.")
            return False

        # Skip stationary cars
        if obj_data.get("stationary"):
            logger.debug("Not processing license plate for a stationary car object.")
            return False

        # don't overwrite sub label for objects that have a sub label
        # that is not a license plate
        if obj_data.get("sub_label") and id not in self.detected_license_plates:
            logger.debug(
                f"Not processing license plate due to existing sub label: {obj_data.get('sub_label')}."
            )
            return False

        license_plate: Optional[dict[str, any]] = None

        if self.requires_license_plate_detection:
            logger.debug("Running dedicated license plate detection.")
            car_box = obj_data.get("box")

            if not car_box:
                return False

            rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
            left, top, right, bottom = car_box
            car = rgb[top:bottom, left:right]
            
            # Run WPOD-NET detection
            detection = self._detect_license_plate(car)
            if not detection:
                logger.debug("No license plate detected for car object.")
                return False
            
            plate_box, affine, detection_score = detection
            
            # Validate plate box coordinates
            x1, y1, x2, y2 = plate_box
            if x1 >= x2 or y1 >= y2:
                logger.debug("Invalid plate box coordinates")
                return False
            
            frame_plate_box = [
                left + x1,
                top + y1,
                left + x2,
                top + y2
            ]
            
            # Get perspective transform
            pts = self._get_plate_points(affine, (x1, y1), (x2-x1, y2-y1))
            if pts is not None:
                pts_h = np.vstack([pts, np.ones(4)])
                t_pts = self._get_rect_points(0, 0, 100, 32)
                H = self._find_transform_matrix(pts_h, t_pts)
                if H is not None:
                    # Apply perspective transform to get frontal view
                    plate_img = cv2.warpPerspective(car, H, (100, 32))
                    _, buffer = cv2.imencode('.jpg', plate_img)
                    plate_attr = {
                        "label": "license_plate",
                        "box": frame_plate_box,
                        "score": detection_score,  # Use stored score instead of best_det.prob
                        "plate_img": buffer.tobytes()
                    }
                else:
                    plate_attr = {
                        "label": "license_plate",
                        "box": frame_plate_box,
                        "score": detection_score  # Use stored score
                    }

            if not obj_data.get("current_attributes"):
                obj_data["current_attributes"] = []
            obj_data["current_attributes"].append(plate_attr)

        else:
            # Handle existing license plate attributes
            if not obj_data.get("current_attributes"):
                logger.debug("No attributes to parse.")
                return False

            attributes = obj_data.get("current_attributes", [])
            for attr in attributes:
                if attr.get("label") != "license_plate":
                    continue

                if license_plate is None or attr.get("score", 0.0) > license_plate.get(
                    "score", 0.0
                ):
                    license_plate = attr

            # no license plates detected in this frame
            if not license_plate:
                return False

        license_plate_box = license_plate.get("box")

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        # Before OCR, apply crop_region
        license_plate_frame = self._crop_region(
            frame_bgr,
            license_plate_box,
            padding=0.1  # 10% padding
        )

        if license_plate_frame is None:
            logger.debug("Failed to crop license plate region")
            return False

        # Convert RGB to BGR if coming from detection path
        if self.requires_license_plate_detection:
            license_plate_frame = cv2.cvtColor(license_plate_frame, cv2.COLOR_RGB2BGR)
            
        # run detection with the properly cropped frame
        license_plates, confidences, areas = (
            self.license_plate_recognition.process_license_plate(license_plate_frame)
        )

        logger.debug(f"Text boxes: {license_plates}")
        logger.debug(f"Confidences: {confidences}")
        logger.debug(f"Areas: {areas}")

        if license_plates:
            for plate, confidence, text_area in zip(license_plates, confidences, areas):
                avg_confidence = (
                    (sum(confidence) / len(confidence)) if confidence else 0
                )

                logger.debug(
                    f"Detected text: {plate} (average confidence: {avg_confidence:.2f}, area: {text_area} pixels)"
                )
        else:
            # no plates found
            logger.debug("No text detected")
            return True

        top_plate, top_char_confidences, top_area = (
            license_plates[0],
            confidences[0],
            areas[0],
        )
        avg_confidence = (
            (sum(top_char_confidences) / len(top_char_confidences))
            if top_char_confidences
            else 0
        )

        # Check if we have a previously detected plate for this ID
        if id in self.detected_license_plates:
            prev_plate = self.detected_license_plates[id]["plate"]
            prev_char_confidences = self.detected_license_plates[id]["char_confidences"]
            prev_area = self.detected_license_plates[id]["area"]
            prev_avg_confidence = (
                (sum(prev_char_confidences) / len(prev_char_confidences))
                if prev_char_confidences
                else 0
            )

            # Define conditions for keeping the previous plate
            shorter_than_previous = len(top_plate) < len(prev_plate)
            lower_avg_confidence = avg_confidence <= prev_avg_confidence
            smaller_area = top_area < prev_area

            # Compare character-by-character confidence where possible
            min_length = min(len(top_plate), len(prev_plate))
            char_confidence_comparison = sum(
                1
                for i in range(min_length)
                if top_char_confidences[i] <= prev_char_confidences[i]
            )
            worse_char_confidences = char_confidence_comparison >= min_length / 2

            if (shorter_than_previous or smaller_area) and (
                lower_avg_confidence and worse_char_confidences
            ):
                logger.debug(
                    f"Keeping previous plate. New plate stats: "
                    f"length={len(top_plate)}, avg_conf={avg_confidence:.2f}, area={top_area} "
                    f"vs Previous: length={len(prev_plate)}, avg_conf={prev_avg_confidence:.2f}, area={prev_area}"
                )
                return True

        # Check against minimum confidence threshold
        if avg_confidence < self.lpr_config.threshold:
            logger.debug(
                f"Average confidence {avg_confidence} is less than threshold ({self.lpr_config.threshold})"
            )
            return True

        # Validate plate detection results
        if plate_box is not None:
            x1, y1, x2, y2 = plate_box
            if x1 >= x2 or y1 >= y2:
                logger.debug("Invalid plate box coordinates")
                return False
                
        # Determine subLabel based on known plates, use regex matching
        # Default to the detected plate, use label name if there's a match
        sub_label = next(
            (
                label
                for label, plates in self.lpr_config.known_plates.items()
                if any(re.match(f"^{plate}$", top_plate) for plate in plates)
            ),
            top_plate,
        )

        # Send the result to the API
        resp = requests.post(
            f"{FRIGATE_LOCALHOST}/api/events/{id}/sub_label",
            json={
                "camera": obj_data.get("camera"),
                "subLabel": sub_label,
                "subLabelScore": avg_confidence,
            },
        )

        if resp.status_code == 200:
            self.detected_license_plates[id] = {
                "plate": top_plate,
                "char_confidences": top_char_confidences,
                "area": top_area,
            }

        return True

    def _crop_region(self, image: np.ndarray, box: Optional[list[int]], padding: float = 0.0) -> Optional[np.ndarray]:
        """Crop a region from image with padding.
        
        Args:
            image: Input image as numpy array
            box: [x1, y1, x2, y2] coordinates or None
            padding: Percentage of padding to add (0-1)
            
        Returns:
            Cropped image or None if invalid
            
        Raises:
            ValueError: If inputs are invalid
        """
        if box is None:
            return None
        
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid image input")
        
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Invalid box coordinates")
        
        if not 0 <= padding <= 1:
            raise ValueError("Padding must be between 0 and 1")

        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        
        # Validate coordinates
        if x1 >= x2 or y1 >= y2:
            logger.debug("Invalid box coordinates")
            return None
        
        # Add padding
        w = x2 - x1
        h = y2 - y1
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Adjust coordinates with padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2]

    def _create_thumbnail(self, yuv_frame, box, height=500) -> Optional[bytes]:
        """Return jpg thumbnail of a region of the frame."""
        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        region = calculate_region(
            frame.shape, box[0], box[1], box[2], box[3], height, multiplier=1.4
        )
        frame = frame[region[1] : region[3], region[0] : region[2]]
        width = int(height * frame.shape[1] / frame.shape[0])
        frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
        ret, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        if ret:
            return jpg.tobytes()

        return None

    def _embed_thumbnail(self, event_id: str, thumbnail: bytes) -> None:
        """Embed the thumbnail for an event."""
        self.embeddings.embed_thumbnail(event_id, thumbnail)

    def _embed_description(self, event: Event, thumbnails: list[bytes]) -> None:
        """Embed the description for an event."""
        camera_config = self.config.cameras[event.camera]

        description = self.genai_client.generate_description(
            camera_config, thumbnails, event
        )

        if not description:
            logger.debug("Failed to generate description for %s", event.id)
            return

        # fire and forget description update
        self.requestor.send_data(
            UPDATE_EVENT_DESCRIPTION,
            {
                "type": TrackedObjectUpdateTypesEnum.description,
                "id": event.id,
                "description": description,
            },
        )

        # Embed the description
        self.embeddings.embed_description(event.id, description)

        logger.debug(
            "Generated description for %s (%d images): %s",
            event.id,
            len(thumbnails),
            description,
        )

    def handle_regenerate_description(self, event_id: str, source: str) -> None:
        try:
            event: Event = Event.get(Event.id == event_id)
        except DoesNotExist:
            logger.error(f"Event {event_id} not found for description regeneration")
            return

        camera_config = self.config.cameras[event.camera]
        if not camera_config.genai.enabled or self.genai_client is None:
            logger.error(f"GenAI not enabled for camera {event.camera}")
            return

        thumbnail = base64.b64decode(event.thumbnail)

        logger.debug(
            f"Trying {source} regeneration for {event}, has_snapshot: {event.has_snapshot}"
        )

        if event.has_snapshot and source == "snapshot":
            snapshot_file = os.path.join(CLIPS_DIR, f"{event.camera}-{event.id}.jpg")

            if not os.path.isfile(snapshot_file):
                logger.error(
                    f"Cannot regenerate description for {event.id}, snapshot file not found: {snapshot_file}"
                )
                return

            with open(snapshot_file, "rb") as image_file:
                snapshot_image = image_file.read()
                img = cv2.imdecode(
                    np.frombuffer(snapshot_image, dtype=np.int8), cv2.IMREAD_COLOR
                )

                # crop snapshot based on region before sending off to genai
                # provide full image if region doesn't exist (manual events)
                region = event.data.get("region", [0, 0, 1, 1])
                height, width = img.shape[:2]
                x1_rel, y1_rel, width_rel, height_rel = region

                x1, y1 = int(x1_rel * width), int(y1_rel * height)
                cropped_image = img[
                    y1 : y1 + int(height_rel * height), x1 : x1 + int(width_rel * width)
                ]

                _, buffer = cv2.imencode(".jpg", cropped_image)
                snapshot_image = buffer.tobytes()

        embed_image = (
            [snapshot_image]
            if event.has_snapshot and source == "snapshot"
            else (
                [data["thumbnail"] for data in self.tracked_events[event_id]]
                if len(self.tracked_events.get(event_id, [])) > 0
                else [thumbnail]
            )
        )

        self._embed_description(event, embed_image)

    def _batch_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box and an array of boxes vectorized."""
        # Calculate intersection
        tl = np.maximum(box[:2], boxes[:,:2])
        br = np.minimum(box[2:], boxes[:,2:])
        wh = np.maximum(0., br - tl)
        intersection = wh[:,0] * wh[:,1]
        
        # Calculate union
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        union = area1 + area2 - intersection
        
        return intersection / union

    def _nms(self, detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
        """Apply Non-Maximum Suppression to filter overlapping license plate detections.
        Args:
            detections: List of Detection objects with bounding boxes and scores
            iou_threshold: IoU threshold for suppressing overlapping detections (0-1)
            
        Returns:
            Filtered list of Detection objects after NMS
            
        Note:
            Detections are sorted by confidence score in descending order before NMS
        """
        
        if not detections:
            return []
        
        # Convert to numpy arrays for vectorized operations
        boxes = np.array([np.concatenate([d.tl(), d.br()]) for d in detections])
        scores = np.array([d.prob for d in detections])
        
        # Sort by confidence descending
        indices = np.argsort(scores)[::-1]
        boxes = boxes[indices]
        
        keep = []
        while len(indices) > 0:
            keep.append(indices[0])
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes using vectorized operations
            ious = self._batch_iou(boxes[0], boxes[1:])
            indices = indices[1:][ious <= iou_threshold]
            boxes = boxes[1:][ious <= iou_threshold]
        
        return [detections[i] for i in keep]
