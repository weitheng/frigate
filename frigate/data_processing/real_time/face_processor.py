"""Handle processing images for face detection and recognition."""

import base64
import datetime
import logging
import os
import random
import shutil
import string
from typing import Optional

import cv2
import numpy as np
import requests

from frigate.comms.embeddings_updater import EmbeddingsRequestEnum
from frigate.config import FrigateConfig
from frigate.const import FACE_DIR, FRIGATE_LOCALHOST, MODEL_CACHE_DIR
from frigate.util.image import area, SharedMemoryFrameManager

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)


MIN_MATCHING_FACES = 2
MIN_FACE_SCORE = 0.8
NMS_THRESHOLD = 0.3
FACE_INPUT_SIZE = (320, 320)
FACE_QUALITY = 100


class FaceProcessor(RealTimeProcessorApi):
    def __init__(self, config: FrigateConfig, metrics: DataProcessorMetrics, frame_manager: Optional[SharedMemoryFrameManager] = None):
        super().__init__(config, metrics)
        self.face_config = config.face_recognition
        self.face_detector: cv2.FaceDetectorYN = None
        self.landmark_detector: cv2.face.FacemarkLBF = None
        self.recognizer: cv2.face.LBPHFaceRecognizer = None
        self.requires_face_detection = "face" not in self.config.objects.all_objects
        self.detected_faces: dict[str, float] = {}
        self.frame_manager = frame_manager

        download_path = os.path.join(MODEL_CACHE_DIR, "facedet")
        self.model_files = {
            "facedet.onnx": "https://github.com/NickM-27/facenet-onnx/releases/download/v1.0/facedet.onnx",
            "landmarkdet.yaml": "https://github.com/NickM-27/facenet-onnx/releases/download/v1.0/landmarkdet.yaml",
        }

        if not all(
            os.path.exists(os.path.join(download_path, n))
            for n in self.model_files.keys()
        ):
            # conditionally import ModelDownloader
            from frigate.util.downloader import ModelDownloader

            self.downloader = ModelDownloader(
                model_name="facedet",
                download_path=download_path,
                file_names=self.model_files.keys(),
                download_func=self.__download_models,
                complete_func=self.__build_detector,
            )
            self.downloader.ensure_model_files()
        else:
            self.__build_detector()

        self.label_map: dict[int, str] = {}
        self.__build_classifier()

    def __download_models(self, path: str) -> None:
        try:
            file_name = os.path.basename(path)
            # conditionally import ModelDownloader
            from frigate.util.downloader import ModelDownloader

            ModelDownloader.download_from_url(self.model_files[file_name], path)
        except Exception as e:
            logger.error(f"Failed to download {path}: {e}")

    def __build_detector(self) -> None:
        self.face_detector = cv2.FaceDetectorYN.create(
            "/config/model_cache/facedet/facedet.onnx",
            config="",
            input_size=FACE_INPUT_SIZE,
            score_threshold=MIN_FACE_SCORE,
            nms_threshold=NMS_THRESHOLD
        )
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel("/config/model_cache/facedet/landmarkdet.yaml")

    def __build_classifier(self) -> None:
        if not self.landmark_detector:
            return None

        labels = []
        faces = []

        dir = "/media/frigate/clips/faces"
        for idx, name in enumerate(os.listdir(dir)):
            if name == "train":
                continue

            face_folder = os.path.join(dir, name)

            if not os.path.isdir(face_folder):
                continue

            self.label_map[idx] = name
            for image in os.listdir(face_folder):
                img = cv2.imread(os.path.join(face_folder, image))

                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = self.__align_face(img, img.shape[1], img.shape[0])
                faces.append(img)
                labels.append(idx)

        if not faces:
            return

        self.recognizer: cv2.face.LBPHFaceRecognizer = (
            cv2.face.LBPHFaceRecognizer_create(
                radius=2, threshold=(1 - self.face_config.min_score) * 1000
            )
        )
        self.recognizer.train(faces, np.array(labels))

    def __align_face(
        self,
        image: np.ndarray,
        output_width: int,
        output_height: int,
    ) -> np.ndarray:
        _, lands = self.landmark_detector.fit(
            image, np.array([(0, 0, image.shape[1], image.shape[0])])
        )
        landmarks: np.ndarray = lands[0][0]

        # get landmarks for eyes
        leftEyePts = landmarks[42:48]
        rightEyePts = landmarks[36:42]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - 0.35

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX**2) + (dY**2))
        desiredDist = desiredRightEyeX - 0.35
        desiredDist *= output_width
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        # grab the rotation matrix for rotating and scaling the face
        eyesCenter = (
            int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2),
        )
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = output_width * 0.5
        tY = output_height * 0.35
        M[0, 2] += tX - eyesCenter[0]
        M[1, 2] += tY - eyesCenter[1]

        # apply the affine transformation
        return cv2.warpAffine(
            image, M, (output_width, output_height), flags=cv2.INTER_CUBIC
        )

    def __clear_classifier(self) -> None:
        self.recognizer = None
        self.label_map = {}

    def __detect_face(self, input: np.ndarray) -> tuple[int, int, int, int]:
        """Detect faces in input image."""
        if not self.face_detector:
            logger.warning("Face detector not initialized")
            return None

        try:
            self.face_detector.setInputSize((input.shape[1], input.shape[0]))
            faces = self.face_detector.detect(input)

            if faces is None or faces[1] is None:
                return None

            face = None
            for _, potential_face in enumerate(faces[1]):
                raw_bbox = potential_face[0:4].astype(np.uint16)
                x: int = max(raw_bbox[0], 0)
                y: int = max(raw_bbox[1], 0)
                w: int = raw_bbox[2]
                h: int = raw_bbox[3]
                bbox = (x, y, x + w, y + h)

                if face is None or area(bbox) > area(face):
                    face = bbox

            return face
        except Exception as e:
            logger.error(f"Error detecting face: {str(e)}")
            return None

    def __classify_face(self, face_image: np.ndarray) -> tuple[str, float] | None:
        if not self.landmark_detector:
            return None

        if not self.recognizer:
            self.__build_classifier()

            if not self.recognizer:
                return None

        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        img = self.__align_face(img, img.shape[1], img.shape[0])
        index, distance = self.recognizer.predict(img)

        if index == -1:
            return None

        score = 1.0 - (distance / 1000)
        return self.label_map[index], round(score, 2)

    def __update_metrics(self, duration: float) -> None:
        self.metrics.face_rec_fps.value = (
            self.metrics.face_rec_fps.value * 9 + duration
        ) / 10

    def process_frame(self, obj_data: dict[str, any], frame: np.ndarray):
        """Look for faces in image."""
        start = datetime.datetime.now().timestamp()
        id = obj_data["id"]
        camera = obj_data.get("camera")

        # don't run for non person objects
        if obj_data.get("label") != "person":
            logger.debug("Not processing face for non person object.")
            return

        # don't overwrite sub label for objects that have a sub label
        # that is not a face
        if obj_data.get("sub_label") and id not in self.detected_faces:
            logger.debug(
                f"Not processing face due to existing sub label: {obj_data.get('sub_label')}."
            )
            return

        face: Optional[dict[str, any]] = None
        face_box = None

        # First detect face location using detect stream
        if self.requires_face_detection:
            logger.debug("Running manual face detection.")
            person_box = obj_data.get("box")

            if not person_box:
                return

            rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
            left, top, right, bottom = person_box
            person = rgb[top:bottom, left:right]
            face_box = self.__detect_face(person)

            if not face_box:
                logger.debug("Detected no faces for person object.")
                return
                
            # Convert face_box coordinates relative to person box
            face_box = (
                face_box[0] + left,  # Add person box left offset
                face_box[1] + top,   # Add person box top offset
                face_box[2] + left,  # Add person box left offset
                face_box[3] + top    # Add person box top offset
            )
        else:
            # Use face attributes from object detector
            if not obj_data.get("current_attributes"):
                logger.debug("No attributes to parse.")
                return

            attributes: list[dict[str, any]] = obj_data.get("current_attributes", [])
            for attr in attributes:
                if attr.get("label") != "face":
                    continue

                if face is None or attr.get("score", 0.0) > face.get("score", 0.0):
                    face = attr

            if not face:
                return

            face_box = face.get("box")

        # Now get high quality image from record stream
        try:
            # Get frame from record stream using timestamp
            record_frame = self.__get_record_frame(camera, obj_data.get("timestamp"))
            
            if record_frame is not None:
                # Get resolution difference between detect and record streams
                detect_height, detect_width = frame.shape[0] // 3 * 2, frame.shape[1]  # YUV420 format
                record_height, record_width = record_frame.shape[0] // 3 * 2, record_frame.shape[1]
                
                # Calculate scale factors
                width_scale = record_width / detect_width
                height_scale = record_height / detect_height
                
                # Scale face box coordinates for record stream resolution
                scaled_face_box = (
                    int(face_box[0] * width_scale),
                    int(face_box[1] * height_scale),
                    int(face_box[2] * width_scale),
                    int(face_box[3] * height_scale)
                )
                
                # Convert record frame to BGR
                record_frame = cv2.cvtColor(record_frame, cv2.COLOR_YUV2BGR_I420)
                
                # Extract face using scaled coordinates
                face_frame = record_frame[
                    max(0, scaled_face_box[1]) : min(record_frame.shape[0], scaled_face_box[3]),
                    max(0, scaled_face_box[0]) : min(record_frame.shape[1], scaled_face_box[2]),
                ]
            else:
                # Fallback to detect stream if record frame not available
                face_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                face_frame = face_frame[
                    max(0, face_box[1]) : min(frame.shape[0], face_box[3]),
                    max(0, face_box[0]) : min(frame.shape[1], face_box[2]),
                ]
        except Exception as e:
            logger.error(f"Error getting record frame: {e}")
            # Fallback to detect stream
            face_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            face_frame = face_frame[
                max(0, face_box[1]) : min(frame.shape[0], face_box[3]),
                max(0, face_box[0]) : min(frame.shape[1], face_box[2]),
            ]

        res = self.__classify_face(face_frame)

        if not res:
            return

        sub_label, score = res

        # calculate the overall face score as the probability * area of face
        # this will help to reduce false positives from small side-angle faces
        # if a large front-on face image may have scored slightly lower but
        # is more likely to be accurate due to the larger face area
        face_score = round(score * face_frame.shape[0] * face_frame.shape[1], 2)

        logger.debug(
            f"Detected best face for person as: {sub_label} with probability {score} and overall face score {face_score}"
        )

        if self.config.face_recognition.save_attempts:
            # write face to library
            folder = os.path.join(FACE_DIR, "train")
            file = os.path.join(folder, f"{id}-{sub_label}-{score}-{face_score}.webp")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(file, face_frame)

        if score < self.config.face_recognition.threshold:
            logger.debug(
                f"Recognized face distance {score} is less than threshold {self.config.face_recognition.threshold}"
            )
            self.__update_metrics(datetime.datetime.now().timestamp() - start)
            return

        if id in self.detected_faces and face_score <= self.detected_faces[id]:
            logger.debug(
                f"Recognized face distance {score} and overall score {face_score} is less than previous overall face score ({self.detected_faces.get(id)})."
            )
            self.__update_metrics(datetime.datetime.now().timestamp() - start)
            return

        resp = requests.post(
            f"{FRIGATE_LOCALHOST}/api/events/{id}/sub_label",
            json={
                "camera": camera,
                "subLabel": sub_label,
                "subLabelScore": score,
            },
        )

        if resp.status_code == 200:
            self.detected_faces[id] = face_score

        self.__update_metrics(datetime.datetime.now().timestamp() - start)

    def handle_request(self, topic: str, request_data: dict[str, any]) -> dict[str, any] | None:
        """Handle face recognition related requests."""
        if not isinstance(topic, str):
            return {
                "message": "Invalid topic type",
                "success": False,
            }
        
        if not isinstance(request_data, dict):
            return {
                "message": "Invalid request data type",
                "success": False,
            }

        try:
            logger.debug(f"Processing request topic: {topic}")
            
            if topic == EmbeddingsRequestEnum.clear_face_classifier.value:
                self.__clear_classifier()
                logger.info(f"Successfully cleared face classifier")
                return {
                    "message": "Successfully cleared face classifier",
                    "success": True,
                }
            elif topic == EmbeddingsRequestEnum.register_face.value:
                face_name = request_data.get("face_name")
                if not self.__validate_face_name(face_name):
                    return {
                        "message": "Invalid face name",
                        "success": False,
                    }

                try:
                    rand_id = "".join(
                        random.choices(string.ascii_lowercase + string.digits, k=6)
                    )
                    id = f"{face_name}-{rand_id}"

                    if request_data.get("cropped"):
                        thumbnail = request_data["image"]
                    else:
                        img = cv2.imdecode(
                            np.frombuffer(
                                base64.b64decode(request_data["image"]), dtype=np.uint8
                            ),
                            cv2.IMREAD_COLOR,
                        )
                        face_box = self.__detect_face(img)

                        if not face_box:
                            return {
                                "message": "No face was detected.",
                                "success": False,
                            }

                        face = img[face_box[1] : face_box[3], face_box[0] : face_box[2]]
                        _, thumbnail = cv2.imencode(
                            ".webp", face, [int(cv2.IMWRITE_WEBP_QUALITY), FACE_QUALITY]
                        )

                    # write face to library
                    folder = os.path.join(FACE_DIR, face_name)
                    file = os.path.join(folder, f"{id}.webp")
                    os.makedirs(folder, exist_ok=True)
                    if not os.access(folder, os.W_OK):
                        return {
                            "message": f"No write permission for directory: {folder}",
                            "success": False
                        }

                    # save face image
                    with open(file, "wb") as output:
                        output.write(thumbnail.tobytes())

                    self.__clear_classifier()
                    logger.info(f"Successfully registered face: {face_name}")
                    return {
                        "message": "Successfully registered face.",
                        "success": True,
                    }
                except cv2.error as e:
                    return {
                        "message": f"Failed to process image: {str(e)}",
                        "success": False,
                    }
                except Exception as e:
                    logger.error(f"Unexpected error registering face: {str(e)}")
                    return {
                        "message": "Internal server error",
                        "success": False,
                    }
            elif topic == EmbeddingsRequestEnum.reprocess_face.value:
                current_file: str = request_data["image_file"]
                id = current_file[0 : current_file.index("-", current_file.index("-") + 1)]
                face_score = current_file[current_file.rfind("-") : current_file.rfind(".")]
                img = None

                if current_file:
                    img = cv2.imread(current_file)

                if img is None:
                    return {
                        "message": "Invalid image file.",
                        "success": False,
                    }

                if not os.path.exists(current_file):
                    return {
                        "message": f"File not found: {current_file}",
                        "success": False
                    }

                try:
                    res = self.__classify_face(img)

                    if not res:
                        return {
                            "message": "No face was detected.",
                            "success": False,
                        }

                    sub_label, score = res

                    if not self.config.face_recognition.save_attempts:
                        return {
                            "message": "Face saving is disabled",
                            "success": False
                        }

                    # write face to library
                    folder = os.path.join(FACE_DIR, "train")
                    new_file = os.path.join(
                        folder, f"{id}-{sub_label}-{score}-{face_score}.webp"
                    )
                    shutil.move(current_file, new_file)
                    self.__clear_classifier()
                    logger.info(f"Successfully reprocessed face: {current_file}")
                    return {
                        "message": "Successfully registered face.",
                        "success": True,
                    }
                except cv2.error as e:
                    return {
                        "message": f"Failed to process image: {str(e)}",
                        "success": False,
                    }
                except Exception as e:
                    logger.error(f"Unexpected error registering face: {str(e)}")
                    return {
                        "message": "Internal server error",
                        "success": False,
                    }
            else:
                return {
                    "message": f"Unknown request topic: {topic}",
                    "success": False,
                }
        except Exception as e:
            logger.error(f"Unexpected error handling request: {str(e)}")
            return {
                "message": "Internal server error",
                "success": False,
            }

    def expire_object(self, object_id: str):
        if object_id in self.detected_faces:
            self.detected_faces.pop(object_id)

    def __validate_face_name(self, name: str) -> bool:
        """Validate face name meets requirements."""
        if not name or not isinstance(name, str):
            return False
        # Add any other validation rules (e.g., no special chars)
        return True

    def cleanup(self):
        """Cleanup resources when shutting down."""
        if self.face_detector:
            self.face_detector = None
        if self.landmark_detector:
            self.landmark_detector = None
        if self.recognizer:
            self.recognizer = None

    def __get_record_frame(self, camera: str, timestamp: float) -> Optional[np.ndarray]:
        """Get the record frame closest to the given timestamp."""
        if not self.frame_manager:
            return None
        
        try:
            # Get camera config to get frame shape
            camera_config = self.config.cameras[camera]
            
            # Check if record stream is enabled
            if not camera_config.record.enabled:
                logger.debug(f"Record stream not enabled for camera {camera}")
                return None

            # Get frame from record stream
            frame = self.frame_manager.get(
                name=f"{camera}-record",  # Record stream frame name
                shape=camera_config.frame_shape_yuv  # Frame shape needed for numpy array
            )
            return frame
        except Exception as e:
            logger.error(f"Error getting record frame: {e}")
            return None