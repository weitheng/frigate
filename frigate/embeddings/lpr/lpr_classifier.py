import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from frigate.comms.inter_process import InterProcessRequestor
from frigate.config.semantic_search import LicensePlateRecognitionConfig
from frigate.embeddings.lpr.lpr import LicensePlateRecognition
from frigate.const import UPDATE_OBJECT_SUB_LABEL
from frigate.types import TrackedObjectUpdateTypesEnum

logger = logging.getLogger(__name__)


class LicensePlateClassifier:
    """
    A classifier that processes high resolution snapshots to refine license plate detections.
    """

    def __init__(
        self,
        config: LicensePlateRecognitionConfig,
        requestor: InterProcessRequestor,
        lpr: LicensePlateRecognition,
    ):
        self.config = config
        self.requestor = requestor
        self.lpr = lpr

    def process_snapshot(
        self, snapshot: np.ndarray, region: Optional[List[int]] = None
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Process a high resolution snapshot to detect and recognize license plates.
        
        Args:
            snapshot (np.ndarray): The high resolution snapshot image to process
            region (Optional[List[int]]): Optional region of interest [x1,y1,x2,y2] to process
                                        If not provided, processes entire image
        
        Returns:
            Tuple[List[str], List[float], List[int]]: Detected plate texts, confidence scores, and areas
        """
        if not self.config.enabled:
            return [], [], []

        if (
            self.lpr.detection_model.runner is None
            or self.lpr.classification_model.runner is None
            or self.lpr.recognition_model.runner is None
        ):
            logger.debug("LPR models not loaded")
            return [], [], []

        # Convert BGR to RGB if needed
        if len(snapshot.shape) == 3 and snapshot.shape[2] == 3:
            snapshot = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)

        # Crop to region if provided
        if region is not None:
            x1, y1, x2, y2 = region
            snapshot = snapshot[y1:y2, x1:x2]

        # Process the snapshot
        try:
            plates, scores, areas = self.lpr.process_license_plate(snapshot)

            # Filter results based on confidence threshold
            results = [
                (plate, score, area)
                for plate, score, area in zip(plates, scores, areas)
                if score >= self.config.threshold
            ]

            if results:
                return map(list, zip(*results))

        except Exception as e:
            logger.error(f"Error processing snapshot for LPR: {e}")

        return [], [], []

    def process_event_snapshot(self, event_id: str, snapshot_path: str, camera: str) -> bool:
        """
        Process a snapshot from an event recording to refine license plate detection.
        
        Args:
            event_id (str): The ID of the event
            snapshot_path (str): Path to the snapshot image
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Read the snapshot
            snapshot = cv2.imread(snapshot_path)
            if snapshot is None:
                logger.error(f"Could not read snapshot from {snapshot_path}")
                return False

            # Process the snapshot
            plates, scores, areas = self.process_snapshot(snapshot)
            
            if not plates:
                logger.debug(f"No license plates found in snapshot for event {event_id}")
                return False

            # Get the best result
            best_plate = plates[0]
            best_score = scores[0]
            
            # Update the event with the refined plate detection using proper message type
            self.requestor.send_data(
                UPDATE_OBJECT_SUB_LABEL,
                {
                    "type": TrackedObjectUpdateTypesEnum.sub_label,
                    "id": event_id,
                    "camera": camera,
                    "sub_label": best_plate,
                    "sub_label_score": best_score,
                },
            )
            
            return True

        except Exception as e:
            logger.error(f"Error processing event snapshot for LPR: {e}")
            return False 