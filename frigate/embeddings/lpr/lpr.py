import logging
import math
import os
import time
from typing import List, Tuple
import threading

import cv2
import numpy as np

from frigate.comms.inter_process import InterProcessRequestor
from frigate.config.classification import LicensePlateRecognitionConfig
from frigate.embeddings.embeddings import Embeddings
from frigate.const import CLIPS_DIR

logger = logging.getLogger(__name__)

MIN_PLATE_LENGTH = 3

class LicensePlateRecognition:
    def __init__(
        self,
        config: LicensePlateRecognitionConfig,
        requestor: InterProcessRequestor,
        embeddings: Embeddings,
    ):
        self.lpr_config = config
        self.requestor = requestor
        self.embeddings = embeddings
        self.recognition_model = self.embeddings.lpr_recognition_model
        self.lpd_model = self.embeddings.lp_detection_model  # WPOD-NET model
        self.ctc_decoder = CTCDecoder()

        self.batch_size = 6

        # Create debug directory
        self.debug_dir = os.path.join(CLIPS_DIR, "lpr")
        os.makedirs(self.debug_dir, exist_ok=True)

        if self.lpr_config.enabled:
            # all models need to be loaded to run LPR
            self.lpd_model._load_model_and_utils()
            self.recognition_model._load_model_and_utils()

    def recognize(
        self, images: List[np.ndarray]
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Recognize the characters on the detected license plates using the recognition model.

        Args:
            images (List[np.ndarray]): A list of images of license plates to recognize.

        Returns:
            Tuple[List[str], List[List[float]]]: A tuple of recognized license plate texts and confidence scores.
        """
        input_shape = [3, 48, 320]
        num_images = len(images)
        all_outputs = []

        # sort images by aspect ratio for processing
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))

        for index in range(0, num_images, self.batch_size):
            batch_indices = indices[index:min(num_images, index + self.batch_size)]
            input_h, input_w = input_shape[1], input_shape[2]
            max_wh_ratio = input_w / input_h
            norm_images = []

            # calculate the maximum aspect ratio in the current batch
            for i in batch_indices:
                h, w = images[i].shape[0:2]
                max_wh_ratio = max(max_wh_ratio, w * 1.0 / h)

            # preprocess the images based on the max aspect ratio
            for i in batch_indices:
                norm_image = self._preprocess_recognition_image(
                    images[i], max_wh_ratio
                )
                norm_image = norm_image[np.newaxis, :]
                norm_images.append(norm_image)

            # Process this batch
            batch_outputs = self.recognition_model(norm_images)
            all_outputs.extend(batch_outputs)

        return self.ctc_decoder(all_outputs)

    def _save_debug_image_async(self, path: str, image: np.ndarray) -> None:
        """Save debug image asynchronously using a thread."""
        def _save():
            try:
                cv2.imwrite(path, image)
                logger.debug(f"Saved debug image to: {path}")
            except Exception as e:
                logger.warning(f"Failed to save debug image: {e}")
            
        threading.Thread(target=_save, daemon=True).start()

    def process_license_plate(
        self, image: np.ndarray, event_id: str = ""
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Process a detected license plate image for recognition.

        Args:
            image (np.ndarray): The cropped license plate image in BGR format.
            event_id (str): The ID of the event associated with this license plate.

        Returns:
            Tuple[List[str], List[float], List[int]]: 
                - List of recognized license plate texts
                - List of confidence scores for each character in each text
                - List of plate areas in pixels
        """
        if self.recognition_model.runner is None:
            # we might still be downloading the models
            logger.debug("Recognition model not loaded")
            return [], [], []

        # Save raw plate image for debugging if event_id is provided
        if event_id:
            raw_filename = f"raw_{event_id}_{int(time.time())}.jpg"
            self._save_debug_image_async(os.path.join(self.debug_dir, raw_filename), image)

        # Run recognition on the plate image
        logger.debug("Running recognition on plate image")
        results, confidences = self.recognize([image])
        logger.debug(f"Recognition results: {list(zip(results, confidences))}")

        if results:
            license_plates = [""] * len(results)
            average_confidences = [[0.0]] * len(results)
            areas = [0] * len(results)

            for i, (plate, conf) in enumerate(zip(results, confidences)):
                height, width = image.shape[:2]
                area = height * width
                average_confidence = conf
                avg_confidence = sum(average_confidence) / len(average_confidence) if average_confidence else 0

                # Save debug image (image is already in BGR format)
                try:
                    filename = f"{plate}_{int(avg_confidence * 100)}_{event_id}_{int(time.time())}.jpg" if event_id else f"{plate}_{int(avg_confidence * 100)}_{int(time.time())}.jpg"
                    self._save_debug_image_async(os.path.join(self.debug_dir, filename), image)
                except Exception as e:
                    logger.warning(f"Failed to save debug image: {e}")

                license_plates[i] = plate
                average_confidences[i] = average_confidence
                areas[i] = area

            # Filter out plates that have a length of less than 3 characters
            # Sort by area, then by plate length, then by confidence all desc
            sorted_data = sorted(
                [
                    (plate, conf, area)
                    for plate, conf, area in zip(
                        license_plates, average_confidences, areas
                    )
                    if len(plate) >= MIN_PLATE_LENGTH
                ],
                key=lambda x: (x[2], len(x[0]), x[1]),
                reverse=True,
            )

            if sorted_data:
                return map(list, zip(*sorted_data))

        # Save debug image for failed recognition (image is already in BGR format)
        filename = f"no_text_{int(time.time())}.jpg"
        self._save_debug_image_async(os.path.join(self.debug_dir, filename), image)

        return [], [], []

    def _preprocess_recognition_image(
        self, image: np.ndarray, max_wh_ratio: float
    ) -> np.ndarray:
        """
        Preprocess an image for recognition by dynamically adjusting its width.

        This method adjusts the width of the image based on the maximum width-to-height ratio
        while keeping the height fixed at 48 pixels. The image is then normalized and padded
        to fit the required input dimensions for recognition.

        Args:
            image (np.ndarray): Input image in BGR format.
            max_wh_ratio (float): Maximum width-to-height ratio for resizing.

        Returns:
            np.ndarray: Preprocessed and padded image in CHW format.
        """
        # fixed height of 48, dynamic width based on ratio
        input_shape = [3, 48, 320]  # CHW format
        input_h, input_w = input_shape[1], input_shape[2]

        # Ensure input is BGR with 3 channels
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape HxWx3, got {image.shape}")

        # dynamically adjust input width based on max_wh_ratio
        input_w = int(input_h * max_wh_ratio)

        # check for model-specific input width
        model_input_w = self.recognition_model.runner.ort.get_inputs()[0].shape[3]
        if isinstance(model_input_w, int) and model_input_w > 0:
            input_w = model_input_w

        h, w = image.shape[:2]
        aspect_ratio = w / h
        resized_w = min(input_w, math.ceil(input_h * aspect_ratio))

        # Resize maintaining aspect ratio
        resized_image = cv2.resize(image, (resized_w, input_h))
        
        # Convert to CHW format
        resized_image = resized_image.transpose((2, 0, 1))
        
        # Normalize to [-0.5, 0.5] range
        resized_image = (resized_image.astype("float32") / 255.0 - 0.5) / 0.5

        # Pad to fixed width
        padded_image = np.zeros((input_shape[0], input_h, input_w), dtype=np.float32)
        padded_image[:, :, :resized_w] = resized_image

        return padded_image


class CTCDecoder:
    """
    A decoder for interpreting the output of a CTC (Connectionist Temporal Classification) model.

    This decoder converts the model's output probabilities into readable sequences of characters
    while removing duplicates and handling blank tokens. It also calculates the confidence scores
    for each decoded character sequence.
    """

    def __init__(self):
        """
        Initialize the CTCDecoder with a list of characters and a character map.

        The character set includes digits, letters, special characters, and a "blank" token
        (used by the CTC model for decoding purposes). A character map is created to map
        indices to characters.
        """
        self.characters = [
            "blank",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ":",
            ";",
            "<",
            "=",
            ">",
            "?",
            "@",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "[",
            "\\",
            "]",
            "^",
            "_",
            "`",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "{",
            "|",
            "}",
            "~",
            "!",
            '"',
            "#",
            "$",
            "%",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            " ",
            " ",
        ]
        self.char_map = {i: char for i, char in enumerate(self.characters)}

    def __call__(
        self, outputs: List[np.ndarray]
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Decode a batch of model outputs into character sequences and their confidence scores.

        The method takes the output probability distributions for each time step and uses
        the best path decoding strategy. It then merges repeating characters and ignores
        blank tokens. Confidence scores for each decoded character are also calculated.

        Args:
            outputs (List[np.ndarray]): A list of model outputs, where each element is
                                        a probability distribution for each time step.

        Returns:
            Tuple[List[str], List[List[float]]]: A tuple of decoded character sequences
                                                and confidence scores for each sequence.
        """
        results = []
        confidences = []
        for output in outputs:
            seq_log_probs = np.log(output + 1e-8)
            best_path = np.argmax(seq_log_probs, axis=1)

            merged_path = []
            merged_probs = []
            for t, char_index in enumerate(best_path):
                if char_index != 0 and (t == 0 or char_index != best_path[t - 1]):
                    merged_path.append(char_index)
                    merged_probs.append(seq_log_probs[t, char_index])

            result = "".join(self.char_map[idx] for idx in merged_path)
            results.append(result)

            confidence = np.exp(merged_probs).tolist()
            confidences.append(confidence)

        return results, confidences
