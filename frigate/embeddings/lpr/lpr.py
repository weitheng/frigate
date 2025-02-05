import logging
import math
import os
import time
from typing import List, Tuple

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
        self.classification_model = self.embeddings.lpr_classification_model
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
            self.classification_model._load_model_and_utils()
            self.recognition_model._load_model_and_utils()

    def classify(
        self, images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[str, float]]]:
        """
        Classify the orientation or category of each detected license plate.

        Args:
            images (List[np.ndarray]): A list of images of detected license plates.

        Returns:
            Tuple[List[np.ndarray], List[Tuple[str, float]]]: A tuple of rotated/normalized plate images
                                                            and classification results with confidence scores.
        """
        num_images = len(images)
        indices = np.argsort([x.shape[1] / x.shape[0] for x in images])
        all_outputs = []

        for i in range(0, num_images, self.batch_size):
            batch_indices = indices[i:min(num_images, i + self.batch_size)]
            norm_images = []
            for idx in batch_indices:
                norm_img = self._preprocess_classification_image(images[idx])
                norm_img = norm_img[np.newaxis, :]
                norm_images.append(norm_img)

            # Process this batch
            batch_outputs = self.classification_model(norm_images)
            all_outputs.extend(batch_outputs)

        return self._process_classification_output(images, all_outputs)

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

    def process_license_plate(
        self, image: np.ndarray, event_id: str = ""
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Complete pipeline for detecting, classifying, and recognizing license plates in the input image.

        Args:
            image (np.ndarray): The input image in which to detect, classify, and recognize license plates.
            event_id (str): The ID of the event associated with this license plate detection.

        Returns:
            Tuple[List[str], List[float], List[int]]: Detected license plate texts, confidence scores, and areas of the plates.
        """
        if (
            self.classification_model.runner is None
            or self.recognition_model.runner is None
        ):
            # we might still be downloading the models
            logger.debug("Model runners not loaded")
            return [], [], []

        # Save raw image before processing if event_id is provided
        if event_id:
            try:
                raw_filename = f"raw_{event_id}.jpg"
                cv2.imwrite(os.path.join(self.debug_dir, raw_filename), image)
            except Exception as e:
                logger.warning(f"Failed to save raw debug image: {e}")

        # Skip detection since we already have a cropped plate from WPOD-NET
        # Directly classify and recognize the plate
        rotated_images, classifications = self.classify([image])
        logger.debug(f"Classification results: {classifications}")

        # keep track of the index of each image for correct area calc later
        sorted_indices = np.argsort([x.shape[1] / x.shape[0] for x in rotated_images])
        reverse_mapping = {
            idx: original_idx for original_idx, idx in enumerate(sorted_indices)
        }

        results, confidences = self.recognize(rotated_images)
        logger.debug(f"Recognition results: {list(zip(results, confidences))}")

        if results:
            license_plates = [""] * len(rotated_images)
            average_confidences = [[0.0]] * len(rotated_images)
            areas = [0] * len(rotated_images)

            # map results back to original image order
            for i, (plate, conf) in enumerate(zip(results, confidences)):
                original_idx = reverse_mapping[i]

                height, width = rotated_images[original_idx].shape[:2]
                area = height * width

                average_confidence = conf
                avg_confidence = sum(average_confidence) / len(average_confidence) if average_confidence else 0

                # Save debug image
                try:
                    save_image = cv2.cvtColor(
                        rotated_images[original_idx], cv2.COLOR_RGB2BGR
                    )
                    filename = f"{plate}_{int(avg_confidence * 100)}_{event_id}.jpg" if event_id else f"{plate}_{int(avg_confidence * 100)}.jpg"
                    cv2.imwrite(os.path.join(self.debug_dir, filename), save_image)
                except Exception as e:
                    logger.warning(f"Failed to save debug image: {e}")

                license_plates[original_idx] = plate
                average_confidences[original_idx] = average_confidence
                areas[original_idx] = area

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

        # Save debug image for failed recognition
        for i, image in enumerate(rotated_images):
            save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            filename = f"no_text_{i}_{int(time.time())}.jpg"
            cv2.imwrite(os.path.join(self.debug_dir, filename), save_image)

        return [], [], []

    @staticmethod
    def _preprocess_classification_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for classification by resizing, normalizing, and padding.

        This method resizes the input image to a fixed height of 48 pixels while adjusting
        the width dynamically up to a maximum of 192 pixels. The image is then normalized and
        padded to fit the required input dimensions for classification.

        Args:
            image (np.ndarray): Input image to preprocess.

        Returns:
            np.ndarray: Preprocessed and padded image.
        """
        # fixed height of 48, dynamic width up to 192
        input_shape = (3, 48, 192)
        input_c, input_h, input_w = input_shape

        h, w = image.shape[:2]
        ratio = w / h
        resized_w = min(input_w, math.ceil(input_h * ratio))

        resized_image = cv2.resize(image, (resized_w, input_h))

        # handle single-channel images (grayscale) if needed
        if input_c == 1 and resized_image.ndim == 2:
            resized_image = resized_image[np.newaxis, :, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1))

        # normalize
        resized_image = (resized_image.astype("float32") / 255.0 - 0.5) / 0.5

        padded_image = np.zeros((input_c, input_h, input_w), dtype=np.float32)
        padded_image[:, :, :resized_w] = resized_image

        return padded_image

    def _process_classification_output(
        self, images: List[np.ndarray], outputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[str, float]]]:
        """
        Process the classification model output by matching labels with confidence scores.

        This method processes the outputs from the classification model and rotates images
        with high confidence of being labeled "180". It ensures that results are mapped to
        the original image order.

        Args:
            images (List[np.ndarray]): List of input images.
            outputs (List[np.ndarray]): Corresponding model outputs.

        Returns:
            Tuple[List[np.ndarray], List[Tuple[str, float]]]: A tuple of processed images and
            classification results (label and confidence score).
        """
        labels = ["0", "180"]
        results = [["", 0.0]] * len(images)
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))

        outputs = np.stack(outputs)

        outputs = [
            (labels[idx], outputs[i, idx])
            for i, idx in enumerate(outputs.argmax(axis=1))
        ]

        for i in range(0, len(images), self.batch_size):
            for j in range(len(outputs)):
                label, score = outputs[j]
                results[indices[i + j]] = [label, score]
                if "180" in label and score >= self.lpr_config.threshold:
                    images[indices[i + j]] = cv2.rotate(images[indices[i + j]], 1)

        return images, results

    def _preprocess_recognition_image(
        self, image: np.ndarray, max_wh_ratio: float
    ) -> np.ndarray:
        """
        Preprocess an image for recognition by dynamically adjusting its width.

        This method adjusts the width of the image based on the maximum width-to-height ratio
        while keeping the height fixed at 48 pixels. The image is then normalized and padded
        to fit the required input dimensions for recognition.

        Args:
            image (np.ndarray): Input image to preprocess.
            max_wh_ratio (float): Maximum width-to-height ratio for resizing.

        Returns:
            np.ndarray: Preprocessed and padded image.
        """
        # fixed height of 48, dynamic width based on ratio
        input_shape = [3, 48, 320]
        input_h, input_w = input_shape[1], input_shape[2]

        assert image.shape[2] == input_shape[0], "Unexpected number of image channels."

        # dynamically adjust input width based on max_wh_ratio
        input_w = int(input_h * max_wh_ratio)

        # check for model-specific input width
        model_input_w = self.recognition_model.runner.ort.get_inputs()[0].shape[3]
        if isinstance(model_input_w, int) and model_input_w > 0:
            input_w = model_input_w

        h, w = image.shape[:2]
        aspect_ratio = w / h
        resized_w = min(input_w, math.ceil(input_h * aspect_ratio))

        resized_image = cv2.resize(image, (resized_w, input_h))
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = (resized_image.astype("float32") / 255.0 - 0.5) / 0.5

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
