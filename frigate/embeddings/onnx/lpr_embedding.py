import logging
import os
import warnings

import cv2
import numpy as np

from frigate.comms.inter_process import InterProcessRequestor
from frigate.const import MODEL_CACHE_DIR
from frigate.types import ModelStatusTypesEnum
from frigate.util.downloader import ModelDownloader

from .base_embedding import BaseEmbedding
from .runner import ONNXModelRunner

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The class CLIPFeatureExtractor is deprecated",
)

logger = logging.getLogger(__name__)

LPR_EMBEDDING_SIZE = 256


class PaddleOCRDetection(BaseEmbedding):
    def __init__(
        self,
        model_size: str,
        requestor: InterProcessRequestor,
        device: str = "AUTO",
    ):
        super().__init__(
            model_name="paddleocr-onnx",
            model_file="detection.onnx",
            download_urls={
                "detection.onnx": "https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/detection.onnx"
            },
        )
        self.requestor = requestor
        self.model_size = model_size
        self.device = device
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.runner: ONNXModelRunner | None = None
        files_names = list(self.download_urls.keys())
        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            ModelDownloader.mark_files_state(
                self.requestor,
                self.model_name,
                files_names,
                ModelStatusTypesEnum.downloaded,
            )
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = ONNXModelRunner(
                os.path.join(self.download_path, self.model_file),
                self.device,
                self.model_size,
            )

    def _preprocess_inputs(self, raw_inputs):
        preprocessed = []
        for x in raw_inputs:
            preprocessed.append(x)
        return [{"x": preprocessed[0]}]


class PaddleOCRClassification(BaseEmbedding):
    def __init__(
        self,
        model_size: str,
        requestor: InterProcessRequestor,
        device: str = "AUTO",
    ):
        super().__init__(
            model_name="paddleocr-onnx",
            model_file="classification.onnx",
            download_urls={
                "classification.onnx": "https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/classification.onnx"
            },
        )
        self.requestor = requestor
        self.model_size = model_size
        self.device = device
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.runner: ONNXModelRunner | None = None
        files_names = list(self.download_urls.keys())
        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            ModelDownloader.mark_files_state(
                self.requestor,
                self.model_name,
                files_names,
                ModelStatusTypesEnum.downloaded,
            )
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = ONNXModelRunner(
                os.path.join(self.download_path, self.model_file),
                self.device,
                self.model_size,
            )

    def _preprocess_inputs(self, raw_inputs):
        processed = []
        for img in raw_inputs:
            processed.append({"x": img})
        return processed


class PaddleOCRRecognition(BaseEmbedding):
    def __init__(
        self,
        model_size: str,
        requestor: InterProcessRequestor,
        device: str = "AUTO",
    ):
        super().__init__(
            model_name="paddleocr-onnx",
            model_file="recognition.onnx",
            download_urls={
                "recognition.onnx": "https://github.com/hawkeye217/paddleocr-onnx/raw/refs/heads/master/models/recognition.onnx"
            },
        )
        self.requestor = requestor
        self.model_size = model_size
        self.device = device
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.runner: ONNXModelRunner | None = None
        files_names = list(self.download_urls.keys())
        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            ModelDownloader.mark_files_state(
                self.requestor,
                self.model_name,
                files_names,
                ModelStatusTypesEnum.downloaded,
            )
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = ONNXModelRunner(
                os.path.join(self.download_path, self.model_file),
                self.device,
                self.model_size,
            )

    def _preprocess_inputs(self, raw_inputs):
        processed = []
        for img in raw_inputs:
            processed.append({"x": img})
        return processed


class LicensePlateDetector(BaseEmbedding):
    def __init__(
        self,
        model_size: str,
        requestor: InterProcessRequestor,
        device: str = "AUTO",
    ):
        super().__init__(
            model_name="lpdetection-onnx",
            model_file="lpdetection.onnx",
            download_urls={
                "lpdetection.onnx": "https://github.com/weitheng/lpdetection-onnx/raw/refs/heads/master/models/lpdetection.onnx"
            },
        )

        self.requestor = requestor
        self.model_size = model_size
        self.device = device
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.runner: ONNXModelRunner | None = None
        files_names = list(self.download_urls.keys())
        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            ModelDownloader.mark_files_state(
                self.requestor,
                self.model_name,
                files_names,
                ModelStatusTypesEnum.downloaded,
            )
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = ONNXModelRunner(
                os.path.join(self.download_path, self.model_file),
                self.device,
                self.model_size,
            )
            
    def __call__(self, inputs):
        """Custom call method for WPOD-NET license plate detector.
        
        Args:
            inputs: List containing a single numpy array image in BGR format
            
        Returns:
            List of model outputs for license plate detection
        """
        self._load_model_and_utils()
        if self.runner is None:
            logger.error(f"{self.model_name} model is not loaded.")
            return []
            
        if len(inputs) != 1:
            logger.warning("WPOD-NET supports single input only. Using first image only.")
            
        try:
            # Preprocess the input image
            processed = self._preprocess_inputs(inputs)
            if not processed:
                logger.error("Failed to preprocess input for license plate detection")
                return []
                
            # Run inference
            outputs = []
            for input_dict in processed:
                # Get input names from model
                input_names = self.runner.get_input_names()
                # Prepare inputs for the model
                onnx_inputs = {}
                for name in input_names:
                    if name in input_dict:
                        onnx_inputs[name] = input_dict[name]
                    else:
                        logger.warning(f"Expected input '{name}' not found in preprocessed data")
                        return []
                
                # Run the model
                result = self.runner.run(onnx_inputs)
                outputs.append(result)
                
            return outputs
            
        except Exception as e:
            logger.error(f"Error in WPOD-NET detection: {str(e)}")
            return []

    def _preprocess_inputs(self, raw_inputs):
        logger.debug(f"Preprocessing {len(raw_inputs)} images for license plate detection")
        preprocessed = []
        for idx, img in enumerate(raw_inputs):
            try:
                logger.debug(f"Processing image {idx} with shape: {img.shape}, dtype: {img.dtype}")

                # Ensure input is uint8 before normalization
                if img.dtype != np.uint8:
                    logger.warning(f"Image {idx} has unexpected dtype: {img.dtype}, expected uint8")
                    continue

                # Convert to BGR if needed (WPOD-NET expects BGR input)
                if len(img.shape) == 2:
                    logger.debug(f"Converting grayscale image {idx} to BGR")
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif len(img.shape) == 3:
                    if img.shape[2] == 4:
                        logger.debug(f"Converting RGBA image {idx} to BGR")
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    elif img.shape[2] == 3 and img.dtype == np.uint8:
                        # Assume it's already BGR since that's OpenCV's default
                        pass
                    else:
                        logger.warning(f"Image {idx} has unexpected format: channels={img.shape[2]}")
                        continue

                # Calculate resize dimensions
                h, w = img.shape[:2]
                ratio = float(max(h, w))/min(h, w)
                side = int(ratio * 288.)  # Base size from example implementation
                bound_dim = min(side + (side % (2**4)), 608)

                # Calculate resize factor
                factor = min(bound_dim / max(h, w), 1.0)
                resize_h = max(int(round(int(h * factor) / 32) * 32), 32)
                resize_w = max(int(round(int(w * factor) / 32) * 32), 32)

                # Pad to multiple of stride
                WPOD_STRIDE = 16
                resize_w += (WPOD_STRIDE - resize_w % WPOD_STRIDE) if resize_w % WPOD_STRIDE != 0 else 0
                resize_h += (WPOD_STRIDE - resize_h % WPOD_STRIDE) if resize_h % WPOD_STRIDE != 0 else 0

                logger.debug(f"Resizing image {idx} from {h}x{w} to {resize_h}x{resize_w}")

                # Resize image
                resized = cv2.resize(img, (resize_w, resize_h))

                # Normalize to 0-1 range (im2single equivalent)
                normalized = resized.astype(np.float32) / 255.0

                # WPOD-NET expects input in NCHW format
                normalized = normalized.transpose(2, 0, 1)  # HWC to CHW
                normalized = np.expand_dims(normalized, axis=0)  # Add batch dimension -> NCHW

                # Verify dimensions and ranges
                if normalized.shape[1] != 3:
                    logger.error(f"Image {idx} has wrong number of channels after preprocessing: {normalized.shape[1]}")
                    continue

                if not (0 <= normalized.min() <= normalized.max() <= 1.0):
                    logger.warning(f"Image {idx} values outside expected range: [{normalized.min():.3f}, {normalized.max():.3f}]")

                logger.debug(f"Preprocessed image {idx} shape: {normalized.shape}, value range: [{normalized.min():.3f}, {normalized.max():.3f}]")
                preprocessed.append({"input": normalized})
            except Exception as e:
                logger.error(f"Error preprocessing image {idx}: {str(e)}")
                continue

        if not preprocessed:
            logger.error("No images were successfully preprocessed for license plate detection")
        else:
            logger.debug(f"Successfully preprocessed {len(preprocessed)} images")

        return preprocessed
