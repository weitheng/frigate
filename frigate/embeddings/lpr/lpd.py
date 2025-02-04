"""
License Plate Detector Integration Module

This module contains all necessary components to integrate the WPOD-NET license plate detection model
with PaddleOCR for license plate recognition.

Required Dependencies:
- numpy
- opencv-python (cv2)
- onnxruntime

The WPOD-NET model expects:
- Input: NCHW format, BGR image normalized to [0,1]
- Output: (H, W, C) where:
    - Channel 0: Confidence scores
    - Channels 2-7: Affine transformation parameters

The output plates will be in BGR format, ready for PaddleOCR processing.
"""

import numpy as np
import cv2
import onnxruntime
import logging
from typing import List, Tuple, Optional, Union
import time
import os

logger = logging.getLogger(__name__)

class Label:
    """Base class for bounding box handling"""
    def __init__(self, cl=-1, tl=np.array([0.,0.]), br=np.array([0.,0.]), prob=None):
        """
        Initialize a label.
        
        Args:
            cl: Class ID (-1 for unassigned, 0 for license plate)
            tl: Top-left point
            br: Bottom-right point
            prob: Confidence score
        """
        self.__tl = tl  # Top-left point (private)
        self.__br = br  # Bottom-right point (private)
        self.__cl = cl  # Class ID (private)
        self.__prob = prob  # Confidence score (private)

    def __str__(self):
        return 'Class: %d, top_left(x:%f,y:%f), bottom_right(x:%f,y:%f)' % (
            self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1]
        )

    def copy(self):
        return Label(self.__cl, self.__tl.copy(), self.__br.copy(), self.__prob)

    def wh(self): return self.__br - self.__tl  # Width/height of box
    def cc(self): return self.__tl + self.wh()/2  # Center point
    def area(self): return np.prod(self.wh())  # Box area
    def tl(self): return self.__tl
    def br(self): return self.__br
    def tr(self): return np.array([self.__br[0], self.__tl[1]])  # Top-right point
    def bl(self): return np.array([self.__tl[0], self.__br[1]])  # Bottom-left point
    def cl(self): return self.__cl
    def prob(self): return self.__prob

    def set_class(self, cl): self.__cl = cl
    def set_tl(self, tl): self.__tl = tl
    def set_br(self, br): self.__br = br
    def set_prob(self, prob): self.__prob = prob
    
    def set_wh(self, wh):
        """Set width and height while maintaining center point"""
        cc = self.cc()
        self.__tl = cc - .5*wh
        self.__br = cc + .5*wh

class DLabel(Label):
    """Extended label with quadrilateral points for license plates"""
    def __init__(self, cl, pts, prob):
        """
        Initialize a license plate label.
        
        Args:
            cl: Class ID (typically 0 for license plate)
            pts: 2x4 array of 4 corner points of license plate
            prob: Detection confidence
        """
        self.pts = pts  # 2x4 array of plate corners
        tl = np.amin(pts, axis=1)  # Get top-left from min coordinates
        br = np.amax(pts, axis=1)  # Get bottom-right from max coordinates
        super().__init__(cl, tl, br, prob)

class LicensePlateDetector:
    def __init__(self, model_path: str, confidence_threshold: float = None, 
                 max_dimension: int = 608, nms_threshold: float = None):
        """
        Initialize the license plate detector.
        
        Args:
            model_path: Path to the ONNX model file
            confidence_threshold: Minimum confidence score for detections (default: 0.5)
            max_dimension: Maximum dimension for input image preprocessing (default: 608)
            nms_threshold: IOU threshold for non-maximum suppression (default: 0.1)
        """
        self.session = onnxruntime.InferenceSession(model_path)
        self.confidence_threshold = confidence_threshold or 0.5  # Default if not in config
        self.max_dimension = max_dimension
        self.nms_threshold = nms_threshold or 0.1  # Default if not in config
        self.net_stride = 16

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect license plates in an image.
        
        Args:
            image: Input image in BGR format (HxWx3)
            
        Returns:
            Dictionary containing:
            - detections: List of detections, each containing:
                - points: List of 4 points defining the plate quadrilateral
                - confidence: Detection confidence score
            - plates: List of warped plate images ready for OCR
            - inference_time: (optional) Model inference time in seconds
        """
        # Ensure input is BGR
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input must be BGR image with shape HxWx3")
            
        # Preprocess
        processed, orig_shape, _ = self._preprocess_image(image)
        
        # Get input shape from model
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) != 4:  # Should be [batch_size, channels, height, width]
            raise ValueError(f"Unexpected input shape from model: {input_shape}")
            
        # Ensure processed image matches expected shape
        if processed.shape[1:] != tuple(input_shape[1:]):  # Compare everything except batch size
            logger.warning(f"Resizing input from {processed.shape} to match model input {input_shape}")
            # Resize to match expected dimensions
            if len(processed.shape) == 3:  # Add batch dimension if missing
                processed = np.expand_dims(processed, axis=0)
            processed = cv2.resize(processed[0].transpose(1, 2, 0), 
                                 (input_shape[3], input_shape[2])).transpose(2, 0, 1)
            processed = np.expand_dims(processed, axis=0)
        
        # Run inference
        try:
            outputs = self.session.run(None, {"input": processed})[0]
            if outputs.ndim == 4:
                outputs = outputs[0]  # Remove batch dimension
        except Exception as e:
            logger.error(f"Error running WPOD-NET inference: {e}")
            return {"detections": [], "plates": []}
        
        # Post-process
        labels = self._detect_plates(image, processed, outputs)
        
        # Prepare results
        detections = []
        plates = []
        for label in labels:
            # Add detection info
            detections.append({
                'points': label.pts.T.tolist(),
                'confidence': float(label.prob()),
                'center': label.cc().tolist(),
                'width': int(label.wh()[0]),
                'height': int(label.wh()[1])
            })
            # Add warped plate
            plates.append(self._warp_plate(image, label))
        
        return {
            'detections': detections,
            'plates': plates
        }

    def draw_detections(self, image: np.ndarray, detections: List[dict], 
                       color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw detected license plates on the image.
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries from detect()['detections']
            color: BGR tuple for drawing
            thickness: Line thickness
            
        Returns:
            Image with drawn detections
        """
        output = image.copy()
        for det in detections:
            pts = np.array(det['points'], dtype=np.int32)
            cv2.polylines(output, [pts], True, color, thickness)
            # Draw confidence score at center
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(output, f"{det['confidence']:.2f}", 
                       tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        return output

    @staticmethod
    def write_labels(file_path: str, labels: List[Label], write_probs: bool = True) -> None:
        """
        Write labels to file.
        
        Args:
            file_path: Path to output file
            labels: List of Label objects
            write_probs: Whether to include confidence scores
        """
        with open(file_path, 'w') as fd:
            for l in labels:
                cc, wh = l.cc(), l.wh()
                prob = l.prob()
                if prob is not None and write_probs:
                    fd.write('%d %f %f %f %f %f\n' % (
                        l.cl(), cc[0], cc[1], wh[0], wh[1], prob))
                else:
                    fd.write('%d %f %f %f %f\n' % (
                        l.cl(), cc[0], cc[1], wh[0], wh[1]))

    @staticmethod
    def read_labels(file_path: str) -> List[Label]:
        """
        Read labels from file.
        
        Args:
            file_path: Path to input file
            
        Returns:
            List of Label objects
        """
        if not os.path.isfile(file_path):
            return []

        labels = []
        with open(file_path, 'r') as fd:
            for line in fd:
                v = line.strip().split()
                cl = int(v[0])
                ccx, ccy = float(v[1]), float(v[2])
                w, h = float(v[3]), float(v[4])
                prob = float(v[5]) if len(v) == 6 else None
                cc = np.array([ccx, ccy])
                wh = np.array([w, h])
                labels.append(Label(cl, cc - wh/2, cc + wh/2, prob=prob))
        return labels

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Preprocess the input image.
        
        Args:
            image: Input image in BGR format (height x width x 3)
            
        Returns:
            - Preprocessed image in NCHW format
            - Original shape (H, W)
            - New size (W, H)
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a NumPy array")
        if image.size == 0:
            raise ValueError("Empty image provided")

        orig_h, orig_w = image.shape[:2]
        min_dim = min(orig_h, orig_w)
        factor = float(self.max_dimension) / min_dim
        
        # Compute new size (ensure multiple of stride)
        new_w = int(orig_w * factor)
        new_h = int(orig_h * factor)
        if new_w % self.net_stride != 0:
            new_w += self.net_stride - (new_w % self.net_stride)
        if new_h % self.net_stride != 0:
            new_h += self.net_stride - (new_h % self.net_stride)
        new_size = (new_w, new_h)
        
        # Resize and normalize
        resized = cv2.resize(image, new_size)
        resized = resized.astype(np.float32) / 255.0
        # Convert to CHW format and add batch dimension to get NCHW
        processed = np.transpose(resized, (2, 0, 1))
        processed = np.expand_dims(processed, axis=0)
        
        return processed, (orig_h, orig_w), new_size

    def _detect_plates(self, original_img: np.ndarray, processed_img: np.ndarray, 
                      model_output: np.ndarray) -> List[DLabel]:
        """
        Detect license plates from model output.
        
        Args:
            original_img: Original input image
            processed_img: Preprocessed image used for inference
            model_output: Raw model output
            
        Returns:
            List of DLabel objects for detected plates
        """
        # Extract probability scores and affine parameters
        probs = model_output[..., 0]
        affines = model_output[..., 2:]
        
        # Get detections above threshold
        ys, xs = np.where(probs > self.confidence_threshold)
        
        labels = []
        for grid_y, grid_x in zip(ys, xs):
            affine = affines[grid_y, grid_x]
            prob = probs[grid_y, grid_x]
            
            if np.isnan(affine).any():
                continue
                
            # Get plate corners
            pts = self._compute_plate_corners(affine, grid_x, grid_y)
            
            # Scale back to original image size
            h, w = model_output.shape[:2]
            scale_x = original_img.shape[1] / w
            scale_y = original_img.shape[0] / h
            pts[0, :] *= scale_x
            pts[1, :] *= scale_y
            
            # Create DLabel
            labels.append(DLabel(0, pts, prob))
            
        # Apply NMS
        return self._non_max_suppression(labels)

    def _warp_plate(self, img: np.ndarray, label: DLabel) -> np.ndarray:
        """
        Warp a detected license plate region to a rectangular shape.
        
        Args:
            img: Original image in BGR format
            label: DLabel object containing plate points
            
        Returns:
            Warped plate image in BGR format, ready for OCR
        """
        points = label.pts
        # Compute output dimensions
        width = max(
            np.linalg.norm(points[:, 1] - points[:, 0]),
            np.linalg.norm(points[:, 2] - points[:, 3])
        )
        height = max(
            np.linalg.norm(points[:, 3] - points[:, 0]),
            np.linalg.norm(points[:, 2] - points[:, 1])
        )
        
        # Add margins (10%)
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)
        out_width = int(width + 2 * margin_x)
        out_height = int(height + 2 * margin_y)
        
        # Destination points
        dst_pts = np.array([
            [margin_x, margin_y],
            [margin_x + width, margin_y],
            [margin_x + width, margin_y + height],
            [margin_x, margin_y + height]
        ], dtype=np.float32).T
        
        # Calculate and apply transform
        H = self._find_transform_matrix(points, dst_pts)
        warped = cv2.warpPerspective(img, H, (out_width, out_height))
        
        return warped

    def _non_max_suppression(self, labels: List[DLabel], iou_threshold: float = None) -> List[DLabel]:
        """Apply NMS to filter overlapping detections."""
        if not labels:
            return []
            
        if iou_threshold is None:
            iou_threshold = self.nms_threshold
            
        # Sort by confidence
        labels.sort(key=lambda x: x.prob(), reverse=True)
        
        filtered = []
        while labels:
            best = labels.pop(0)
            filtered.append(best)
            
            # Filter overlapping detections using Label class methods
            labels = [
                label for label in labels
                if self._calculate_iou(label.tl(), label.br(), best.tl(), best.br()) < iou_threshold
            ]
            
        return filtered

    def _calculate_iou(self, tl1: np.ndarray, br1: np.ndarray, tl2: np.ndarray, br2: np.ndarray) -> float:
        """Calculate IoU between two boxes using Label class coordinates."""
        wh1 = br1 - tl1
        wh2 = br2 - tl2
        intersect_tl = np.maximum(tl1, tl2)
        intersect_br = np.minimum(br1, br2)
        intersect_wh = np.maximum(intersect_br - intersect_tl, 0)
        area_intersect = np.prod(intersect_wh)
        area1 = np.prod(wh1)
        area2 = np.prod(wh2)
        return area_intersect / (area1 + area2 - area_intersect)

    def _compute_plate_corners(self, affine: np.ndarray, grid_x: int, grid_y: int) -> np.ndarray:
        """Compute plate corners from affine parameters."""
        A = affine.reshape(2, 3)
        A[0, 0] = max(A[0, 0], 0.0)
        A[1, 1] = max(A[1, 1], 0.0)
        
        vxx = vyy = 0.5
        side = 7.75  # ((208 + 40) / 2) / net_stride
        
        base = np.array([
            [-vxx, -vyy, 1.0],
            [ vxx, -vyy, 1.0],
            [ vxx,  vyy, 1.0],
            [-vxx,  vyy, 1.0]
        ]).T
        
        mn = np.array([grid_x + 0.5, grid_y + 0.5]).reshape((2, 1))
        return (A @ base) * side + mn

    def _find_transform_matrix(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """Calculate perspective transform matrix using SVD."""
        A = np.zeros((8, 9))
        for i in range(4):
            x, y = src_pts[0, i], src_pts[1, i]
            u, v = dst_pts[0, i], dst_pts[1, i]
            A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
            A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
        
        _, _, V = np.linalg.svd(A)
        H = V[-1, :].reshape((3, 3))
        return H / H[2, 2]

# Example usage:
if __name__ == "__main__":
    # Initialize detector
    detector = LicensePlateDetector(
        model_path="path/to/model.onnx",
        confidence_threshold=0.5
    )
    
    # Load and process image
    image = cv2.imread("test_image.jpg")
    results = detector.detect(image)
    
    # Draw results
    output_image = detector.draw_detections(image, results['detections'])
    
    # Display or save results
    cv2.imshow("Detections", output_image)
    cv2.waitKey(0)
    
    # Access individual plate images
    for i, plate in enumerate(results['plates']):
        cv2.imshow(f"Plate {i+1}", plate)
        cv2.waitKey(0) 