"""
集成测试

测试场景：
1. ImageProcessor -> PlateDetector 串联调用
2. 验证端到端数据流转格式正确
3. mock 掉模型但保留真实的图片预处理流程
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor.image_processor import ImageProcessor
from src.detector.plate_detector import PlateDetector, PlateDetection
from src.utils.constants import PlateType


class TestIntegrationProcessorDetector:
    """ImageProcessor + PlateDetector 集成测试"""
    
    def test_full_pipeline_normal_image(
        self,
        normal_plate_image: np.ndarray,
        mock_hyperlpr_catcher: MagicMock
    ):
        """测试完整流程：正常图像"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                processed_image = processor.process(normal_plate_image)
                
                assert processed_image is not None
                assert isinstance(processed_image, np.ndarray)
                assert len(processed_image.shape) == 3
                assert processed_image.dtype == np.uint8
                
                detections = detector.detect(processed_image)
                
                assert isinstance(detections, list)
                
                for det in detections:
                    self._validate_detection_format(det)
    
    def test_full_pipeline_large_image(
        self,
        large_resolution_image: np.ndarray,
        mock_hyperlpr_catcher: MagicMock,
        mock_memory_tracker
    ):
        """测试完整流程：超大分辨率图像"""
        initial_memory = mock_memory_tracker.get_current()
        
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(max_input_size=1280, fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                original_h, original_w = large_resolution_image.shape[:2]
                assert original_h == 3000
                assert original_w == 4000
                
                processed_image = processor.process(large_resolution_image)
                
                processed_h, processed_w = processed_image.shape[:2]
                assert max(processed_h, processed_w) <= 1280
                
                detections = detector.detect(processed_image)
                
                assert isinstance(detections, list)
                
                current_memory = mock_memory_tracker.get_current()
                memory_increase = current_memory - initial_memory
                
                assert memory_increase < 500, \
                    f"内存增加过多: {memory_increase:.2f} MB"
    
    def test_full_pipeline_multi_plate(
        self,
        multi_plate_image: np.ndarray,
        mock_multi_plate_hyperlpr: MagicMock
    ):
        """测试完整流程：多车牌图像"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_multi_plate_hyperlpr
                detector.yolo_plate_model_available = False
                
                processed_image = processor.process(multi_plate_image)
                
                detections = detector.detect(processed_image)
                
                assert len(detections) >= 2
                
                if len(detections) >= 2:
                    confidences = [det.confidence for det in detections]
                    for i in range(len(confidences) - 1):
                        assert confidences[i] >= confidences[i + 1], \
                            "检测结果应该按置信度降序排列"
                
                for det in detections:
                    self._validate_detection_format(det)
    
    def test_full_pipeline_blurry_image(
        self,
        blurry_image: np.ndarray,
        mock_hyperlpr_catcher: MagicMock
    ):
        """测试完整流程：模糊图像"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(
                    fast_mode=False,
                    denoise=True,
                    contrast_enhance=True
                )
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                processed_image = processor.process(blurry_image)
                
                assert processed_image is not None
                assert isinstance(processed_image, np.ndarray)
                
                detections = detector.detect(processed_image)
                
                assert isinstance(detections, list)
    
    def test_full_pipeline_empty_image(
        self,
        empty_image: np.ndarray,
        mock_hyperlpr_catcher: MagicMock
    ):
        """测试完整流程：空图像"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                processed_image = processor.process(empty_image)
                
                detections = detector.detect(processed_image)
                
                assert isinstance(detections, list)
    
    def test_data_flow_format_validation(
        self,
        normal_plate_image: np.ndarray,
        mock_hyperlpr_catcher: MagicMock
    ):
        """验证端到端数据流转格式"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                input_h, input_w = normal_plate_image.shape[:2]
                input_dtype = normal_plate_image.dtype
                input_channels = normal_plate_image.shape[2] if len(normal_plate_image.shape) == 3 else 1
                
                processed_image = processor.process(normal_plate_image)
                
                assert processed_image.dtype == input_dtype
                assert len(processed_image.shape) == 3
                assert processed_image.shape[2] == input_channels
                
                assert processed_image.min() >= 0
                assert processed_image.max() <= 255
                
                detections = detector.detect(processed_image)
                
                for det in detections:
                    self._validate_detection_data_types(det)
    
    def test_full_mode_preprocessing_effect(
        self,
        standard_test_image: np.ndarray
    ):
        """测试完整模式预处理效果"""
        fast_processor = ImageProcessor(fast_mode=True)
        full_processor = ImageProcessor(
            fast_mode=False,
            denoise=True,
            contrast_enhance=True,
            white_balance=True
        )
        
        fast_result = fast_processor.process(standard_test_image)
        full_result = full_processor.process(standard_test_image)
        
        assert fast_result.shape == standard_test_image.shape
        assert full_result.shape == standard_test_image.shape
        
        assert fast_result.dtype == np.uint8
        assert full_result.dtype == np.uint8
    
    def test_cropped_image_in_detection(
        self,
        normal_plate_image: np.ndarray,
        mock_hyperlpr_catcher: MagicMock
    ):
        """测试检测结果中的裁剪图像"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                processed_image = processor.process(normal_plate_image)
                detections = detector.detect(processed_image)
                
                for det in detections:
                    if det.cropped_image is not None:
                        assert isinstance(det.cropped_image, np.ndarray)
                        assert len(det.cropped_image.shape) == 3
                        assert det.cropped_image.dtype == np.uint8
                        
                        x1, y1, x2, y2 = det.bbox
                        expected_h = y2 - y1
                        expected_w = x2 - x1
                        
                        crop_h, crop_w = det.cropped_image.shape[:2]
                        
                        assert crop_h >= expected_h
                        assert crop_w >= expected_w
    
    def test_pipeline_with_undistort(
        self,
        normal_plate_image: np.ndarray,
        mock_hyperlpr_catcher: MagicMock
    ):
        """测试带畸变矫正的流程"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(
                    fast_mode=False,
                    camera_preset="standard"
                )
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                assert processor.undistort is True
                assert processor.camera_matrix is not None
                assert processor.dist_coeffs is not None
                
                processed_image = processor.process(normal_plate_image)
                
                assert processed_image is not None
                
                detections = detector.detect(processed_image)
                
                assert isinstance(detections, list)
    
    def test_pipeline_resize_flow(
        self,
        large_resolution_image: np.ndarray,
        mock_hyperlpr_catcher: MagicMock
    ):
        """测试缩放流程"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                max_size = 1280
                processor = ImageProcessor(max_input_size=max_size, fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                original_h, original_w = large_resolution_image.shape[:2]
                original_max = max(original_h, original_w)
                
                processed_image = processor.process(large_resolution_image)
                
                processed_h, processed_w = processed_image.shape[:2]
                processed_max = max(processed_h, processed_w)
                
                if original_max > max_size:
                    assert processed_max <= max_size
                    
                    scale = max_size / original_max
                    expected_w = int(original_w * scale)
                    expected_h = int(original_h * scale)
                    
                    assert abs(processed_w - expected_w) <= 1
                    assert abs(processed_h - expected_h) <= 1
                
                detections = detector.detect(processed_image)
                
                assert isinstance(detections, list)
    
    def _validate_detection_format(self, det: PlateDetection):
        """验证检测结果格式"""
        assert isinstance(det, PlateDetection)
        
        assert isinstance(det.bbox, tuple)
        assert len(det.bbox) == 4
        x1, y1, x2, y2 = det.bbox
        assert isinstance(x1, int)
        assert isinstance(y1, int)
        assert isinstance(x2, int)
        assert isinstance(y2, int)
        assert x1 >= 0
        assert y1 >= 0
        assert x2 > x1
        assert y2 > y1
        
        assert isinstance(det.confidence, float)
        assert 0 <= det.confidence <= 1.0
        
        assert isinstance(det.plate_type, str)
        valid_types = [PlateType.BLUE, PlateType.YELLOW, PlateType.GREEN,
                       PlateType.WHITE, PlateType.BLACK, PlateType.UNKNOWN]
        assert det.plate_type in valid_types
        
        assert isinstance(det.angle, float)
        
        if det.corners is not None:
            assert isinstance(det.corners, np.ndarray)
            assert det.corners.shape == (4, 2)
        
        if det.cropped_image is not None:
            assert isinstance(det.cropped_image, np.ndarray)
            assert len(det.cropped_image.shape) == 3
        
        if det.corrected_image is not None:
            assert isinstance(det.corrected_image, np.ndarray)
            assert len(det.corrected_image.shape) == 3
        
        if det.plate_text is not None:
            assert isinstance(det.plate_text, str)
    
    def _validate_detection_data_types(self, det: PlateDetection):
        """验证检测结果数据类型"""
        assert all(isinstance(x, int) for x in det.bbox)
        assert isinstance(det.confidence, (float, np.floating))
        assert isinstance(det.plate_type, str)
        assert isinstance(det.angle, (float, np.floating))
        
        if det.cropped_image is not None:
            assert det.cropped_image.dtype == np.uint8
        
        if det.corrected_image is not None:
            assert det.corrected_image.dtype == np.uint8


class TestIntegrationEdgeCases:
    """集成测试边界情况"""
    
    def test_pipeline_with_file_input(
        self,
        temp_image_file: Path,
        mock_hyperlpr_catcher: MagicMock
    ):
        """测试从文件到检测的完整流程"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                image = processor.load_image(str(temp_image_file))
                
                assert image is not None
                assert isinstance(image, np.ndarray)
                
                processed = processor.process(image)
                
                detections = detector.detect(processed)
                
                assert isinstance(detections, list)
    
    def test_pipeline_with_bytes_input(
        self,
        image_bytes: bytes,
        mock_hyperlpr_catcher: MagicMock
    ):
        """测试从字节数据到检测的完整流程"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                processor = ImageProcessor(fast_mode=True)
                detector = PlateDetector(use_yolo=False)
                detector.hyperlpr_available = True
                detector.hyperlpr_catcher = mock_hyperlpr_catcher
                detector.yolo_plate_model_available = False
                
                image = processor.load_image(image_bytes)
                
                assert image is not None
                
                processed = processor.process(image)
                
                detections = detector.detect(processed)
                
                assert isinstance(detections, list)
    
    def test_pipeline_non_image_file(
        self,
        non_image_file: Path
    ):
        """测试非图像文件的流程"""
        processor = ImageProcessor()
        
        image = processor.load_image(str(non_image_file))
        
        assert image is None
