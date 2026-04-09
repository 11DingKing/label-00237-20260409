"""
PlateDetector 单元测试

测试场景：
1. 正常车牌检测
2. 模糊图片检测
3. 空图片检测
4. 非图片文件处理
5. 多车牌同框检测（验证按置信度降序排列）
6. 超大分辨率图片检测（验证不会 OOM）
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector.plate_detector import PlateDetector, PlateDetection
from src.utils.constants import PlateType, DEFAULT_CONFIDENCE_THRESHOLD


class TestPlateDetectorNormal:
    """正常车牌检测测试"""
    
    def test_normal_plate_detection_with_hyperlpr(
        self,
        mock_detector_with_hyperlpr: PlateDetector,
        normal_plate_image: np.ndarray
    ):
        """测试正常车牌检测 - 使用 HyperLPR3"""
        detections = mock_detector_with_hyperlpr.detect(normal_plate_image)
        
        assert len(detections) > 0, "应该检测到至少一个车牌"
        
        for det in detections:
            assert isinstance(det, PlateDetection)
            
            assert len(det.bbox) == 4
            x1, y1, x2, y2 = det.bbox
            assert x1 >= 0 and y1 >= 0
            assert x2 > x1 and y2 > y1
            
            assert 0 <= det.confidence <= 1.0
            
            assert det.plate_type in [PlateType.BLUE, PlateType.YELLOW, 
                                   PlateType.GREEN, PlateType.WHITE, 
                                   PlateType.BLACK, PlateType.UNKNOWN]
            
            assert isinstance(det.angle, float)
            
            if det.cropped_image is not None:
                assert isinstance(det.cropped_image, np.ndarray)
                assert len(det.cropped_image.shape) == 3
    
    def test_normal_plate_detection_with_yolo(
        self,
        mock_detector_with_yolo: PlateDetector,
        normal_plate_image: np.ndarray
    ):
        """测试正常车牌检测 - 使用 YOLO"""
        detections = mock_detector_with_yolo.detect(normal_plate_image)
        
        assert len(detections) > 0
        
        for det in detections:
            assert isinstance(det, PlateDetection)
            assert len(det.bbox) == 4
            assert 0 <= det.confidence <= 1.0


class TestPlateDetectorBlurry:
    """模糊图片检测测试"""
    
    def test_blurry_image_detection(
        self,
        mock_detector_with_hyperlpr: PlateDetector,
        blurry_image: np.ndarray
    ):
        """测试模糊图片检测"""
        detections = mock_detector_with_hyperlpr.detect(blurry_image)
        
        assert isinstance(detections, list)
        
        for det in detections:
            assert isinstance(det, PlateDetection)
    
    def test_blurry_image_with_cv_method(
        self,
        mock_detector_no_models: PlateDetector,
        blurry_image: np.ndarray
    ):
        """测试模糊图片使用传统 CV 方法"""
        detections = mock_detector_no_models.detect(blurry_image)
        
        assert isinstance(detections, list)


class TestPlateDetectorEmpty:
    """空图片检测测试"""
    
    def test_empty_image_detection(
        self,
        mock_detector_with_hyperlpr: PlateDetector,
        empty_image: np.ndarray
    ):
        """测试空图片检测"""
        detections = mock_detector_with_hyperlpr.detect(empty_image)
        
        assert isinstance(detections, list)
        
        for det in detections:
            assert isinstance(det, PlateDetection)
    
    def test_empty_image_with_cv_method(
        self,
        mock_detector_no_models: PlateDetector,
        empty_image: np.ndarray
    ):
        """测试空图片使用传统 CV 方法"""
        detections = mock_detector_no_models.detect(empty_image)
        
        assert isinstance(detections, list)


class TestPlateDetectorNonImage:
    """非图片文件处理测试"""
    
    def test_non_image_file_loading(
        self,
        non_image_file: Path
    ):
        """测试非图片文件加载"""
        from src.preprocessor.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        result = processor.load_image(str(non_image_file))
        
        assert result is None, "非图片文件应该返回 None"
    
    def test_invalid_numpy_array(
        self,
        mock_detector_with_hyperlpr: PlateDetector
    ):
        """测试无效的 numpy 数组"""
        invalid_array = np.array([])
        
        with pytest.raises(ValueError):
            mock_detector_with_hyperlpr.detect(invalid_array)
    
    def test_none_image(
        self,
        mock_detector_with_hyperlpr: PlateDetector
    ):
        """测试 None 图像"""
        with patch.object(mock_detector_with_hyperlpr.logger, 'error'):
            with pytest.raises(Exception):
                mock_detector_with_hyperlpr.detect(None)


class TestPlateDetectorMultiPlate:
    """多车牌同框检测测试"""
    
    def test_multi_plate_detection(
        self,
        mock_detector_with_multi_plate: PlateDetector,
        multi_plate_image: np.ndarray
    ):
        """测试多车牌同框检测"""
        detections = mock_detector_with_multi_plate.detect(multi_plate_image)
        
        assert len(detections) >= 2, "应该检测到多个车牌"
        
        for det in detections:
            assert isinstance(det, PlateDetection)
            assert len(det.bbox) == 4
            assert 0 <= det.confidence <= 1.0
    
    def test_multi_plate_sorted_by_confidence(
        self,
        mock_detector_with_multi_plate: PlateDetector,
        multi_plate_image: np.ndarray
    ):
        """测试多车牌结果按置信度降序排列"""
        detections = mock_detector_with_multi_plate.detect(multi_plate_image)
        
        if len(detections) >= 2:
            confidences = [det.confidence for det in detections]
            
            for i in range(len(confidences) - 1):
                assert confidences[i] >= confidences[i + 1], \
                    f"检测结果应该按置信度降序排列，" \
                    f"但第{i}个({confidences[i]}) < 第{i+1}个({confidences[i+1]})"
    
    def test_multi_plate_bbox_distinct(
        self,
        mock_detector_with_multi_plate: PlateDetector,
        multi_plate_image: np.ndarray
    ):
        """测试多车牌边界框不重叠或重叠合理"""
        detections = mock_detector_with_multi_plate.detect(multi_plate_image)
        
        if len(detections) >= 2:
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    bbox1 = detections[i].bbox
                    bbox2 = detections[j].bbox
                    
                    iou = self._compute_iou(bbox1, bbox2)
                    
                    assert iou < 0.9, \
                        f"车牌 {i} 和 {j} 的边界框重叠度过高 (IoU={iou})"
    
    @staticmethod
    def _compute_iou(
        box1: tuple,
        box2: tuple
    ) -> float:
        """计算两个边界框的 IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class TestPlateDetectorLargeResolution:
    """超大分辨率图片检测测试"""
    
    def test_large_resolution_image_shape(
        self,
        large_resolution_image: np.ndarray
    ):
        """测试超大分辨率图像的尺寸"""
        h, w = large_resolution_image.shape[:2]
        
        assert h == 3000, f"高度应该是 3000，实际是 {h}"
        assert w == 4000, f"宽度应该是 4000，实际是 {w}"
    
    def test_large_resolution_detection(
        self,
        mock_detector_with_hyperlpr: PlateDetector,
        large_resolution_image: np.ndarray,
        mock_memory_tracker
    ):
        """测试超大分辨率图片检测 - 验证不会 OOM"""
        initial_memory = mock_memory_tracker.get_current()
        
        detections = mock_detector_with_hyperlpr.detect(large_resolution_image)
        
        current_memory = mock_memory_tracker.get_current()
        memory_increase = current_memory - initial_memory
        
        assert isinstance(detections, list)
        
        assert memory_increase < 500, \
            f"内存增加过多: {memory_increase:.2f} MB，" \
            f"初始: {initial_memory:.2f} MB，" \
            f"当前: {current_memory:.2f} MB"
    
    def test_large_resolution_with_processor(
        self,
        large_resolution_image: np.ndarray
    ):
        """测试超大分辨率图片通过 ImageProcessor 处理"""
        from src.preprocessor.image_processor import ImageProcessor
        
        processor = ImageProcessor(max_input_size=1280, fast_mode=True)
        
        processed = processor.process(large_resolution_image)
        
        h, w = processed.shape[:2]
        
        assert max(h, w) <= 1280, \
            f"处理后图像尺寸应该被缩放到 1280 以内，" \
            f"实际是 {w}x{h}"


class TestPlateDetectorEdgeCases:
    """边界情况测试"""
    
    def test_confidence_threshold(self):
        """测试置信度阈值"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                detector = PlateDetector(
                    confidence_threshold=0.9,
                    use_yolo=False
                )
                
                assert detector.confidence_threshold == 0.9
    
    def test_max_detections(self):
        """测试最大检测数量"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                detector = PlateDetector(
                    max_detections=5,
                    use_yolo=False
                )
                
                assert detector.max_detections == 5
    
    def test_device_setting(self):
        """测试设备设置"""
        with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
            with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
                detector = PlateDetector(
                    device="cuda",
                    use_yolo=False
                )
                
                assert detector.device == "cuda"
    
    def test_detection_result_attributes(
        self,
        mock_detector_with_hyperlpr: PlateDetector,
        normal_plate_image: np.ndarray
    ):
        """测试检测结果的所有属性"""
        detections = mock_detector_with_hyperlpr.detect(normal_plate_image)
        
        if detections:
            det = detections[0]
            
            assert hasattr(det, 'bbox')
            assert hasattr(det, 'confidence')
            assert hasattr(det, 'plate_type')
            assert hasattr(det, 'angle')
            assert hasattr(det, 'corners')
            assert hasattr(det, 'cropped_image')
            assert hasattr(det, 'corrected_image')
            assert hasattr(det, 'plate_text')


class TestPlateDetectionDataclass:
    """PlateDetection 数据类测试"""
    
    def test_plate_detection_creation(self):
        """测试 PlateDetection 创建"""
        detection = PlateDetection(
            bbox=(100, 200, 300, 260),
            confidence=0.95,
            plate_type=PlateType.BLUE,
            angle=0.0,
            corners=None
        )
        
        assert detection.bbox == (100, 200, 300, 260)
        assert detection.confidence == 0.95
        assert detection.plate_type == PlateType.BLUE
        assert detection.angle == 0.0
        assert detection.corners is None
        assert detection.cropped_image is None
        assert detection.corrected_image is None
        assert detection.plate_text is None
    
    def test_plate_detection_with_image(self):
        """测试 PlateDetection 带图像"""
        test_img = np.zeros((60, 200, 3), dtype=np.uint8)
        
        detection = PlateDetection(
            bbox=(100, 200, 300, 260),
            confidence=0.95,
            plate_type=PlateType.BLUE,
            angle=0.0,
            corners=None,
            cropped_image=test_img,
            corrected_image=test_img,
            plate_text="京A12345"
        )
        
        assert detection.cropped_image is not None
        assert detection.corrected_image is not None
        assert detection.plate_text == "京A12345"
