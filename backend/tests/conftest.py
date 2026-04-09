"""
测试配置和 Fixture
"""

import pytest
import numpy as np
import cv2
import io
import base64
from pathlib import Path
from typing import Generator, Tuple, List, Dict, Any
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class MockDetection:
    """模拟 YOLO 检测结果"""
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


@dataclass
class MockYOLOResult:
    """模拟 YOLO 推理结果"""
    boxes: MockDetection


@pytest.fixture
def test_image_path() -> Path:
    """测试图片路径 fixture"""
    samples_dir = Path(__file__).parent / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir


@pytest.fixture
def normal_plate_image() -> np.ndarray:
    """正常车牌图像 fixture - 模拟包含车牌的图像"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    
    cv2.rectangle(img, (200, 200), (440, 280), (255, 100, 0), -1)
    cv2.putText(img, "JINGA12345", (210, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    return img


@pytest.fixture
def blurry_image() -> np.ndarray:
    """模糊图像 fixture"""
    img = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (25, 25), 0)
    return img


@pytest.fixture
def empty_image() -> np.ndarray:
    """空图像 fixture - 纯黑/纯白无内容"""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def multi_plate_image() -> np.ndarray:
    """多车牌同框图像 fixture"""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (180, 180, 180)
    
    cv2.rectangle(img, (100, 150), (340, 230), (255, 100, 0), -1)
    cv2.putText(img, "JINGA12345", (110, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.rectangle(img, (450, 300), (690, 380), (0, 200, 100), -1)
    cv2.putText(img, "HUB67890", (460, 355), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.rectangle(img, (200, 420), (440, 500), (0, 100, 255), -1)
    cv2.putText(img, "YUEC11111", (210, 475), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return img


@pytest.fixture
def large_resolution_image() -> np.ndarray:
    """超大分辨率图像 fixture (4000x3000)"""
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    
    cv2.rectangle(img, (1800, 1300), (2200, 1450), (255, 100, 0), -1)
    cv2.putText(img, "JINGA12345", (1820, 1410), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
    
    return img


@pytest.fixture
def standard_test_image() -> np.ndarray:
    """标准测试图像 - 用于预处理测试"""
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    return img


@pytest.fixture
def mock_yolo_model() -> MagicMock:
    """Mock YOLO 模型 fixture"""
    mock_model = MagicMock()
    
    def mock_predict(image, conf=None, iou=None, max_det=None, verbose=None):
        if isinstance(image, np.ndarray) and image.size > 0:
            h, w = image.shape[:2]
            
            mock_boxes = MockDetection(
                xyxy=np.array([[w*0.3, h*0.4, w*0.7, h*0.55]], dtype=np.float32),
                conf=np.array([0.95], dtype=np.float32),
                cls=np.array([0], dtype=np.int32)
            )
            return [MockYOLOResult(boxes=mock_boxes)]
        return [MockYOLOResult(boxes=None)]
    
    mock_model.__call__ = mock_predict
    mock_model.to = MagicMock(return_value=mock_model)
    
    return mock_model


@pytest.fixture
def mock_hyperlpr_catcher() -> MagicMock:
    """Mock HyperLPR3 catcher fixture"""
    mock_catcher = MagicMock()
    
    def mock_detect(image):
        if isinstance(image, np.ndarray) and image.size > 0:
            h, w = image.shape[:2]
            return [
                ("京A12345", 0.98, 0, [int(w*0.3), int(h*0.4), int(w*0.7), int(h*0.55)]),
            ]
        return []
    
    mock_catcher.__call__ = mock_detect
    return mock_catcher


@pytest.fixture
def mock_multi_plate_hyperlpr() -> MagicMock:
    """Mock 多车牌 HyperLPR3 结果"""
    mock_catcher = MagicMock()
    
    def mock_detect(image):
        if isinstance(image, np.ndarray) and image.size > 0:
            h, w = image.shape[:2]
            return [
                ("京A12345", 0.98, 0, [int(w*0.125), int(h*0.25), int(w*0.425), int(h*0.383)]),
                ("沪B67890", 0.85, 0, [int(w*0.562), int(h*0.5), int(w*0.862), int(h*0.633)]),
                ("粤C11111", 0.92, 0, [int(w*0.25), int(h*0.7), int(w*0.55), int(h*0.833)]),
            ]
        return []
    
    mock_catcher.__call__ = mock_detect
    return mock_catcher


@pytest.fixture
def mock_paddleocr() -> MagicMock:
    """Mock PaddleOCR fixture"""
    mock_ocr = MagicMock()
    
    def mock_ocr(image, cls=False):
        if isinstance(image, np.ndarray) and image.size > 0:
            return [[[
                [[10, 10], [100, 10], [100, 40], [10, 40]],
                ("京A12345", 0.95)
            ]]]
        return [[]]
    
    mock_ocr.ocr = mock_ocr
    return mock_ocr


@pytest.fixture
def temp_image_file(tmp_path: Path, normal_plate_image: np.ndarray) -> Path:
    """临时图像文件 fixture"""
    file_path = tmp_path / "test_plate.jpg"
    cv2.imwrite(str(file_path), normal_plate_image)
    return file_path


@pytest.fixture
def temp_large_image_file(tmp_path: Path, large_resolution_image: np.ndarray) -> Path:
    """临时大图像文件 fixture"""
    file_path = tmp_path / "large_plate.jpg"
    cv2.imwrite(str(file_path), large_resolution_image)
    return file_path


@pytest.fixture
def image_bytes(normal_plate_image: np.ndarray) -> bytes:
    """图像字节数据 fixture"""
    _, buffer = cv2.imencode('.jpg', normal_plate_image)
    return buffer.tobytes()


@pytest.fixture
def image_base64(image_bytes: bytes) -> str:
    """Base64 编码图像 fixture"""
    return base64.b64encode(image_bytes).decode('utf-8')


@pytest.fixture
def non_image_file(tmp_path: Path) -> Path:
    """非图像文件 fixture"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is not an image file")
    return file_path


@pytest.fixture
def exif_rotated_image_bytes() -> bytes:
    """
    带有 EXIF 旋转信息的图像字节
    模拟手机竖拍照片（Orientation = 6 表示顺时针旋转90度）
    """
    from PIL import Image as PILImage, ExifTags
    
    img = PILImage.new('RGB', (480, 640), color='gray')
    
    exif = img.getexif()
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            exif[orientation] = 6
            break
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', exif=exif)
    buffer.seek(0)
    
    return buffer.getvalue()


@pytest.fixture
def mock_detector_with_hyperlpr(mock_hyperlpr_catcher: MagicMock) -> Generator:
    """使用 mock HyperLPR3 的检测器 fixture"""
    with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr') as mock_init:
        with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
            from src.detector.plate_detector import PlateDetector
            
            detector = PlateDetector(use_yolo=False)
            detector.hyperlpr_available = True
            detector.hyperlpr_catcher = mock_hyperlpr_catcher
            detector.yolo_plate_model_available = False
            
            yield detector


@pytest.fixture
def mock_detector_with_multi_plate(mock_multi_plate_hyperlpr: MagicMock) -> Generator:
    """使用 mock 多车牌 HyperLPR3 的检测器 fixture"""
    with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
        with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
            from src.detector.plate_detector import PlateDetector
            
            detector = PlateDetector(use_yolo=False)
            detector.hyperlpr_available = True
            detector.hyperlpr_catcher = mock_multi_plate_hyperlpr
            detector.yolo_plate_model_available = False
            
            yield detector


@pytest.fixture
def mock_detector_with_yolo(mock_yolo_model: MagicMock) -> Generator:
    """使用 mock YOLO 的检测器 fixture"""
    with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
        with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
            from src.detector.plate_detector import PlateDetector
            
            detector = PlateDetector(use_yolo=True)
            detector.hyperlpr_available = False
            detector.yolo_plate_model_available = True
            detector.yolo_model = mock_yolo_model
            
            yield detector


@pytest.fixture
def mock_detector_no_models() -> Generator:
    """没有可用模型的检测器 fixture（使用传统 CV 方法）"""
    with patch('src.detector.plate_detector.PlateDetector._init_hyperlpr'):
        with patch('src.detector.plate_detector.PlateDetector._init_yolo_plate_model'):
            from src.detector.plate_detector import PlateDetector
            
            detector = PlateDetector(use_yolo=False)
            detector.hyperlpr_available = False
            detector.yolo_plate_model_available = False
            
            yield detector


@pytest.fixture
def test_client() -> Generator:
    """FastAPI TestClient fixture"""
    from fastapi.testclient import TestClient
    from src.main import app
    
    with patch('src.api.routes.get_detector') as mock_get_detector:
        with patch('src.api.routes.get_recognizer'):
            with patch('src.api.routes.get_preprocessor'):
                mock_detector = MagicMock()
                mock_detector.detect.return_value = []
                mock_get_detector.return_value = mock_detector
                
                client = TestClient(app)
                yield client


@pytest.fixture
def mock_memory_tracker() -> Generator:
    """内存追踪器 fixture - 用于 OOM 测试"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    class MemoryTracker:
        def __init__(self):
            self.initial = initial_memory
            self.peaks = []
        
        def get_current(self):
            return process.memory_info().rss / 1024 / 1024
        
        def check_oom(self, max_increase_mb: float = 500) -> bool:
            current = self.get_current()
            increase = current - self.initial
            self.peaks.append(current)
            return increase < max_increase_mb
    
    yield MemoryTracker()
