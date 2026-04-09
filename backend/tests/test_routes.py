"""
API 层测试

测试场景：
1. 正常请求
2. 无文件请求
3. 文件格式错误
4. 文件过大
5. 健康检查
6. Base64 识别
"""

import pytest
import numpy as np
import cv2
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector.plate_detector import PlateDetection
from src.utils.constants import PlateType, ResponseCode


class TestHealthEndpoint:
    """健康检查接口测试"""
    
    def test_health_check(self, test_client: TestClient):
        """测试健康检查接口"""
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data
        assert "gpu_available" in data


class TestRecognizeFileEndpoint:
    """文件上传识别接口测试"""
    
    def create_test_image_bytes(self, width: int = 640, height: int = 480) -> bytes:
        """创建测试图像字节"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (200, 200, 200)
        
        cv2.rectangle(img, (200, 200), (440, 280), (255, 100, 0), -1)
        cv2.putText(img, "TEST", (220, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    
    def create_large_image_bytes(self, size_mb: int = 10) -> bytes:
        """创建大图像字节（用于测试文件大小限制）"""
        pixels_per_mb = 1024 * 1024 // 3
        total_pixels = pixels_per_mb * size_mb
        
        side = int(np.sqrt(total_pixels))
        img = np.random.randint(0, 255, (side, side, 3), dtype=np.uint8)
        
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    
    def test_recognize_file_normal(self):
        """测试正常文件上传识别"""
        from src.main import app
        
        mock_detection = MagicMock(spec=PlateDetection)
        mock_detection.bbox = (100, 200, 300, 260)
        mock_detection.confidence = 0.95
        mock_detection.plate_type = PlateType.BLUE
        mock_detection.angle = 0.0
        mock_detection.corners = None
        mock_detection.cropped_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.corrected_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.plate_text = "京A12345"
        
        with patch('src.api.routes.get_detector') as mock_get_detector:
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor') as mock_get_preprocessor:
                    mock_detector = MagicMock()
                    mock_detector.detect.return_value = [mock_detection]
                    mock_get_detector.return_value = mock_detector
                    
                    mock_processor = MagicMock()
                    mock_processor.process = lambda x: x
                    mock_get_preprocessor.return_value = mock_processor
                    
                    client = TestClient(app)
                    
                    image_bytes = self.create_test_image_bytes()
                    
                    response = client.post(
                        "/api/v1/recognize/file",
                        files={"file": ("test.jpg", image_bytes, "image/jpeg")}
                    )
                    
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert "code" in data
                    assert "message" in data
                    assert "data" in data
    
    def test_recognize_file_no_file(self):
        """测试无文件请求"""
        from src.main import app
        
        with patch('src.api.routes.get_detector'):
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor'):
                    client = TestClient(app)
                    
                    response = client.post("/api/v1/recognize/file")
                    
                    assert response.status_code == 422
    
    def test_recognize_file_invalid_format(self):
        """测试文件格式错误"""
        from src.main import app
        
        with patch('src.api.routes.get_detector'):
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor'):
                    client = TestClient(app)
                    
                    response = client.post(
                        "/api/v1/recognize/file",
                        files={"file": ("test.txt", b"not an image", "text/plain")}
                    )
                    
                    assert response.status_code == 400
                    
                    data = response.json()
                    assert "detail" in data
                    assert "不支持的文件格式" in data["detail"]
    
    def test_recognize_file_png_format(self):
        """测试 PNG 格式文件"""
        from src.main import app
        
        mock_detection = MagicMock(spec=PlateDetection)
        mock_detection.bbox = (100, 200, 300, 260)
        mock_detection.confidence = 0.95
        mock_detection.plate_type = PlateType.BLUE
        mock_detection.angle = 0.0
        mock_detection.corners = None
        mock_detection.cropped_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.corrected_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.plate_text = "京A12345"
        
        with patch('src.api.routes.get_detector') as mock_get_detector:
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor') as mock_get_preprocessor:
                    mock_detector = MagicMock()
                    mock_detector.detect.return_value = [mock_detection]
                    mock_get_detector.return_value = mock_detector
                    
                    mock_processor = MagicMock()
                    mock_processor.process = lambda x: x
                    mock_get_preprocessor.return_value = mock_processor
                    
                    client = TestClient(app)
                    
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.png', img)
                    image_bytes = buffer.tobytes()
                    
                    response = client.post(
                        "/api/v1/recognize/file",
                        files={"file": ("test.png", image_bytes, "image/png")}
                    )
                    
                    assert response.status_code == 200
    
    def test_recognize_file_no_plate_detected(self):
        """测试未检测到车牌的情况"""
        from src.main import app
        
        with patch('src.api.routes.get_detector') as mock_get_detector:
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor') as mock_get_preprocessor:
                    mock_detector = MagicMock()
                    mock_detector.detect.return_value = []
                    mock_get_detector.return_value = mock_detector
                    
                    mock_processor = MagicMock()
                    mock_processor.process = lambda x: x
                    mock_get_preprocessor.return_value = mock_processor
                    
                    client = TestClient(app)
                    
                    image_bytes = self.create_test_image_bytes()
                    
                    response = client.post(
                        "/api/v1/recognize/file",
                        files={"file": ("test.jpg", image_bytes, "image/jpeg")}
                    )
                    
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert data["code"] == ResponseCode.NO_PLATE_DETECTED
                    assert data["data"]["plate_count"] == 0
                    assert data["data"]["plates"] == []
    
    def test_recognize_file_invalid_image_data(self):
        """测试无效图像数据"""
        from src.main import app
        
        with patch('src.api.routes.get_detector'):
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor'):
                    client = TestClient(app)
                    
                    invalid_data = b"this is not a valid image"
                    
                    response = client.post(
                        "/api/v1/recognize/file",
                        files={"file": ("test.jpg", invalid_data, "image/jpeg")}
                    )
                    
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert data["code"] == ResponseCode.IMAGE_LOAD_ERROR


class TestRecognizeBase64Endpoint:
    """Base64 识别接口测试"""
    
    def test_recognize_base64_normal(self):
        """测试正常 Base64 识别"""
        from src.main import app
        
        mock_detection = MagicMock(spec=PlateDetection)
        mock_detection.bbox = (100, 200, 300, 260)
        mock_detection.confidence = 0.95
        mock_detection.plate_type = PlateType.BLUE
        mock_detection.angle = 0.0
        mock_detection.corners = None
        mock_detection.cropped_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.corrected_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.plate_text = "京A12345"
        
        with patch('src.api.routes.get_detector') as mock_get_detector:
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor') as mock_get_preprocessor:
                    mock_detector = MagicMock()
                    mock_detector.detect.return_value = [mock_detection]
                    mock_get_detector.return_value = mock_detector
                    
                    mock_processor = MagicMock()
                    mock_processor.process = lambda x: x
                    mock_get_preprocessor.return_value = mock_processor
                    
                    client = TestClient(app)
                    
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', img)
                    base64_data = base64.b64encode(buffer).decode()
                    
                    response = client.post(
                        "/api/v1/recognize/base64",
                        json={
                            "image_data": base64_data,
                            "image_id": "test_001"
                        }
                    )
                    
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert "code" in data
                    assert data["data"]["image_id"] == "test_001"
    
    def test_recognize_base64_invalid_data(self):
        """测试无效 Base64 数据"""
        from src.main import app
        
        with patch('src.api.routes.get_detector'):
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor'):
                    client = TestClient(app)
                    
                    response = client.post(
                        "/api/v1/recognize/base64",
                        json={
                            "image_data": "invalid_base64_data!!!",
                            "image_id": "test_002"
                        }
                    )
                    
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert data["code"] == ResponseCode.INTERNAL_ERROR
    
    def test_recognize_base64_missing_field(self):
        """测试缺少必填字段"""
        from src.main import app
        
        with patch('src.api.routes.get_detector'):
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor'):
                    client = TestClient(app)
                    
                    response = client.post(
                        "/api/v1/recognize/base64",
                        json={
                            "image_id": "test_003"
                        }
                    )
                    
                    assert response.status_code == 422


class TestStatsEndpoint:
    """系统状态接口测试"""
    
    def test_get_stats(self, test_client: TestClient):
        """测试获取系统状态"""
        response = test_client.get("/api/v1/stats")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "cpu_usage_percent" in data
        assert "memory_usage_percent" in data
        assert "memory_available_mb" in data
        assert "uptime_seconds" in data
        assert "uptime_formatted" in data


class TestRootEndpoint:
    """根路径测试"""
    
    def test_root(self, test_client: TestClient):
        """测试根路径"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert "车牌识别系统" in response.text


class TestVideoFormatsEndpoint:
    """视频格式接口测试"""
    
    def test_get_video_formats(self, test_client: TestClient):
        """测试获取支持的视频格式"""
        response = test_client.get("/api/v1/video/formats")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_formats" in data
        assert "supported_codecs" in data
        assert "supported_streams" in data
        assert "max_resolution" in data
        assert "min_resolution" in data
        assert "target_fps_range" in data


class TestBatchRecognition:
    """批量识别接口测试"""
    
    def test_batch_recognition_normal(self):
        """测试正常批量识别"""
        from src.main import app
        
        mock_detection = MagicMock(spec=PlateDetection)
        mock_detection.bbox = (100, 200, 300, 260)
        mock_detection.confidence = 0.95
        mock_detection.plate_type = PlateType.BLUE
        mock_detection.angle = 0.0
        mock_detection.corners = None
        mock_detection.cropped_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.corrected_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.plate_text = "京A12345"
        
        with patch('src.api.routes.get_detector') as mock_get_detector:
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor') as mock_get_preprocessor:
                    mock_detector = MagicMock()
                    mock_detector.detect.return_value = [mock_detection]
                    mock_get_detector.return_value = mock_detector
                    
                    mock_processor = MagicMock()
                    mock_processor.process = lambda x: x
                    mock_get_preprocessor.return_value = mock_processor
                    
                    client = TestClient(app)
                    
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', img)
                    base64_data = base64.b64encode(buffer).decode()
                    
                    response = client.post(
                        "/api/v1/recognize/batch",
                        json={
                            "images": [
                                {"image_data": base64_data, "image_id": "test_1"},
                                {"image_data": base64_data, "image_id": "test_2"}
                            ]
                        }
                    )
                    
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert "total" in data
                    assert "success_count" in data
                    assert "results" in data
                    assert len(data["results"]) == 2


class TestResponseValidation:
    """响应格式验证测试"""
    
    def test_recognition_response_format(self):
        """验证识别响应格式"""
        from src.main import app
        
        mock_detection = MagicMock(spec=PlateDetection)
        mock_detection.bbox = (100, 200, 300, 260)
        mock_detection.confidence = 0.95
        mock_detection.plate_type = PlateType.BLUE
        mock_detection.angle = 0.0
        mock_detection.corners = None
        mock_detection.cropped_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.corrected_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.plate_text = "京A12345"
        
        with patch('src.api.routes.get_detector') as mock_get_detector:
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor') as mock_get_preprocessor:
                    mock_detector = MagicMock()
                    mock_detector.detect.return_value = [mock_detection]
                    mock_get_detector.return_value = mock_detector
                    
                    mock_processor = MagicMock()
                    mock_processor.process = lambda x: x
                    mock_get_preprocessor.return_value = mock_processor
                    
                    client = TestClient(app)
                    
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', img)
                    image_bytes = buffer.tobytes()
                    
                    response = client.post(
                        "/api/v1/recognize/file",
                        files={"file": ("test.jpg", image_bytes, "image/jpeg")}
                    )
                    
                    data = response.json()
                    
                    assert "code" in data
                    assert isinstance(data["code"], int)
                    
                    assert "message" in data
                    assert isinstance(data["message"], str)
                    
                    assert "data" in data
                    assert isinstance(data["data"], dict)
                    
                    if data["code"] == ResponseCode.SUCCESS:
                        assert "image_id" in data["data"]
                        assert "plate_count" in data["data"]
                        assert "plates" in data["data"]
                        assert isinstance(data["data"]["plates"], list)
                        
                        if data["data"]["plates"]:
                            plate = data["data"]["plates"][0]
                            assert "plate_number" in plate
                            assert "confidence" in plate
                            assert "plate_type" in plate
                            assert "bbox" in plate
                            assert "char_results" in plate


class TestQueryParameters:
    """查询参数测试"""
    
    def test_preprocess_query_param(self):
        """测试 preprocess 查询参数"""
        from src.main import app
        
        mock_detection = MagicMock(spec=PlateDetection)
        mock_detection.bbox = (100, 200, 300, 260)
        mock_detection.confidence = 0.95
        mock_detection.plate_type = PlateType.BLUE
        mock_detection.angle = 0.0
        mock_detection.corners = None
        mock_detection.cropped_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.corrected_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.plate_text = "京A12345"
        
        with patch('src.api.routes.get_detector') as mock_get_detector:
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor') as mock_get_preprocessor:
                    mock_detector = MagicMock()
                    mock_detector.detect.return_value = [mock_detection]
                    mock_get_detector.return_value = mock_detector
                    
                    mock_processor = MagicMock()
                    mock_processor.process = lambda x: x
                    mock_get_preprocessor.return_value = mock_processor
                    
                    client = TestClient(app)
                    
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', img)
                    image_bytes = buffer.tobytes()
                    
                    response = client.post(
                        "/api/v1/recognize/file?preprocess=false",
                        files={"file": ("test.jpg", image_bytes, "image/jpeg")}
                    )
                    
                    assert response.status_code == 200
    
    def test_undistort_preset_param(self):
        """测试 undistort_preset 查询参数"""
        from src.main import app
        
        mock_detection = MagicMock(spec=PlateDetection)
        mock_detection.bbox = (100, 200, 300, 260)
        mock_detection.confidence = 0.95
        mock_detection.plate_type = PlateType.BLUE
        mock_detection.angle = 0.0
        mock_detection.corners = None
        mock_detection.cropped_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.corrected_image = np.zeros((60, 200, 3), dtype=np.uint8)
        mock_detection.plate_text = "京A12345"
        
        with patch('src.api.routes.get_detector') as mock_get_detector:
            with patch('src.api.routes.get_recognizer'):
                with patch('src.api.routes.get_preprocessor') as mock_get_preprocessor:
                    mock_detector = MagicMock()
                    mock_detector.detect.return_value = [mock_detection]
                    mock_get_detector.return_value = mock_detector
                    
                    mock_processor = MagicMock()
                    mock_processor.process = lambda x: x
                    mock_get_preprocessor.return_value = mock_processor
                    
                    client = TestClient(app)
                    
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', img)
                    image_bytes = buffer.tobytes()
                    
                    for preset in ["standard", "wide_angle", "fisheye"]:
                        response = client.post(
                            f"/api/v1/recognize/file?undistort_preset={preset}",
                            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
                        )
                        
                        assert response.status_code == 200
