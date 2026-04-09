"""
ImageProcessor 单元测试

测试场景：
1. 预处理流程验证（尺寸、通道数、像素值范围）
2. EXIF 旋转信息处理
3. 各种预处理功能测试
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage, ExifTags
import io

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor.image_processor import ImageProcessor


class TestImageProcessorBasic:
    """ImageProcessor 基础测试"""
    
    def test_initialization_default(self):
        """测试默认初始化"""
        processor = ImageProcessor()
        
        assert processor.denoise is True
        assert processor.contrast_enhance is True
        assert processor.white_balance is True
        assert processor.undistort is False
        assert processor.fast_mode is True
        assert processor.max_input_size == 1280
    
    def test_initialization_custom(self):
        """测试自定义初始化"""
        processor = ImageProcessor(
            denoise=False,
            contrast_enhance=False,
            white_balance=False,
            undistort=True,
            max_input_size=1920,
            fast_mode=False
        )
        
        assert processor.denoise is False
        assert processor.contrast_enhance is False
        assert processor.white_balance is False
        assert processor.undistort is True
        assert processor.fast_mode is False
        assert processor.max_input_size == 1920
    
    def test_initialization_with_camera_preset(self):
        """测试使用相机预设初始化"""
        processor = ImageProcessor(camera_preset="standard")
        
        assert processor.undistort is True
        assert processor.camera_matrix is not None
        assert processor.dist_coeffs is not None


class TestImageProcessorLoadImage:
    """图像加载测试"""
    
    def test_load_image_from_numpy(self, standard_test_image: np.ndarray):
        """测试从 numpy 数组加载图像"""
        processor = ImageProcessor()
        
        result = processor.load_image(standard_test_image)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == standard_test_image.shape
        assert result.dtype == standard_test_image.dtype
    
    def test_load_image_from_bytes(self, image_bytes: bytes):
        """测试从字节数据加载图像"""
        processor = ImageProcessor()
        
        result = processor.load_image(image_bytes)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
    
    def test_load_image_from_file(self, temp_image_file: Path):
        """测试从文件加载图像"""
        processor = ImageProcessor()
        
        result = processor.load_image(str(temp_image_file))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
    
    def test_load_image_nonexistent_file(self):
        """测试加载不存在的文件"""
        processor = ImageProcessor()
        
        result = processor.load_image("/nonexistent/path/image.jpg")
        
        assert result is None
    
    def test_load_image_unsupported_format(self, tmp_path: Path):
        """测试加载不支持的格式"""
        processor = ImageProcessor()
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an image")
        
        result = processor.load_image(str(txt_file))
        
        assert result is None
    
    def test_load_image_invalid_type(self):
        """测试加载无效类型"""
        processor = ImageProcessor()
        
        result = processor.load_image(12345)
        
        assert result is None


class TestImageProcessorPreprocessing:
    """预处理流程测试"""
    
    def test_process_fast_mode(self, standard_test_image: np.ndarray):
        """测试快速模式预处理"""
        processor = ImageProcessor(fast_mode=True)
        
        result = processor.process(standard_test_image)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == standard_test_image.shape
    
    def test_process_full_mode(self, standard_test_image: np.ndarray):
        """测试完整模式预处理"""
        processor = ImageProcessor(
            fast_mode=False,
            denoise=True,
            contrast_enhance=True,
            white_balance=True
        )
        
        result = processor.process(standard_test_image)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
    
    def test_process_output_dimensions(self, standard_test_image: np.ndarray):
        """验证输出图像尺寸"""
        processor = ImageProcessor(fast_mode=True)
        
        result = processor.process(standard_test_image)
        
        h, w = standard_test_image.shape[:2]
        result_h, result_w = result.shape[:2]
        
        assert result_h == h
        assert result_w == w
    
    def test_process_output_channels(self, standard_test_image: np.ndarray):
        """验证输出图像通道数"""
        processor = ImageProcessor(fast_mode=True)
        
        result = processor.process(standard_test_image)
        
        assert len(result.shape) == 3
        assert result.shape[2] == 3
    
    def test_process_pixel_value_range(self, standard_test_image: np.ndarray):
        """验证输出图像像素值范围"""
        processor = ImageProcessor(fast_mode=True)
        
        result = processor.process(standard_test_image)
        
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.dtype == np.uint8
    
    def test_resize_if_needed_large_image(self, large_resolution_image: np.ndarray):
        """测试大图像自动缩放"""
        processor = ImageProcessor(max_input_size=1280, fast_mode=True)
        
        result = processor.process(large_resolution_image)
        
        h, w = result.shape[:2]
        
        assert max(h, w) <= 1280
    
    def test_resize_if_needed_small_image(self, standard_test_image: np.ndarray):
        """测试小图像不缩放"""
        processor = ImageProcessor(max_input_size=1280, fast_mode=True)
        
        result = processor.process(standard_test_image)
        
        assert result.shape == standard_test_image.shape


class TestImageProcessorResize:
    """调整大小功能测试"""
    
    def test_resize_keep_ratio(self, standard_test_image: np.ndarray):
        """测试保持宽高比调整大小"""
        processor = ImageProcessor()
        
        h, w = standard_test_image.shape[:2]
        target_size = (320, 240)
        
        resized, scale = processor.resize(
            standard_test_image,
            target_size=target_size,
            keep_ratio=True
        )
        
        assert isinstance(resized, np.ndarray)
        assert isinstance(scale, float)
        assert scale > 0
    
    def test_resize_no_keep_ratio(self, standard_test_image: np.ndarray):
        """测试不保持宽高比调整大小"""
        processor = ImageProcessor()
        
        target_size = (320, 240)
        
        resized, scale = processor.resize(
            standard_test_image,
            target_size=target_size,
            keep_ratio=False
        )
        
        assert resized.shape[1] == target_size[0]
        assert resized.shape[0] == target_size[1]
        assert scale == 1.0
    
    def test_resize_by_max_size(self, standard_test_image: np.ndarray):
        """测试按最大边长调整大小"""
        processor = ImageProcessor()
        
        resized, scale = processor.resize(
            standard_test_image,
            max_size=320
        )
        
        h, w = resized.shape[:2]
        assert max(h, w) <= 320


class TestImageProcessorRotation:
    """旋转矫正测试"""
    
    def test_correct_rotation_small_angle(self, standard_test_image: np.ndarray):
        """测试小角度旋转（不旋转）"""
        processor = ImageProcessor()
        
        result = processor.correct_rotation(standard_test_image, 0.3)
        
        assert result.shape == standard_test_image.shape
    
    def test_correct_rotation_significant_angle(self, standard_test_image: np.ndarray):
        """测试显著角度旋转"""
        processor = ImageProcessor()
        
        angle = 15.0
        result = processor.correct_rotation(standard_test_image, angle)
        
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3


class TestImageProcessorEXIF:
    """EXIF 旋转信息处理测试"""
    
    def create_exif_rotated_image(self, orientation: int) -> bytes:
        """创建带有 EXIF 旋转信息的图像"""
        img = PILImage.new('RGB', (480, 640), color='gray')
        
        exif = img.getexif()
        for tag in ExifTags.TAGS.keys():
            if ExifTags.TAGS[tag] == 'Orientation':
                exif[tag] = orientation
                break
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', exif=exif)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def test_exif_orientation_1(self):
        """测试 EXIF Orientation = 1（正常）"""
        image_bytes = self.create_exif_rotated_image(1)
        
        processor = ImageProcessor()
        result = processor.load_image(image_bytes)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_exif_orientation_6(self):
        """测试 EXIF Orientation = 6（顺时针旋转90度）"""
        image_bytes = self.create_exif_rotated_image(6)
        
        processor = ImageProcessor()
        result = processor.load_image(image_bytes)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_exif_orientation_3(self):
        """测试 EXIF Orientation = 3（旋转180度）"""
        image_bytes = self.create_exif_rotated_image(3)
        
        processor = ImageProcessor()
        result = processor.load_image(image_bytes)
        
        assert result is not None
    
    def test_exif_orientation_8(self):
        """测试 EXIF Orientation = 8（逆时针旋转90度）"""
        image_bytes = self.create_exif_rotated_image(8)
        
        processor = ImageProcessor()
        result = processor.load_image(image_bytes)
        
        assert result is not None


class TestImageProcessorDenoise:
    """降噪功能测试"""
    
    def test_denoise_function(self, standard_test_image: np.ndarray):
        """测试降噪功能"""
        processor = ImageProcessor()
        
        noisy_image = standard_test_image.copy()
        noise = np.random.normal(0, 25, noisy_image.shape).astype(np.int16)
        noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        denoised = processor._denoise(noisy_image)
        
        assert isinstance(denoised, np.ndarray)
        assert denoised.shape == noisy_image.shape
        assert denoised.dtype == np.uint8


class TestImageProcessorContrast:
    """对比度增强测试"""
    
    def test_enhance_contrast_function(self, standard_test_image: np.ndarray):
        """测试对比度增强功能"""
        processor = ImageProcessor(contrast_alpha=1.5)
        
        enhanced = processor._enhance_contrast(standard_test_image)
        
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == standard_test_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_fast_enhance_contrast(self, standard_test_image: np.ndarray):
        """测试快速对比度增强"""
        processor = ImageProcessor()
        
        enhanced = processor._fast_enhance_contrast(standard_test_image)
        
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == standard_test_image.shape


class TestImageProcessorWhiteBalance:
    """白平衡测试"""
    
    def test_white_balance_function(self, standard_test_image: np.ndarray):
        """测试白平衡功能"""
        processor = ImageProcessor()
        
        balanced = processor._white_balance(standard_test_image)
        
        assert isinstance(balanced, np.ndarray)
        assert balanced.shape == standard_test_image.shape
        assert balanced.dtype == np.uint8
        
        assert balanced.min() >= 0
        assert balanced.max() <= 255


class TestImageProcessorUndistort:
    """畸变矫正测试"""
    
    def test_undistort_with_preset(self, standard_test_image: np.ndarray):
        """测试使用预设进行畸变矫正"""
        processor = ImageProcessor(camera_preset="standard")
        
        undistorted = processor._undistort(standard_test_image)
        
        assert isinstance(undistorted, np.ndarray)
        assert len(undistorted.shape) == 3
    
    def test_undistort_without_params(self, standard_test_image: np.ndarray):
        """测试无参数时畸变矫正"""
        processor = ImageProcessor(undistort=False)
        processor.camera_matrix = None
        processor.dist_coeffs = None
        
        undistorted = processor._undistort(standard_test_image)
        
        assert undistorted is standard_test_image


class TestImageProcessorCropPlate:
    """车牌裁剪测试"""
    
    def test_crop_plate_function(self, standard_test_image: np.ndarray):
        """测试车牌裁剪功能"""
        processor = ImageProcessor()
        
        h, w = standard_test_image.shape[:2]
        bbox = (int(w*0.2), int(h*0.3), int(w*0.8), int(h*0.6))
        
        cropped = processor.crop_plate(standard_test_image, bbox)
        
        assert isinstance(cropped, np.ndarray)
        assert len(cropped.shape) == 3
        
        crop_h, crop_w = cropped.shape[:2]
        expected_w = bbox[2] - bbox[0]
        expected_h = bbox[3] - bbox[1]
        
        assert crop_w >= expected_w
        assert crop_h >= expected_h
    
    def test_crop_plate_with_padding(self, standard_test_image: np.ndarray):
        """测试带 padding 的车牌裁剪"""
        processor = ImageProcessor()
        
        h, w = standard_test_image.shape[:2]
        bbox = (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
        
        cropped = processor.crop_plate(standard_test_image, bbox, padding=0.1)
        
        assert isinstance(cropped, np.ndarray)


class TestImageProcessorPerspective:
    """透视变换测试"""
    
    def test_perspective_transform(self, standard_test_image: np.ndarray):
        """测试透视变换功能"""
        processor = ImageProcessor()
        
        h, w = standard_test_image.shape[:2]
        
        src_points = np.array([
            [w*0.1, h*0.1],
            [w*0.9, h*0.1],
            [w*0.9, h*0.9],
            [w*0.1, h*0.9]
        ], dtype=np.float32)
        
        transformed = processor.perspective_transform(
            standard_test_image,
            src_points,
            dst_size=(440, 140)
        )
        
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[1] == 440
        assert transformed.shape[0] == 140


class TestImageProcessorStaticMethods:
    """静态方法测试"""
    
    def test_to_gray_color(self, standard_test_image: np.ndarray):
        """测试彩色图像转灰度"""
        gray = ImageProcessor.to_gray(standard_test_image)
        
        assert isinstance(gray, np.ndarray)
        assert len(gray.shape) == 2
    
    def test_to_gray_already_gray(self, standard_test_image: np.ndarray):
        """测试已经是灰度图的情况"""
        gray = cv2.cvtColor(standard_test_image, cv2.COLOR_BGR2GRAY)
        result = ImageProcessor.to_gray(gray)
        
        assert result is gray
    
    def test_to_binary_adaptive(self, standard_test_image: np.ndarray):
        """测试自适应二值化"""
        binary = ImageProcessor.to_binary(standard_test_image, adaptive=True)
        
        assert isinstance(binary, np.ndarray)
        assert len(binary.shape) == 2
        
        unique_values = np.unique(binary)
        assert all(v in [0, 255] for v in unique_values)
    
    def test_to_binary_otsu(self, standard_test_image: np.ndarray):
        """测试 Otsu 二值化"""
        binary = ImageProcessor.to_binary(standard_test_image, adaptive=False, threshold=0)
        
        assert isinstance(binary, np.ndarray)
        assert len(binary.shape) == 2


class TestImageProcessorEdgeCases:
    """边界情况测试"""
    
    def test_process_empty_image(self):
        """测试处理空图像"""
        processor = ImageProcessor()
        
        empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
        
        result = processor.process(empty_image)
        
        assert isinstance(result, np.ndarray)
        assert result.size == 0
    
    def test_process_single_channel(self):
        """测试处理单通道图像"""
        processor = ImageProcessor()
        
        single_channel = np.zeros((100, 100), dtype=np.uint8)
        
        with patch.object(processor.logger, 'warning'):
            result = processor.process(single_channel)
        
        assert isinstance(result, np.ndarray)
    
    def test_load_image_from_pathlib(self, temp_image_file: Path):
        """测试从 Path 对象加载图像"""
        processor = ImageProcessor()
        
        result = processor.load_image(temp_image_file)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
