import numpy as np
from PIL import Image

def resize_short_side(img_path, target_size=224):
    """
    保留比例缩放短边，并确保长宽都是 14 的倍数 (DINOv2 patch size)
    """
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    
    # 计算缩放比例
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    
    # 调整到 14 的倍数 (DINOv2 的硬性要求)
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14
    
    # DINOv2 官方建议使用 BICUBIC 插值
    image = image.resize((new_w, new_h), Image.BICUBIC)
    
    img_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_np = (img_np - mean) / std
    return img_np.transpose(2, 0, 1)[None] # (1, 3, H, W)

def center_crop(img_path, crop_size=224):
    """标准的中心裁剪"""
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    
    img_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    return img_np.transpose(2, 0, 1)[None]