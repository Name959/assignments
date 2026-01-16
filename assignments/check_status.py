import numpy as np
import os
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

def main():
    print("========== 📊 系统状态报告 ==========")

    # -------------------------------------------------
    # 任务 1: 输出向量图库图片数量
    # -------------------------------------------------
    index_file = "index_features.npz"
    if os.path.exists(index_file):
        try:
            data = np.load(index_file)
            # 兼容不同版本的保存格式 (优先用 paths 计数)
            if "paths" in data:
                count = len(data["paths"])
            elif "features" in data:
                count = data["features"].shape[0]
            else:
                count = 0
            print(f"🖼️  向量图库图片数量: {count}")
        except Exception as e:
            print(f"⚠️  无法读取向量库: {e}")
    else:
        print("⚠️  向量库文件不存在 (0 张)")

    # -------------------------------------------------
    # 任务 2: 输出平均标准特征差异
    # -------------------------------------------------
    model_file = "vit-dinov2-base.npz"
    std_feat_file = "demo_data/cat_dog_feature.npy"
    img_cat = "demo_data/cat.jpg"
    img_dog = "demo_data/dog.jpg"

    # 检查必要文件是否存在
    if (os.path.exists(model_file) and 
        os.path.exists(std_feat_file) and 
        os.path.exists(img_cat) and 
        os.path.exists(img_dog)):
        
        try:
            # 加载模型
            weights = np.load(model_file)
            model = Dinov2Numpy(weights)
            
            # 加载标准特征 (NumPy 格式)
            std_features = np.load(std_feat_file)
            
            # 计算当前环境下的特征
            cat_input = center_crop(img_cat)
            dog_input = center_crop(img_dog)
            
            cat_feat = model(cat_input).flatten()
            dog_feat = model(dog_input).flatten()
            
            # 计算绝对误差 (L1 Loss)
            diff_cat = np.mean(np.abs(cat_feat - std_features[0]))
            diff_dog = np.mean(np.abs(dog_feat - std_features[1]))
            
            # 计算平均值
            avg_diff = (diff_cat + diff_dog) / 2
            
            print(f"📉 平均标准特征差异: {avg_diff:.10f}")
            
        except Exception as e:
            print(f"❌ 计算特征差异时出错: {e}")
    else:
        print("⚠️  缺少模型或 Demo 数据，无法计算特征差异。")

    print("=====================================")

if __name__ == "__main__":
    main()