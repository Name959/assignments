import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side, center_crop

GALLERY_DIR = "gallery_images"
INDEX_FILE = "index_features.npz"
CHECKPOINT_FILE = "index_checkpoint.npz"
WEIGHTS_PATH = "vit-dinov2-base.npz"

# ============ 子进程初始化 ============

_worker_model = None

def init_worker():
    global _worker_model
    weights = np.load(WEIGHTS_PATH)
    _worker_model = Dinov2Numpy(weights)


def process_one_image(img_path):
    try:
        # resize 优先
        try:
            pixel_values = resize_short_side(img_path)
        except Exception:
            pixel_values = center_crop(img_path)

        feat = _worker_model(pixel_values)
        feat = feat / (np.linalg.norm(feat) + 1e-6)

        return img_path, feat

    except Exception as e:
        return img_path, None


# ============ 断点系统 ============

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        data = np.load(CHECKPOINT_FILE, allow_pickle=True)
        features = list(data["features"])
        paths = list(data["paths"])
        print(f"🔄 已恢复进度：{len(paths)} 张图片")
        return features, paths
    return [], []


def save_checkpoint(features, paths):
    np.savez(CHECKPOINT_FILE,
             features=np.array(features),
             paths=np.array(paths))


# ============ 主流程 ============

def main():
    image_paths = sorted(glob.glob(os.path.join(GALLERY_DIR, "*.jpg")))

    if not image_paths:
        print("❌ 没有找到图片")
        return

    # 断点恢复
    all_features, valid_paths = load_checkpoint()
    processed_set = set(valid_paths)

    todo_paths = [p for p in image_paths if p not in processed_set]

    print(f"图库总数: {len(image_paths)}")
    print(f"已处理: {len(valid_paths)}")
    print(f"待处理: {len(todo_paths)}")
    print(f"CPU 核心数: {cpu_count()}")

    if not todo_paths:
        print("✅ 已全部完成，直接生成索引")
    else:
        try:
            with Pool(cpu_count(), initializer=init_worker) as pool:
                for img_path, feat in tqdm(
                        pool.imap_unordered(process_one_image, todo_paths),
                        total=len(todo_paths)):

                    if feat is None:
                        print(f"❌ 跳过损坏图片: {img_path}")
                        continue

                    all_features.append(feat)
                    valid_paths.append(img_path)
                    processed_set.add(img_path)

                    # 每 10 张保存一次断点（避免 IO 太频繁）
                    if len(valid_paths) % 10 == 0:
                        save_checkpoint(all_features, valid_paths)

        except KeyboardInterrupt:
            print("\n⏸️ 用户中断，进度已保存")
            save_checkpoint(all_features, valid_paths)
            return

    # 构建最终索引
    features_matrix = np.vstack(all_features)
    np.savez(INDEX_FILE,
             features=features_matrix,
             paths=np.array(valid_paths))

    # 清理断点文件
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n✅ 索引构建完成: {len(valid_paths)} 张图片")
    print(f"📁 保存至: {INDEX_FILE}")


if __name__ == "__main__":
    main()
