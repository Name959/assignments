import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

# 配置
CSV_PATH = "data.csv"         # 你的CSV文件路径
SAVE_DIR = "gallery_images"   # 图片保存文件夹
URL_COLUMN = "image_url"            # CSV中存放链接的列名，请根据实际情况修改
MAX_WORKERS = 16              # 线程数

def download_image(args):
    idx, url = args
    save_path = Path(SAVE_DIR) / f"{idx}.jpg"
    
    if save_path.exists():
        return # 跳过已存在的

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 读取CSV
    # 假设CSV没有表头，如果是第一列，用 iloc[:, 0]
    # 如果有表头，请修改下面的逻辑
    try:
        df = pd.read_csv(CSV_PATH)
        urls = df[URL_COLUMN].tolist() 
    except KeyError:
        print(f"错误: CSV中未找到列名 '{URL_COLUMN}'，请检查data.csv")
        return

    print(f"准备下载 {len(urls)} 张图片...")

    # 多线程下载
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(download_image, enumerate(urls)), 
            total=len(urls), 
            unit="img"
        ))

    print("下载完成。")

if __name__ == "__main__":
    main()