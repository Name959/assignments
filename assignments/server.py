import numpy as np
import shutil
import os
import tempfile
import time  # [新增] 用于计时
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from contextlib import asynccontextmanager
# [新增] Pydantic 用于接收 JSON 请求体
from pydantic import BaseModel

# 导入预处理函数 (确保这两个文件在同一目录下)
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop, resize_short_side

# 配置路径
GALLERY_DIR = "gallery_images"
INDEX_FILE = "index_features.npz"
MODEL_FILE = "vit-dinov2-base.npz"

# 全局变量
model = None
index_features = None
index_paths = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, index_features, index_paths
    
    # 1. 加载模型
    print(f"Loading model from {MODEL_FILE}...")
    if os.path.exists(MODEL_FILE):
        try:
            weights = np.load(MODEL_FILE)
            model = Dinov2Numpy(weights)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️ Warning: Model file '{MODEL_FILE}' not found.")
    
    # 2. 加载索引
    print("Loading index...")
    if os.path.exists(INDEX_FILE):
        try:
            data = np.load(INDEX_FILE, allow_pickle=True)
            index_features = data["features"]
            index_paths = data["paths"]
            
            print("Normalizing index features...")
            norm = np.linalg.norm(index_features, axis=1, keepdims=True)
            index_features = index_features / (norm + 1e-6)
            
            print(f"✅ Index loaded with {len(index_paths)} images.")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
    else:
        print(f"⚠️ Warning: Index file '{INDEX_FILE}' not found. Please run build_index.py.")
    
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(GALLERY_DIR, exist_ok=True)
app.mount("/gallery_images", StaticFiles(directory=GALLERY_DIR), name="gallery")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: index.html not found</h1>")

# ==========================================
# [重构] 核心搜索逻辑提取
# ==========================================
def core_search_logic(image_path_for_inference):
    """
    输入一个本地图片路径，执行推理和比对，返回 Top 10 结果和调试信息
    """
    # 1. 验证图片
    try:
        with Image.open(image_path_for_inference) as img:
            img.verify()
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. 智能预处理策略 (Fallback Mechanism)
    query_feat = None
    used_method = "resize"
    try:
        # 方案 A: 优先尝试 Resize
        img_tensor = resize_short_side(image_path_for_inference)
        query_feat = model(img_tensor)
    except Exception as e:
        print(f"⚠️ Resize inference failed: {e}. Switching to Center Crop.")
        # 方案 B: 降级使用 Center Crop
        used_method = "crop"
        img_tensor = center_crop(image_path_for_inference)
        query_feat = model(img_tensor)

    # 3. 归一化查询向量
    query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-6)
    
    # 4. 计算相似度 (确保 query_feat 展平为 (768,))
    scores = index_features @ query_feat.flatten()
    
    # 5. 获取 Top 15
    top_k = min(15, len(scores))
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        full_path = str(index_paths[idx])
        # 确保路径分隔符在不同系统下统一
        full_path = full_path.replace("\\", "/")
        # 移除可能存在的冗余前缀，确保 url 是 /gallery_images/xxx.jpg 格式
        if full_path.startswith(GALLERY_DIR + "/"):
             cleaned_path = full_path
        elif full_path.startswith(GALLERY_DIR):
             cleaned_path = full_path.replace(GALLERY_DIR, GALLERY_DIR + "/")
        else:
             # 处理旧索引可能只存文件名的情况
             cleaned_path = f"{GALLERY_DIR}/{os.path.basename(full_path)}"

        score = float(scores[idx])
        results.append({
            "url": f"/{cleaned_path}",
            "score": score
        })
        
    return results, used_method

# ==========================================
# 接口 1: 上传文件搜索
# ==========================================
@app.post("/search")
async def search_image_upload(file: UploadFile = File(...)):
    # 检查服务状态
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded.")
    if index_features is None: raise HTTPException(status_code=503, detail="Index empty.")
    
    start_time = time.time() # [新增] 开始计时

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        # 调用核心逻辑
        results, used_method = core_search_logic(temp_path)
        
        end_time = time.time() # [新增] 结束计时
        duration_ms = round((end_time - start_time) * 1000, 2)

        # [修改] 返回结构增加统计信息
        return {
            "results": results,
            "stats": {
                "duration_ms": duration_ms,
                "total_indexed": len(index_paths),
                "method_used": used_method
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Search critical error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

# ==========================================
# [新增] 接口 2: 通过现有路径搜索 ("找相似")
# ==========================================
# 定义请求体模型
class PathSearchRequest(BaseModel):
    image_path: str

@app.post("/search_by_path")
async def search_image_path(request: PathSearchRequest):
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded.")
    if index_features is None: raise HTTPException(status_code=503, detail="Index empty.")

    start_time = time.time()
    
    # 处理前端传来的路径 (例如 "/gallery_images/123.jpg")
    # 去掉开头的 "/" 以便在本地文件系统查找
    relative_path = request.image_path.lstrip("/")
    
    # 安全检查：确保路径试图访问 gallery 目录内部
    if not os.path.abspath(relative_path).startswith(os.path.abspath(GALLERY_DIR)):
         raise HTTPException(status_code=403, detail="Access denied to non-gallery path.")

    if not os.path.exists(relative_path):
        raise HTTPException(status_code=404, detail=f"Image path not found on server: {relative_path}")

    try:
        # 直接调用核心逻辑，传入服务器上的现有路径
        results, used_method = core_search_logic(relative_path)
        
        end_time = time.time()
        duration_ms = round((end_time - start_time) * 1000, 2)

        return {
            "results": results,
            "stats": {
                "duration_ms": duration_ms,
                "total_indexed": len(index_paths),
                "method_used": used_method
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Path search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # 使用 8000 端口
    print("🚀 Server starting at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)