# dinov2_numpy_optimized.py
import numpy as np
from scipy.ndimage import zoom

def gelu(x):
    """更精确的GELU实现"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

class Embeddings:
    def __init__(self, weights):
        self.hidden_size = 768
        self.patch_size = 14
        
        # 确保使用正确的数据类型
        self.cls_token = weights["embeddings.cls_token"].astype(np.float32)
        self.position_embeddings = weights["embeddings.position_embeddings"].astype(np.float32)
        
        patch_weight = weights["embeddings.patch_embeddings.projection.weight"].astype(np.float32)
        patch_bias = weights["embeddings.patch_embeddings.projection.bias"].astype(np.float32)
        
        # 精确的权重重塑
        self.patch_embed_w = patch_weight.reshape(self.hidden_size, -1).T
        self.patch_embed_b = patch_bias.reshape(1, 1, -1)

    def pixel2patches(self, pixel_values): 
        B, C, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        patches = []
        for i in range(0, H, self.patch_size):
            for j in range(0, W, self.patch_size):
                patch = pixel_values[:, :, i:i+self.patch_size, j:j+self.patch_size].reshape(B, -1)
                patches.append(patch)

        patches = np.stack(patches, axis=1)
        return patches

    def interpolate_pos_encoding(self, embeddings, height, width):
        B, N, D = embeddings.shape
        num_patches = N - 1
        
        orig_num_patches = self.position_embeddings.shape[1] - 1
        orig_grid_size = int(np.sqrt(orig_num_patches))
        new_grid_size = int(np.sqrt(num_patches))
        
        if orig_grid_size == new_grid_size:
            return self.position_embeddings[:, :N, :]
        
        patch_pos_embed = self.position_embeddings[:, 1:, :]
        patch_pos_embed_2d = patch_pos_embed.reshape(1, orig_grid_size, orig_grid_size, D)
        
        zoom_factors = (1, new_grid_size/orig_grid_size, new_grid_size/orig_grid_size, 1)
        interpolated = zoom(patch_pos_embed_2d, zoom_factors, order=1)
        
        interpolated = interpolated.reshape(1, num_patches, D)
        
        cls_pos_embed = self.position_embeddings[:, :1, :]
        new_pos_embed = np.concatenate([cls_pos_embed, interpolated], axis=1)
        
        return new_pos_embed[:, :N, :]
    
    def __call__(self, pixel_values):
        B, _, H, W = pixel_values.shape

        patch_values = self.pixel2patches(pixel_values)
        embeddings = patch_values @ self.patch_embed_w + self.patch_embed_b
        
        cls_token = np.tile(self.cls_token, (B, 1, 1))
        embeddings = np.concatenate([cls_token, embeddings], axis=1)

        pos_embed = self.interpolate_pos_encoding(embeddings, H, W)
        embeddings = embeddings + pos_embed
        
        return embeddings

class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        # 使用与PyTorch相同的epsilon值
        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.eps = eps

    def __call__(self, x):
        u = np.mean(x, axis=-1, keepdims=True)
        s = np.var(x, axis=-1, keepdims=True)  # 使用var而不是mean(square(x-u))
        x = (x - u) / np.sqrt(s + self.eps)
        return self.weight * x + self.bias

class LayerScale:
    def __init__(self, lambda1):
        self.lambda1 = lambda1.astype(np.float32).reshape(1, 1, -1)

    def __call__(self, x):
        return x * self.lambda1

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32)

    def __call__(self, x):
        return x @ self.weight.T + self.bias

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads
        self.hidden_size = config['hidden_size']

        # 确保使用正确的数据类型
        q_w = weights[f"{prefix}.attention.attention.query.weight"].astype(np.float32)
        q_b = weights[f"{prefix}.attention.attention.query.bias"].astype(np.float32)
        k_w = weights[f"{prefix}.attention.attention.key.weight"].astype(np.float32)
        k_b = weights[f"{prefix}.attention.attention.key.bias"].astype(np.float32)
        v_w = weights[f"{prefix}.attention.attention.value.weight"].astype(np.float32)
        v_b = weights[f"{prefix}.attention.attention.value.bias"].astype(np.float32)
        o_w = weights[f"{prefix}.attention.output.dense.weight"].astype(np.float32)
        o_b = weights[f"{prefix}.attention.output.dense.bias"].astype(np.float32)

        self.q_proj = Linear(q_w, q_b)
        self.k_proj = Linear(k_w, k_b)
        self.v_proj = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        B, N, D = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 使用双精度计算点积，然后转换回单精度
        att_scores = (q.astype(np.float64) @ k.transpose(0, 1, 3, 2).astype(np.float64)) / np.sqrt(float(self.head_dim))
        
        # 数值稳定的softmax
        att_scores_max = np.max(att_scores, axis=-1, keepdims=True)
        exp_scores = np.exp(att_scores - att_scores_max)
        att_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 应用注意力权重
        att_output = att_weights.astype(np.float32) @ v
        att_output = att_output.transpose(0, 2, 1, 3).reshape(B, N, D)
        
        output = self.out_proj(att_output)
        return output

class MLP:
    def __init__(self, prefix, weights):
        w1 = weights[f"{prefix}.mlp.fc1.weight"].astype(np.float32)
        b1 = weights[f"{prefix}.mlp.fc1.bias"].astype(np.float32)
        w2 = weights[f"{prefix}.mlp.fc2.weight"].astype(np.float32)
        b2 = weights[f"{prefix}.mlp.fc2.bias"].astype(np.float32)

        self.fc1 = Linear(w1, b1)
        self.fc2 = Linear(w2, b2)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

class TransformerBlock:
    def __init__(self, config, idx, weights):
        prefix = f"encoder.layer.{idx}"
        
        self.norm1 = LayerNorm(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
        self.scale1 = LayerScale(weights[f"{prefix}.layer_scale1.lambda1"])
        self.attn = MultiHeadAttention(config, prefix, weights)

        self.norm2 = LayerNorm(weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
        self.scale2 = LayerScale(weights[f"{prefix}.layer_scale2.lambda1"])
        self.mlp = MLP(prefix, weights)

    def __call__(self, x):
        # 注意力残差块
        attn_output = self.attn(self.norm1(x))
        scaled_attn = self.scale1(attn_output)
        x = x + scaled_attn
        
        # MLP残差块
        mlp_output = self.mlp(self.norm2(x))
        scaled_mlp = self.scale2(mlp_output)
        x = x + scaled_mlp
        
        return x

class Dinov2Numpy:
    def __init__(self, weights, config=None):
        self.weights = weights
        self.config = config or {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "patch_size": 14,
        }

        self.embeddings = Embeddings(weights)
        self.blocks = [TransformerBlock(self.config, i, weights) for i in range(self.config["num_layers"])]
        self.norm = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, pixel_values):
        x = self.embeddings(pixel_values)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]