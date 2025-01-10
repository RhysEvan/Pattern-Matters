import torch
import torch.nn as nn

# Helper to divide image into patches
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(480, 640), patch_size=16, in_chans=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Linear embedding for patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Convert input image to patch embeddings
        x = x.flatten(2).transpose(1, 2)  # Rearrange to (B, num_patches, embed_dim)
        return x

# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head self-attention
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        # Feedforward MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

# Vision Transformer (ViT) Encoder
class ViTEncoder(nn.Module):
    def __init__(self, img_size=(480, 640), patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072):
        super(ViTEncoder, self).__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        # Positional Encoding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])

    def forward(self, x):
        # Patch embedding + position embedding
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.dropout(x)

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        return x

# U-Net Style Decoder to Upsample to Original Image Size
class ViTDecoder(nn.Module):
    def __init__(self, embed_dim, img_size=(480, 640), patch_size=16, out_channels=1):
        super(ViTDecoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        # Linear layer to map transformer output to image space
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x):
        B, N, _ = x.shape
        H, W = self.grid_size

        # Project transformer output back to image patches
        x = self.proj(x)
        x = x.view(B, H, W, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, -1, H * self.patch_size, W * self.patch_size)  # Reshape to (B, C, H, W)

        return x

# Transformer-based Depth Estimation Network
class DepthEstimationTransformer(nn.Module):
    def __init__(self, img_size=(480, 640), 
                 patch_size=16, embed_dim=768, 
                 depth=12, num_heads=12, 
                 mlp_dim=3072, out_channels=1,
                 bilinear=False, prepadding=4, feature_base=64,
                 expansive_path=True, kernel_size=3, scale_factor=2):
        super(DepthEstimationTransformer, self).__init__()
        self.prepadding = prepadding

        # ViT encoder
        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_dim=mlp_dim)

        self.decoder = ViTDecoder(embed_dim, img_size=img_size, patch_size=patch_size, out_channels=out_channels)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encode input with transformer
        x = self.encoder(x)

        # Decode to produce depth map
        x = self.decoder(x)

        depth_map = self.sigmoid(x)

        # Ensure depth map matches input resolution
        return depth_map