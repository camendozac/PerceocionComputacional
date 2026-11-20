import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Una convolución con kernel_size=patch_size y stride=patch_size equivale a dividir en parches y proyectar
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        # Aplanar dimensiones espaciales y transponer -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x

imagen = torch.randn(2, 3, 224, 224)
patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
tokens_visuales = patch_embed(imagen)

print(f"Secuencia de tokens visuales para Transformer: {tokens_visuales.shape}") 
# Resultado: [2, 196, 768] -> 196 parches de 16x16 proyectados a 768 dimensiones