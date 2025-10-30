# ==============================
#  DINOv3 + Hyperbolic Embedding Demo
# ==============================
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import geoopt
from geoopt import PoincareBall
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Load pretrained DINOv3 backbone
# ------------------------------
# (Requires torch hub connection)
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dino.eval()  # inference mode

# ------------------------------
# 2️⃣ Define a hyperbolic projector
# ------------------------------
class HyperbolicProjector(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=256, curvature=1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ball = PoincareBall(c=curvature)
        self.scale = nn.Parameter(torch.tensor(0.1))  # small scale for stability

    def forward(self, x):
        z = self.mlp(x) * self.scale
        # map Euclidean vector to Poincaré ball
        h = self.ball.expmap0(z)
        h = self.ball.projx(h)
        return h

projector = HyperbolicProjector(in_dim=768)
projector.eval()

# ------------------------------
# 3️⃣ Load and preprocess an image
# ------------------------------
# Use any satellite image patch (RGB) around 224x224 pixels.
# Example: "amazon_sentinel2_rgb.jpg"
image_path = "amazon_basin_nasa.jpg"

# DINOv3 expects normalized 224×224 RGB images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

img = Image.open(image_path).convert("RGB")
img_t = transform(img).unsqueeze(0)  # shape (1,3,224,224)

# ------------------------------
# 4️⃣ Extract DINOv3 features
# ------------------------------
with torch.no_grad():
    features = dino(img_t)  # shape (1, 768)

# ------------------------------
# 5️⃣ Map features into hyperbolic space
# ------------------------------
with torch.no_grad():
    hyperbolic_feats = projector(features)  # shape (1, 256)

# ------------------------------
# 6️⃣ Use or visualize embeddings
# ------------------------------
print("Euclidean feature shape:", features.shape)
print("Hyperbolic feature shape:", hyperbolic_feats.shape)
print("Example norm in hyperbolic ball:",
      hyperbolic_feats.norm().item())

ball = PoincareBall(c=1.0)
with torch.no_grad():
    # 20 random 2D points inside the Poincaré ball
    euclidean_points = torch.randn(20, 2) * 0.2
    hyperbolic_points = ball.expmap0(euclidean_points)
    norms = hyperbolic_points.norm(dim=1)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(5, 5))
circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--')
ax.add_artist(circle)

# color points by norm (distance from center)
sc = ax.scatter(hyperbolic_points[:, 0],
                hyperbolic_points[:, 1],
                c=norms,
                cmap='viridis',
                s=80)

plt.colorbar(sc, label='Norm (distance from center)')
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Points inside the Poincaré ball")
plt.show()
