import torch
from PIL import Image
from torchvision import transforms
from dinov3.models.vision_transformer import vit_large
import netCDF4 as nc
import numpy as np
import os
import re
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math

# Path to your model weights
weights_path = "dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

# Create model
model = vit_large(patch_size=16, num_classes=0)
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

print("✅ DINOv3 ViT-L/16 (SAT-493M) loaded successfully!")

# Preprocessing (same normalization used in training)
transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



########

lat_min, lat_max = -15, 5
lon_min, lon_max = -75, -50

data_dir = "/Users/carmenoliver/Desktop/SIF_anomalies/SIF_DATA_TROPOMI/"

files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
print("Files found:", files)


sif_amazon = []
time_list = []

data_dir = data_dir = "/Users/carmenoliver/Desktop/SIF_anomalies/SIF_DATA_TROPOMI/"

# Get all available files
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
print("Files found:", files)
for file in files:
    match = re.search(r"month-(\d{6})", file)
    if match:
        date_str = match.group(1)  
        year = int(date_str[:4])  
        month = int(date_str[4:6])  

        file_path = os.path.join(data_dir, file)
        
        try:
            ds = xr.open_dataset(file_path)
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            continue
        ds_amazon = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
        
        sif_amazon.append(ds_amazon["solar_induced_fluorescence"].values)  
        time_list.append(f"{year}-{month:02d}")
        # Ensure data was loaded
if not sif_amazon or not time_list:
    raise ValueError("No valid data found. Check the input files and filtering logic.")
sif_amazon = np.stack(sif_amazon, axis=0).squeeze()  # Shape: [N_months, lat, lon]

years = sorted(set(int(t.split("-")[0]) for t in time_list))
num_years = len(years)

sif_monthly = np.full((num_years, 12, *sif_amazon.shape[1:]), np.nan)
for i, month_data in enumerate(sif_amazon):
    year_idx = i // 12  
    month_idx = i % 12  
    sif_monthly[year_idx, month_idx] = month_data 


print("Fixed sif_monthly shape:", sif_monthly.shape)


# Compute DINOv3 embeddings for all months and years
all_embeddings = []
for year_idx in range(sif_monthly.shape[0]):
    for month_idx in range(sif_monthly.shape[1]):
        sif_img = sif_monthly[year_idx, month_idx]
        if np.isnan(sif_img).all():
            all_embeddings.append(np.full((1, model.embed_dim), np.nan))
            continue
        sif_min, sif_max = np.nanmin(sif_img), np.nanmax(sif_img)
        sif_norm = (sif_img - sif_min) / (sif_max - sif_min + 1e-8)
        sif_rgb = np.stack([sif_norm]*3, axis=-1)
        sif_rgb_img = Image.fromarray((sif_rgb * 255).astype(np.uint8))
        x_sif = transform(sif_rgb_img).unsqueeze(0)
        with torch.no_grad():
            sif_feats = model(x_sif)
        all_embeddings.append(sif_feats.cpu().numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)  # shape: [N_months_total, embedding_dim]
print("✅ All SIF embeddings shape:", all_embeddings.shape)


pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(all_embeddings)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_pca)
labels = kmeans.labels_

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10')
plt.title("Unsupervised clusters of SIF embeddings")
plt.show()

plt.plot(range(len(labels)), labels, marker='o')
plt.title("Cluster evolution over time")
plt.xlabel("Time step")
plt.ylabel("Cluster ID")
plt.show()


# Plot all SIF images and indicate their cluster assignment
plt.figure(figsize=(24, 12))
num_images = len(labels)
cols = 6  # Number of columns in the grid
rows = math.ceil(num_images / cols)
for idx in range(num_images):
    year_idx = idx // 12
    month_idx = idx % 12
    sif_img = sif_monthly[year_idx, month_idx]
    cluster_id = labels[idx]
    plt.subplot(1, num_images, idx + 1)
    plt.imshow(sif_img, cmap='viridis')
    plt.title(f"{time_list[idx]}\nCluster {cluster_id}")
    plt.axis('off')
plt.suptitle("All SIF images with cluster assignment")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

