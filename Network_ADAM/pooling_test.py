import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load the .pth file
data = torch.load(r'.\data\lines_16p_data_1.pth')

# Assuming `data` is a tensor containing images
image = data[100]  # Access the first image

# If the image is already loaded as a tensor, skip the above line
# If it’s a 2D image, add batch and channel dimensions
if len(image.shape) == 2:  
    image = image.unsqueeze(0).unsqueeze(0)

# Helper function to prepare images for visualization
def process_pooling(image, pooling_function, steps=5):
    pooled_images = [image]
    current_image = image.clone()
    for _ in range(steps):
        current_image = pooling_function(current_image, kernel_size=2)
        pooled_images.append(current_image)
    return pooled_images

# Perform max pooling and average pooling
max_pooled_images = process_pooling(image, F.max_pool2d)
avg_pooled_images = process_pooling(image, F.avg_pool2d)

# Visualize results in one figure
fig, axes = plt.subplots(3, 6, figsize=(18, 9))

# Plot original image
axes[0, 0].imshow(image.squeeze().numpy(), cmap='gray')
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Plot max pooling steps
for i in range(5):
    axes[1, i + 1].imshow(max_pooled_images[i + 1].squeeze().numpy(), cmap='gray')
    axes[1, i + 1].set_title(f"Max Pool {i + 1}")
    axes[1, i + 1].axis('off')

# Plot average pooling steps
for i in range(5):
    axes[2, i + 1].imshow(avg_pooled_images[i + 1].squeeze().numpy(), cmap='gray')
    axes[2, i + 1].set_title(f"Avg Pool {i + 1}")
    axes[2, i + 1].axis('off')

# Remove unused subplot axes
for ax in axes[0, 1:]:
    ax.axis('off')

# Add labels for rows
axes[1, 0].axis('off')
axes[1, 0].text(-0.3, 0.5, "Max Pooling", fontsize=12, ha='center', va='center', rotation=90, transform=axes[1, 0].transAxes)
axes[2, 0].axis('off')
axes[2, 0].text(-0.3, 0.5, "Avg Pooling", fontsize=12, ha='center', va='center', rotation=90, transform=axes[2, 0].transAxes)

plt.tight_layout()
plt.show()
