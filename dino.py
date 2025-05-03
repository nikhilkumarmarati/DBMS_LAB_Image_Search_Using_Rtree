import numpy as np
import torch
from PIL import Image
import gc
from transformers import AutoModel
from sklearn.decomposition import PCA
import os

# ================================
# STEP 1: Load and Process STL-10 Images in Chunks
# ================================
def load_stl10_image_chunk(filename, start_idx, chunk_size):
    try:
        with open(filename, 'rb') as f:
            # Each image is 3 channels x 96 x 96 = 27,648 bytes
            f.seek(start_idx * 3 * 96 * 96)
            data = np.fromfile(f, dtype=np.uint8, count=chunk_size * 3 * 96 * 96)
        
        if len(data) == 0:
            print(f"No data read from index {start_idx}")
            return None
        
        # Reshape to get the actual number of complete images
        num_complete_images = len(data) // (3 * 96 * 96)
        if num_complete_images == 0:
            print(f"Not enough data for a complete image at index {start_idx}")
            return None
            
        images = data[:num_complete_images * 3 * 96 * 96].reshape(-1, 3, 96, 96)
        return images
    except Exception as e:
        print(f"Error loading images: {e}")
        return None

def preprocess_image(image):
    image = Image.fromarray(np.transpose(image, (1, 2, 0)))
    image = image.resize((224, 224))
    return np.array(image).transpose(2, 0, 1) / 255.0  # Normalize here

# ================================
# STEP 2: Extract Features Using DINOv2 in Small Chunks
# ================================
def extract_features_in_chunks(filename, model, total_images=5000, chunk_size=10):
    model.eval()
    all_features = []
    processed_count = 0
    
    for start_idx in range(0, total_images, chunk_size):
        # Load a small chunk of images
        print(f"Loading chunk starting at index {start_idx}")
        images = load_stl10_image_chunk(filename, start_idx, chunk_size)
        
        if images is None:
            print(f"No more images found after index {start_idx}")
            break
            
        if len(images) == 0:
            print(f"Empty chunk at index {start_idx}")
            continue
        
        # Preprocess images
        print(f"Preprocessing {len(images)} images...")
        preprocessed_images = [preprocess_image(img) for img in images]
        images_tensor = torch.tensor(np.array(preprocessed_images), dtype=torch.float32)
        
        # Extract features
        print(f"Extracting features...")
        with torch.no_grad():
            output = model(images_tensor)
            features = output["pooler_output"].cpu().numpy()
            all_features.append(features)
        
        processed_count += len(images)
        print(f"Processed {processed_count}/{total_images} images")
        
        # Clear memory
        del images, preprocessed_images, images_tensor
        gc.collect()
        
    if not all_features:
        raise ValueError("No features were extracted. Check your input file path and data.")
        
    return np.vstack(all_features)

# ================================
# STEP 3: Wrapped Model Definition
# ================================
class WrappedModelWithPCA(torch.nn.Module):
    def __init__(self, model, pca_components):
        super(WrappedModelWithPCA, self).__init__()
        self.model = model
        self.pca_components = torch.nn.Parameter(torch.tensor(pca_components, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        output = self.model(x)
        features = output["pooler_output"]
        return torch.matmul(features, self.pca_components.T)  # Apply PCA transformation

# ================================
# STEP 4: Main Execution with Memory Optimization
# ================================
def main():
    # Check if file exists
    file_path = "stl10_binary/train_X.bin"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    
    # Load model
    print("Loading DINOv2 model...")
    base_model = AutoModel.from_pretrained("facebook/dinov2-small")
    
    # Extract features
    print("Extracting features...")
    all_features = extract_features_in_chunks(file_path, base_model, chunk_size=5)
    print("Extracted features shape:", all_features.shape)
    
    # Apply PCA for dimensionality reduction (25 dimensions)
    print("Applying PCA...")
    pca = PCA(n_components=25)
    pca.fit(all_features)
    
    # Save PCA components
    np.save("pca_components.npy", pca.components_)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Free memory before creating the wrapped model
    del all_features
    gc.collect()
    
    # Create and save wrapped model
    print("Creating wrapped model with PCA...")
    pca_components = np.load("pca_components.npy")
    wrapped_model = WrappedModelWithPCA(base_model, pca_components)
    wrapped_model.eval()
    
    # Save model as TorchScript
    print("Saving model as TorchScript...")
    dummy_input = torch.rand(1, 3, 224, 224)
    scripted_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
    scripted_model.save("dinov2_pca_25d.pt")
    
    print("Model saved as dinov2_pca_25d.pt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
