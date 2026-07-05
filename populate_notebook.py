import nbformat as nbf
import os

notebook_path = "notebooks/evaluation.ipynb"

# If the notebook doesn't exist yet, wait or create it.
if not os.path.exists(notebook_path):
    os.makedirs("notebooks", exist_ok=True)
    nb = nbf.v4.new_notebook()
else:
    nb = nbf.read(notebook_path, as_version=4)

cells = [
    nbf.v4.new_markdown_cell("# Model Evaluation & Inference\nThis notebook demonstrates how to load the trained Face Similarity model, calculate evaluation metrics, and perform single-pair inference."),
    nbf.v4.new_code_cell("""import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root to path
sys.path.append('..')

from face_similarity import EmbeddingNet, get_transforms, compute_similarity, evaluate
from argparse import Namespace"""),
    nbf.v4.new_markdown_cell("## 1. Define device and load model"),
    nbf.v4.new_code_cell("""device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

checkpoint_path = "../checkpoints/best.pt"
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

embedding_dim = ckpt.get('embedding_dim', 256)
img_size = ckpt.get('img_size', 112)

model = EmbeddingNet(embedding_dim=embedding_dim).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("Model loaded successfully.")"""),
    nbf.v4.new_markdown_cell("## 2. Single Pair Inference\nLet's test the model on two images and compute their cosine distance."),
    nbf.v4.new_code_cell("""# Choose two images to compare from the test set
# Modify paths based on actual available images
img1_path = "../dataset/images/subject_1/image1.jpg"
img2_path = "../dataset/images/subject_1/image2.jpg"

try:
    dist, is_same = compute_similarity(model, img1_path, img2_path, device, img_size=img_size, threshold=0.415)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(Image.open(img1_path))
    ax1.axis('off')
    ax1.set_title('Image 1')
    
    ax2.imshow(Image.open(img2_path))
    ax2.axis('off')
    ax2.set_title('Image 2')
    
    plt.suptitle(f"Distance: {dist:.4f} | Prediction: {'Same' if is_same else 'Different'}")
    plt.show()
except FileNotFoundError:
    print(f"Please update the paths for img1_path and img2_path with actual images.")"""),
    nbf.v4.new_markdown_cell("## 3. Run Full Test Set Evaluation\nCalculates FAR/FRR metrics across a sample of pairs."),
    nbf.v4.new_code_cell("""args = Namespace(
    checkpoint="../checkpoints/best.pt",
    eval_split="test",
    manifest_csv="../dataset/splits/manifest.csv",
    project_root="..",
    embedding_dim=256,
    img_size=112,
    max_pairs=500_000
)

results = evaluate(args)""")
]

# Append or replace cells
nb.cells = cells

nbf.write(nb, notebook_path)
print(f"Successfully populated {notebook_path}")
