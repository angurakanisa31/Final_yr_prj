import os
import numpy as np
from PIL import Image, ImageChops, ImageFilter

def calculate_image_similarity(img1_path: str, img2_path: str, output_diff_path: str = None) -> dict:
    """
    Compares two images (e.g., original logo vs uploaded logo, or original product vs uploaded product).
    Calculates similarity percentage, detects mismatches, and highlights mismatched regions.
    Saves the diff image to output_diff_path if provided.
    """
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return {
            "similarity_score": 0.0,
            "logo_match": False,
            "packaging_match": False,
            "is_genuine": False,
            "confidence_score": 0.0,
            "reason": "One or both product images are missing for verification."
        }

    try:
        # Load and resize images to standard dimensions
        size = (300, 300)
        img1 = Image.open(img1_path).convert("RGB").resize(size)
        img2 = Image.open(img2_path).convert("RGB").resize(size)

        # Convert to numpy arrays
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        # 1. Color Similarity (Cosine Similarity of color histograms)
        hist1 = [np.histogram(arr1[:, :, c], bins=32, range=(0, 256))[0] for c in range(3)]
        hist2 = [np.histogram(arr2[:, :, c], bins=32, range=(0, 256))[0] for c in range(3)]
        
        color_sims = []
        for c in range(3):
            h1 = hist1[c]
            h2 = hist2[c]
            denom = (np.linalg.norm(h1) * np.linalg.norm(h2))
            sim = np.dot(h1, h2) / denom if denom > 0 else 0
            color_sims.append(sim)
        color_similarity = float(np.mean(color_sims))

        # 2. Structural Similarity (MSE & Absolute Pixel Difference)
        diff_arr = np.abs(arr1 - arr2)
        mse = np.mean(diff_arr ** 2)
        # Convert MSE to similarity: a high MSE means low similarity
        structural_similarity = float(np.exp(-mse / 1500.0))

        # 3. Shape/Edge Similarity (Applying high-pass filters and comparing edges)
        edges1 = img1.filter(ImageFilter.FIND_EDGES).convert("L")
        edges2 = img2.filter(ImageFilter.FIND_EDGES).convert("L")
        e_arr1 = np.array(edges1, dtype=np.float32) / 255.0
        e_arr2 = np.array(edges2, dtype=np.float32) / 255.0
        
        edge_denom = (np.linalg.norm(e_arr1) * np.linalg.norm(e_arr2))
        edge_similarity = float(np.dot(e_arr1.flatten(), e_arr2.flatten()) / edge_denom) if edge_denom > 0 else 1.0

        # Weighted Hybrid Similarity
        similarity_score = (0.35 * color_similarity) + (0.45 * structural_similarity) + (0.20 * edge_similarity)
        similarity_percentage = max(0.0, min(1.0, similarity_score))

        # Determine mismatches
        color_diff_detected = color_similarity < 0.85
        shape_diff_detected = edge_similarity < 0.75
        structural_diff_detected = structural_similarity < 0.70

        reasons = []
        logo_match = True
        packaging_match = True

        if color_diff_detected:
            reasons.append("Color palette difference detected.")
            logo_match = False
        if shape_diff_detected:
            reasons.append("Shape or outline discrepancy found (altered design).")
            logo_match = False
            packaging_match = False
        if structural_diff_detected:
            reasons.append("Local pattern mismatch detected (packaging texture mismatch).")
            packaging_match = False

        # Highlight mismatched regions
        # We create a visualization of the difference
        # Map pixel-wise difference to a red heatmap overlay
        diff_mask = np.mean(diff_arr, axis=2)  # Gray scale difference
        # Threshold the mask to find significant changes
        threshold = 30.0
        hotspots = diff_mask > threshold

        # Create diff overlay image
        # Base is the original image, we overlay red on mismatched regions
        highlighted = img2.copy()
        highlight_pixels = highlighted.load()
        
        for y in range(size[1]):
            for x in range(size[0]):
                if hotspots[y, x]:
                    # Make it red-ish or overlay a red box
                    r, g, b = highlight_pixels[x, y]
                    # Blend 60% red
                    nr = int(0.4 * r + 0.6 * 255)
                    ng = int(0.4 * g)
                    nb = int(0.4 * b)
                    highlight_pixels[x, y] = (nr, ng, nb)

        if output_diff_path:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_diff_path), exist_ok=True)
            highlighted.save(output_diff_path)

        # Decision rule
        is_genuine = similarity_percentage >= 0.80
        confidence_score = similarity_percentage if is_genuine else (1.0 - similarity_percentage)

        reason = "All product characteristics match. Authenticity verified."
        if not is_genuine:
            reason = "Counterfeit indicators found: " + ", ".join(reasons)
        elif reasons:
            reason = "Warning: Minor discrepancies found. " + ", ".join(reasons)

        return {
            "similarity_score": similarity_percentage,
            "logo_match": logo_match,
            "packaging_match": packaging_match,
            "is_genuine": is_genuine,
            "confidence_score": confidence_score,
            "reason": reason
        }

    except Exception as e:
        return {
            "similarity_score": 0.0,
            "logo_match": False,
            "packaging_match": False,
            "is_genuine": False,
            "confidence_score": 0.0,
            "reason": f"Error running image verification pipeline: {str(e)}"
        }


# Below are representations of CNN/Siamese/CLIP architectures for references as requested by Module 9/11
# These scripts will be included in the source deliverables

siamese_pytorch_code = """
import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Use pretrained ResNet50 for feature embeddings
        self.resnet = models.resnet50(pretrained=True)
        # Replace classification layer with embedding layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Calculate euclidean distance
        euclidean_distance = torch.pow((output1 - output2), 2).sum(1).sqrt()
        return euclidean_distance
"""
