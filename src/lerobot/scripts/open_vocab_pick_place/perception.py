"""
Open Vocabulary Object Detection for Real Robot Pick-and-Place
Uses OWL-ViT for zero-shot object detection
"""

import numpy as np
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class ObjectDetector:
    """Open vocabulary object detector using OWL-ViT"""
    
    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        device: Optional[str] = None,
        score_threshold: float = 0.01,
    ):
        """
        Initialize the object detector.
        
        Args:
            model_name: HuggingFace model name for OWL-ViT
            device: Device to run inference on (cuda/cpu)
            score_threshold: Minimum confidence score for detections
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        
        print(f"Loading OWL-ViT model on {self.device}...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def detect(self, image: np.ndarray, text_queries: List[str]) -> Dict:
        """
        Detect objects in image based on text queries.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            text_queries: List of object names to detect
            
        Returns:
            Dictionary with scores, labels, boxes, and text_queries
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        inputs = self.processor(
            images=image,
            text=text_queries,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        # Process outputs
        logits = torch.max(outputs.logits[0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().numpy()
        labels = logits.indices.cpu().numpy()
        boxes = outputs.pred_boxes[0].cpu().numpy()
        
        return {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
            "text_queries": text_queries,
        }
    
    def get_best_instance(
        self, 
        detections: Dict, 
        query: str,
        workspace_bounds: Optional[Dict] = None,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Optional[Dict]:
        """
        Get the best detection for a specific query.
        
        Args:
            detections: Output from detect()
            query: Object name to find
            workspace_bounds: Optional dict with x, y bounds to filter detections
            image_shape: (height, width) of the image
            
        Returns:
            Dictionary with best detection or None if not found
        """
        best = None
        
        for score, box, label_idx in zip(
            detections["scores"],
            detections["boxes"],
            detections["labels"],
        ):
            if score < self.score_threshold:
                continue
            
            if detections["text_queries"][label_idx] != query:
                continue
            
            # Optional: Filter by workspace bounds (requires camera calibration)
            if workspace_bounds is not None and image_shape is not None:
                cx, cy, _, _ = box
                px = int(cx * image_shape[1])
                py = int(cy * image_shape[0])
                # This would need camera calibration to convert to world coords
                # For now, we skip this check in real robot scenario
            
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "box_norm": box,  # normalized [cx, cy, w, h]
                    "label": query,
                }
        
        return best
    
    def get_detection_center_pixel(
        self, 
        detection: Dict, 
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get pixel coordinates of detection center.
        
        Args:
            detection: Output from get_best_instance()
            image_shape: (height, width) of the image
            
        Returns:
            (px, py) pixel coordinates
        """
        H, W = image_shape
        cx, cy, _, _ = detection["box_norm"]
        px = int(cx * W)
        py = int(cy * H)
        return px, py
    
    def visualize(
        self, 
        image: np.ndarray, 
        detections: Dict, 
        best_detections: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize detections on image.
        
        Args:
            image: RGB image
            detections: All detections from detect()
            best_detections: List of best detections to highlight
            save_path: Optional path to save visualization
        """
        H, W = image.shape[:2]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        ax.axis("off")
        
        # Plot all detections in red
        for score, box, label_idx in zip(
            detections["scores"],
            detections["boxes"],
            detections["labels"],
        ):
            if score < self.score_threshold:
                continue
            
            cx, cy, w, h = box
            x0 = (cx - w / 2) * W
            x1 = (cx + w / 2) * W
            y0 = (cy - h / 2) * H
            y1 = (cy + h / 2) * H
            
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "r", linewidth=1)
        
        # Plot best detections in green
        if best_detections:
            for best in best_detections:
                cx, cy, w, h = best["box_norm"]
                x0 = (cx - w / 2) * W
                x1 = (cx + w / 2) * W
                y0 = (cy - h / 2) * H
                y1 = (cy + h / 2) * H
                
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "g", linewidth=2)
                ax.text(
                    x0, y1, 
                    f"{best['label']} ({best['score']:.2f})",
                    color="green",
                    fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="green", alpha=0.7)
                )
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Test the detector with a sample image"""
    import os
    
    # Initialize detector
    detector = ObjectDetector(score_threshold=0.1)
    
    # Load test image (you'll need to provide your own)
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"Please provide a test image at {test_image_path}")
        return
    
    image = np.array(Image.open(test_image_path))
    
    # Detect objects
    text_queries = ["cup", "bottle", "phone", "keyboard"]
    detections = detector.detect(image, text_queries)
    
    # Get best instances
    best_detections = []
    for query in text_queries:
        best = detector.get_best_instance(detections, query, image_shape=image.shape[:2])
        if best:
            print(f"Found {query} with confidence {best['score']:.3f}")
            best_detections.append(best)
    
    # Visualize
    detector.visualize(image, detections, best_detections, save_path="detection_result.png")


if __name__ == "__main__":
    main()
