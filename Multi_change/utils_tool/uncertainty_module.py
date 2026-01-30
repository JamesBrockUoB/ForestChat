"""
Uncertainty Quantification Module for Change Detection
Integrated for predict.py - Simple dictionaries, precise object geometry
"""

import json
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from skimage import measure


class MCDropoutPredictor:
    """Monte Carlo Dropout for epistemic uncertainty estimation."""

    def __init__(
        self, encoder, encoder_trans, device, n_samples=10, force_dropout=True
    ):
        self.encoder = encoder
        self.encoder_trans = encoder_trans
        self.device = device
        self.n_samples = n_samples
        self.force_dropout = force_dropout

    def enable_dropout(self, model):
        """Enable dropout layers during inference"""
        dropout_found = False
        for module in model.modules():
            module_name = module.__class__.__name__
            if "Dropout" in module_name or "dropout" in module_name.lower():
                module.train()
                dropout_found = True

        if not dropout_found:
            print(
                "WARNING: No dropout layers found in model! Epistemic uncertainty will be zero."
            )
            print("Available layer types:")
            layer_types = set([m.__class__.__name__ for m in model.modules()])
            for lt in sorted(layer_types):
                print(f"  - {lt}")

    def enable_dropout_aggressive(self, model):
        """More aggressive dropout enabling - sets entire model to train mode"""
        # Set model to train mode (enables dropout and batch norm)
        model.train()

        # But freeze batch norm layers if present
        for module in model.modules():
            if "BatchNorm" in module.__class__.__name__:
                module.eval()

    def predict_with_uncertainty(self, imgA, imgB):
        """
        Run multiple forward passes to estimate uncertainty.

        Returns:
            predictions: List of segmentation predictions (numpy arrays)
            probabilities: List of class probability maps (numpy arrays)
        """
        # Put models in TRAIN mode to enable dropout
        self.encoder.train()
        self.encoder_trans.train()

        # But freeze batch norm layers so they don't update stats
        for model in [self.encoder, self.encoder_trans]:
            for module in model.modules():
                if "BatchNorm" in module.__class__.__name__:
                    module.eval()

        print(f"Models set to train mode for MC Dropout (batch norms frozen)")

        predictions = []
        probabilities = []

        with torch.no_grad():
            from tqdm import tqdm

            for i in tqdm(
                range(self.n_samples), desc="MC Dropout sampling", leave=False
            ):
                feat1, feat2 = self.encoder(imgA, imgB)
                feat1, feat2, seg_logits = self.encoder_trans(feat1, feat2)

                seg_probs = torch.softmax(seg_logits, dim=1)
                seg_pred = torch.argmax(seg_probs, dim=1)

                predictions.append(seg_pred.cpu().numpy())
                probabilities.append(seg_probs.cpu().numpy())

        # Check if predictions actually vary
        pred_stack = np.stack([p[0] for p in predictions])
        variance = np.var(pred_stack, axis=0).mean()
        print(f"  Prediction variance across samples: {variance:.6f}")
        if variance < 1e-6:
            print("  WARNING: Predictions are identical! Dropout may not be working.")

        # Set back to eval mode
        self.encoder.eval()
        self.encoder_trans.eval()

        return predictions, probabilities


class UncertaintyEstimator:
    """Estimate aleatoric and epistemic uncertainty for change detection."""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def compute_aleatoric_uncertainty(self, class_probs: np.ndarray) -> np.ndarray:
        """
        Compute aleatoric (data) uncertainty from class probabilities using entropy.

        Args:
            class_probs: Class probabilities [num_classes, H, W]

        Returns:
            uncertainty_map: Aleatoric uncertainty [H, W], normalized to [0, 1]
        """
        epsilon = 1e-10
        probs = np.clip(class_probs, epsilon, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=0)
        max_entropy = np.log(self.num_classes)
        normalized_entropy = entropy / max_entropy
        return normalized_entropy

    def compute_epistemic_uncertainty(
        self, predictions: List[np.ndarray], probabilities: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute epistemic (model) uncertainty from MC Dropout samples.
        Uses mutual information: epistemic = H(E[p]) - E[H(p)]

        Returns:
            epistemic_map: Epistemic uncertainty [H, W]
            mean_probs: Mean class probabilities [num_classes, H, W]
        """
        prob_stack = np.stack(probabilities, axis=0)
        mean_probs = np.mean(prob_stack, axis=0)

        # H(E[p]) - entropy of mean prediction
        mean_entropy = self.compute_aleatoric_uncertainty(mean_probs)

        # E[H(p)] - mean of individual entropies
        individual_entropies = [
            self.compute_aleatoric_uncertainty(probs) for probs in prob_stack
        ]
        mean_individual_entropy = np.mean(individual_entropies, axis=0)

        # Mutual information
        epistemic = mean_entropy - mean_individual_entropy
        epistemic = np.clip(epistemic, 0, 1)

        return epistemic, mean_probs

    def extract_object_uncertainties(
        self,
        seg_mask: np.ndarray,
        mean_probs: np.ndarray,
        aleatoric_map: np.ndarray,
        epistemic_map: np.ndarray,
        min_area: int = 50,
    ) -> List[Dict]:
        """
        Extract per-object uncertainty statistics with full geometry.

        Returns list of dictionaries with:
        - object_id: unique identifier
        - class: predicted class
        - contour: precise boundary points [[x, y], ...]
        - mask: binary mask for this object
        - area: area in pixels
        - perimeter: perimeter length
        - centroid: center point (x, y)
        - moments: cv2 moments for shape analysis
        - compactness: shape compactness measure
        - elongation: major/minor axis ratio
        - orientation: orientation angle
        - class_probability: mean probability for predicted class
        - aleatoric_uncertainty: mean data uncertainty
        - epistemic_uncertainty: mean model uncertainty
        - total_uncertainty: sum of both
        - confidence: reliability score
        """
        objects = []

        for class_id in range(1, self.num_classes):
            class_mask = (seg_mask == class_id).astype(np.uint8)

            if np.sum(class_mask) == 0:
                continue

            # Find contours for this class
            contours, hierarchy = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour_idx, contour in enumerate(contours):
                # Create individual object mask
                obj_mask = np.zeros_like(class_mask, dtype=np.uint8)
                cv2.drawContours(obj_mask, [contour], 0, 1, -1)

                area = cv2.contourArea(contour)
                if area < min_area:  # Filter small objects
                    continue

                # Geometry analysis
                perimeter = cv2.arcLength(contour, True)
                moments = cv2.moments(contour)

                # Centroid
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                else:
                    cx, cy = 0, 0

                # Shape descriptors
                compactness = (4 * np.pi * area) / (perimeter**2 + 1e-6)

                # Fit ellipse for elongation (if enough points)
                if len(contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        major_axis = max(ellipse[1])
                        minor_axis = min(ellipse[1])
                        elongation = major_axis / (minor_axis + 1e-6)
                        orientation = ellipse[2]
                    except:
                        elongation = 1.0
                        orientation = 0.0
                else:
                    elongation = 1.0
                    orientation = 0.0

                # Extract uncertainty statistics
                obj_probs = mean_probs[:, obj_mask > 0]
                obj_aleatoric = aleatoric_map[obj_mask > 0]
                obj_epistemic = epistemic_map[obj_mask > 0]

                mean_class_prob = np.mean(obj_probs[class_id])
                mean_aleatoric = np.mean(obj_aleatoric)
                mean_epistemic = np.mean(obj_epistemic)

                # Weighted combination: epistemic matters more for decision-making
                # Aleatoric is about data quality (can't fix with better model)
                # Epistemic is about model confidence (what users actually care about)
                weighted_uncertainty = (mean_epistemic * 3.0) + (mean_aleatoric * 0.5)

                # Normalize to reasonable range
                confidence = 1.0 / (1.0 + weighted_uncertainty)

                # Store both raw and weighted
                total_unc = mean_aleatoric + mean_epistemic

                # Create object dictionary
                obj = {
                    "object_id": len(objects) + 1,
                    "class": int(class_id),
                    "contour": (
                        contour.squeeze().tolist()
                        if len(contour.squeeze().shape) == 2
                        else []
                    ),
                    "mask": obj_mask,
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "centroid": (float(cx), float(cy)),
                    "moments": {k: float(v) for k, v in moments.items()},
                    "compactness": float(compactness),
                    "elongation": float(elongation),
                    "orientation": float(orientation),
                    "class_probability": float(mean_class_prob),
                    "aleatoric_uncertainty": float(mean_aleatoric),
                    "epistemic_uncertainty": float(mean_epistemic),
                    "total_uncertainty": float(total_unc),
                    "weighted_uncertainty": float(weighted_uncertainty),
                    "confidence": float(confidence),
                }

                objects.append(obj)

        return objects


class UncertaintyVisualizer:
    """Create visualizations of uncertainty."""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.class_colors = {0: [0, 0, 0], 1: [0, 255, 255], 2: [0, 0, 255]}

    def create_uncertainty_visualization(
        self,
        imgA: np.ndarray,
        imgB: np.ndarray,
        seg_mask: np.ndarray,
        aleatoric_map: np.ndarray,
        epistemic_map: np.ndarray,
        save_path: str,
    ):
        """Create main 6-panel uncertainty visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Convert images if needed
        if len(imgA.shape) == 3 and imgA.shape[2] == 3:
            imgA_rgb = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
            imgB_rgb = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
        else:
            imgA_rgb = imgA
            imgB_rgb = imgB

        axes[0, 0].imshow(imgA_rgb)
        axes[0, 0].set_title("Image A (Before)", fontsize=14, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(imgB_rgb)
        axes[0, 1].set_title("Image B (After)", fontsize=14, fontweight="bold")
        axes[0, 1].axis("off")

        seg_rgb = self.mask_to_rgb(seg_mask)
        axes[0, 2].imshow(seg_rgb)
        axes[0, 2].set_title("Predicted Change Mask", fontsize=14, fontweight="bold")
        axes[0, 2].axis("off")

        # Bottom row: Redesigned to emphasize what users care about

        # 1. Model Uncertainty (epistemic) - MOST IMPORTANT
        epistemic_max = epistemic_map.max()
        epistemic_95 = (
            np.percentile(epistemic_map[epistemic_map > 0], 95)
            if np.any(epistemic_map > 0)
            else epistemic_max
        )
        vmax_epistemic = max(epistemic_95 * 1.2, 0.05)

        im1 = axes[1, 0].imshow(
            epistemic_map, cmap="plasma", vmin=0, vmax=vmax_epistemic
        )
        axes[1, 0].set_title(
            f"Model Uncertainty (Epistemic)\nmax={epistemic_max:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 2. Data Quality (aleatoric) - SECONDARY
        im2 = axes[1, 1].imshow(aleatoric_map, cmap="YlOrRd", vmin=0, vmax=1)
        axes[1, 1].set_title(
            "Data Quality Issues (Aleatoric) \n(noise, blur, misalignment)",
            fontsize=11,
            fontweight="bold",
        )
        axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 3. Prediction Reliability - WEIGHTED (epistemic emphasized)
        reliability_map = 1.0 / (1.0 + (epistemic_map * 3.0) + (aleatoric_map * 0.5))

        im3 = axes[1, 2].imshow(reliability_map, cmap="RdYlGn", vmin=0, vmax=1)
        axes[1, 2].set_title(
            "Prediction Reliability\n(Green=Reliable, Red=Review)",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 2].axis("off")
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def create_object_overlay(
        self, imgB: np.ndarray, objects: List[Dict], save_path: str, min_area: int = 50
    ):
        """
        Create side-by-side visualization: image with contours + clean legend panel.
        """
        # Filter small objects
        filtered_objects = [obj for obj in objects if obj["area"] >= min_area]

        # Original image with contours
        img_overlay = imgB.copy()
        if len(img_overlay.shape) == 2:
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_GRAY2BGR)

        # Draw contours color-coded by confidence
        for obj in filtered_objects:
            confidence = obj["confidence"]

            if confidence > 0.7:
                color = (0, 255, 0)  # Green
                thickness = 3
            elif confidence > 0.4:
                color = (0, 165, 255)  # Orange
                thickness = 3
            else:
                color = (0, 0, 255)  # Red
                thickness = 4

            if len(obj["contour"]) > 0:
                contour_array = np.array(obj["contour"], dtype=np.int32)
                if len(contour_array.shape) == 2:
                    contour_array = contour_array.reshape(-1, 1, 2)
                    cv2.drawContours(img_overlay, [contour_array], 0, color, thickness)

        # Create legend panel
        h, w = img_overlay.shape[:2]
        legend_w = w
        legend = np.ones((h, legend_w, 3), dtype=np.uint8) * 240

        # Title
        y = 40
        cv2.putText(
            legend,
            "Object Confidence",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        y += 40
        cv2.line(legend, (10, y), (legend_w - 10, y), (150, 150, 150), 2)

        # Legend items
        y += 40
        cv2.rectangle(legend, (10, y - 15), (50, y + 5), (0, 255, 0), -1)
        cv2.putText(
            legend, "High (>0.7)", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

        y += 35
        cv2.rectangle(legend, (10, y - 15), (50, y + 5), (0, 165, 255), -1)
        cv2.putText(
            legend,
            "Medium (0.4-0.7)",
            (60, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        y += 35
        cv2.rectangle(legend, (10, y - 15), (50, y + 5), (0, 0, 255), -1)
        cv2.putText(
            legend, "Low (<0.4)", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

        y += 50
        cv2.line(legend, (10, y), (legend_w - 10, y), (150, 150, 150), 2)

        # Stats
        y += 30
        high_conf = sum(1 for obj in filtered_objects if obj["confidence"] > 0.7)
        med_conf = sum(1 for obj in filtered_objects if 0.4 <= obj["confidence"] <= 0.7)
        low_conf = sum(1 for obj in filtered_objects if obj["confidence"] < 0.4)

        cv2.putText(
            legend,
            f"Total: {len(filtered_objects)}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        y += 30
        cv2.putText(
            legend,
            f"High: {high_conf}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 100, 0),
            1,
        )
        y += 30
        cv2.putText(
            legend,
            f"Medium: {med_conf}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 100, 200),
            1,
        )
        y += 30
        cv2.putText(
            legend,
            f"Low: {low_conf}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 150),
            1,
        )

        # Combine side by side
        combined = np.hstack([img_overlay, legend])
        cv2.imwrite(save_path, combined)

    def mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to RGB."""
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in self.class_colors.items():
            if class_id < self.num_classes:
                rgb[mask == class_id] = color
        return rgb

    def save_uncertainty_masks(
        self,
        aleatoric_map: np.ndarray,
        epistemic_map: np.ndarray,
        save_dir: str,
        name: str,
    ):
        """Save individual uncertainty masks with adaptive scaling for visibility."""
        # Aleatoric
        aleatoric_255 = (aleatoric_map * 255).astype(np.uint8)
        aleatoric_colored = cv2.applyColorMap(aleatoric_255, cv2.COLORMAP_HOT)

        # Epistemic with adaptive scaling
        epistemic_max = epistemic_map.max()
        if epistemic_max > 0:
            epistemic_95 = np.percentile(epistemic_map[epistemic_map > 0], 95)
            scale_max = max(epistemic_95 * 1.2, 0.05)
            epistemic_scaled = np.clip(epistemic_map / scale_max, 0, 1)
        else:
            epistemic_scaled = epistemic_map
        epistemic_255 = (epistemic_scaled * 255).astype(np.uint8)
        epistemic_colored = cv2.applyColorMap(epistemic_255, cv2.COLORMAP_PLASMA)

        # Total
        total_255 = (np.clip(aleatoric_map + epistemic_map, 0, 2) * 127.5).astype(
            np.uint8
        )
        total_colored = cv2.applyColorMap(total_255, cv2.COLORMAP_VIRIDIS)

        cv2.imwrite(f"{save_dir}/{name}_aleatoric_unc.png", aleatoric_colored)
        cv2.imwrite(
            f"{save_dir}/{name}_epistemic_unc_max{epistemic_max:.4f}.png",
            epistemic_colored,
        )
        cv2.imwrite(f"{save_dir}/{name}_total_unc.png", total_colored)


def export_object_statistics(objects: List[Dict], save_path: str):
    """Export per-object uncertainty statistics to JSON."""
    # Create serializable version (exclude mask which is numpy array)
    export_objects = []
    for obj in objects:
        export_obj = {k: v for k, v in obj.items() if k != "mask"}
        export_objects.append(export_obj)

    with open(save_path, "w") as f:
        json.dump(export_objects, f, indent=2)


def generate_uncertainty_report(objects: List[Dict], save_path: str):
    """Generate human-readable uncertainty report."""
    report = []
    report.append("=" * 80)
    report.append("UNCERTAINTY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    if len(objects) == 0:
        report.append("No objects detected.")
        with open(save_path, "w") as f:
            f.write("\n".join(report))
        return

    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total objects: {len(objects)}")

    high_conf = sum(1 for obj in objects if obj["confidence"] > 0.7)
    med_conf = sum(1 for obj in objects if 0.4 <= obj["confidence"] <= 0.7)
    low_conf = sum(1 for obj in objects if obj["confidence"] < 0.4)

    report.append(f"  High confidence (>0.7): {high_conf}")
    report.append(f"  Medium confidence (0.4-0.7): {med_conf}")
    report.append(f"  Low confidence (<0.4): {low_conf}")
    report.append("")

    avg_aleatoric = np.mean([obj["aleatoric_uncertainty"] for obj in objects])
    avg_epistemic = np.mean([obj["epistemic_uncertainty"] for obj in objects])

    report.append(f"Average aleatoric uncertainty: {avg_aleatoric:.3f}")
    report.append(f"Average epistemic uncertainty: {avg_epistemic:.3f}")
    report.append("")

    # Top confident
    report.append("TOP 5 MOST CONFIDENT OBJECTS")
    report.append("-" * 80)
    sorted_conf = sorted(objects, key=lambda x: x["confidence"], reverse=True)[:5]
    for i, obj in enumerate(sorted_conf, 1):
        report.append(f"{i}. Object #{obj['object_id']} (Class {obj['class']})")
        report.append(
            f"   Area: {obj['area']:.1f} px, Confidence: {obj['confidence']:.3f}"
        )
        report.append(
            f"   Aleatoric: {obj['aleatoric_uncertainty']:.3f}, "
            f"Epistemic: {obj['epistemic_uncertainty']:.3f}"
        )
        report.append("")

    # Most uncertain
    report.append("TOP 5 MOST UNCERTAIN OBJECTS (REVIEW NEEDED)")
    report.append("-" * 80)
    sorted_unc = sorted(objects, key=lambda x: x["total_uncertainty"], reverse=True)[:5]
    for obj in sorted_unc:
        report.append(f"Object #{obj['object_id']} (Class {obj['class']})")
        report.append(f"  Total uncertainty: {obj['total_uncertainty']:.3f}")
        report.append(f"  Aleatoric: {obj['aleatoric_uncertainty']:.3f} (data)")
        report.append(f"  Epistemic: {obj['epistemic_uncertainty']:.3f} (model)")
        if obj["aleatoric_uncertainty"] > 0.6:
            report.append("  ⚠ High data uncertainty - check image quality")
        if obj["epistemic_uncertainty"] > 0.3:
            report.append("  ⚠ High model uncertainty - unusual pattern")
        report.append("")

    report.append("=" * 80)

    with open(save_path, "w") as f:
        f.write("\n".join(report))

    return "\n".join(report)
