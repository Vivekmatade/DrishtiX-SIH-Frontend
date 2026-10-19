from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

class MetricsCalculator:
    """Calculates standard computer vision and semantic segmentation evaluation metrics."""

    @staticmethod
    def calculate_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculates IoU, Precision, Recall, and F1 score for binary segmentation masks.
        Accepts arrays of any shape as long as sizes match.
        """
        true_flat = (y_true.flatten() > 0).astype(np.uint8)
        if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
            pred_flat = (y_pred.flatten() >= threshold).astype(np.uint8)
        else:
            pred_flat = (y_pred.flatten() > 0).astype(np.uint8)

        # Intersection and Union
        intersection = np.logical_and(true_flat, pred_flat).sum()
        union = np.logical_or(true_flat, pred_flat).sum()
        iou = float(intersection / union) if union > 0 else (1.0 if true_flat.sum() == 0 else 0.0)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average="binary", zero_division=0
        )

        return {
            "iou": round(float(iou), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4)
        }

    @staticmethod
    def calculate_multiclass_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Calculates per-class Precision, Recall, F1, IoU, overall accuracy, and confusion matrix.
        """
        true_flat = y_true.flatten().astype(np.int64)
        pred_flat = y_pred.flatten().astype(np.int64)

        labels = sorted(list(class_names.keys()))
        cm = confusion_matrix(true_flat, pred_flat, labels=labels)
        overall_acc = accuracy_score(true_flat, pred_flat)

        per_class = {}
        ious = []

        for idx, cid in enumerate(labels):
            c_name = class_names[cid]
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0

            ious.append(iou)
            per_class[c_name] = {
                "class_id": cid,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "iou": round(iou, 4)
            }

        mean_iou = float(np.mean(ious)) if ious else 0.0

        return {
            "overall_accuracy": round(float(overall_acc), 4),
            "mean_iou": round(mean_iou, 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist()
        }
