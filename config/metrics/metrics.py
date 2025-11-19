from vision_bench.typing.metric import Metric, ToleranceMetric, MetricPipeline
from vision_bench.utils.export import flood_1d, binary_average_precision
import torch

class Recall(MetricPipeline, ToleranceMetric): 
    def __preprocess__(self, preds, targets):
        preds = (preds > 0.5).to(torch.int)
        targets = (targets > 0.5).to(torch.int)
        preds = flood_1d(preds, self.tolerance)
        return preds, targets  # matched = recalled ones

class RecallPerClass(Recall): 
    def __compute__(self, preds, targets):
        tp = (preds & targets).sum(dim=0).float()
        fn = (targets & (~preds)).clamp(min=0).sum(dim=0).float()
        recall = tp / (tp + fn + 1e-8)
        return recall.tolist()

class RecallMacro(ToleranceMetric):
    def __call__(self, preds, targets):
        return torch.tensor(RecallPerClass(self.tolerance)(preds, targets)).mean().item()

class RecallMicro(Recall): 
    def __compute__(self, preds, targets):
        tp = (preds & targets).sum().float()
        fn = (targets - tp).clamp(min=0).sum().float()
        recall = tp / (tp + fn + 1e-8)
        return recall.item()
    
class Precision(MetricPipeline, ToleranceMetric): 
    def __preprocess__(self, preds, targets):
        preds = (preds > 0.5).to(torch.int)
        targets = (targets > 0.5).to(torch.int)
        targets = flood_1d(targets, self.tolerance)
        return preds, targets  # matched = true positives

class PrecisionPerClass(Precision):
    def __compute__(self, preds, targets):
        tp = (preds & targets).sum(dim=0).float()
        fp = (preds & (~targets)).clamp(min=0).sum(dim=0).float()
        precision = tp / (tp + fp + 1e-8)
        return precision.tolist()

class PrecisionMacro(ToleranceMetric):
    def __call__(self, preds, targets):
        return torch.tensor(PrecisionPerClass(self.tolerance)(preds, targets)).mean().item()

class PrecisionMicro(Precision): 
    def __compute__(self, preds, targets):
        tp = (preds & targets).sum().float()
        fp = (preds - tp).clamp(min=0).sum().float()
        precision = tp / (tp + fp + 1e-8)
        return precision.item()


# ========= F1 Metrics =========
class F1PerClass(ToleranceMetric):
    def __call__(self, preds, targets):
        precision_per_class = PrecisionPerClass(self.tolerance)(preds, targets)
        recall_per_class = RecallPerClass(self.tolerance)(preds, targets)
        f1_per_class = []
        for p, r in zip(precision_per_class, recall_per_class):
            if p + r == 0:
                f1_per_class.append(0.0)
            else:
                f1_per_class.append(2 * p * r / (p + r))
        return f1_per_class
    
class F1Micro(ToleranceMetric):
    def __call__(self, preds, targets):
        p = PrecisionMicro(self.tolerance)
        r = RecallMicro(self.tolerance)
        precision = p(preds, targets)
        recall = r(preds, targets)
        return 2 * precision * recall / (precision + recall + 1e-8)

class F1Macro(ToleranceMetric):
    def __call__(self, preds, targets):
        f1_per_class = F1PerClass(self.tolerance)(preds, targets)
        return torch.tensor(f1_per_class).mean().item()

class PositivePerClass(Metric):
    def __call__(self, preds, targets):
        positives = targets.sum(dim=0).tolist()
        return positives

class Accuracy(MetricPipeline, ToleranceMetric):
    def __preprocess__(self, preds, targets):
        preds = (preds > 0.5).to(torch.int)
        targets = (targets > 0.5).to(torch.int)
        preds_expanded = (flood_1d(preds, self.tolerance) & targets.bool()) | preds.bool()
        targets_expanded = (flood_1d(targets, self.tolerance) & preds.bool()) | targets.bool()
        return preds_expanded, targets_expanded

    def __compute__(self, preds, targets):
        correct = (preds == targets).float().sum()
        total = torch.numel(preds)
        return (correct / total).item()

class APPerClass: 
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor): 
        return [binary_average_precision(preds[:, i], targets[:, i]) for i in range(preds.shape[1])]

class mAP: 
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        return torch.tensor(APPerClass()(preds, targets)).mean().item()

def binary_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class (column-wise) binary confusion matrices: shape [num_classes, 2, 2]
    [
 [[TN, FN],     # class 0
  [FP, TP]],

 [[TN, FN],     # class 1
  [FP, TP]],

 [[TN, FN],     # class 2
  [FP, TP]]
]

    """
    # Binarize predictions using threshold 0.5
    preds = (preds >= 0.5).int()
    targets = targets.int()

    num_classes = preds.shape[1]
    conf_matrices = torch.zeros((num_classes, 2, 2), dtype=torch.int)

    for i in range(num_classes):
        p = preds[:, i]
        t = targets[:, i]
        for pred_val in (0, 1):
            for true_val in (0, 1):
                conf_matrices[i, pred_val, true_val] = ((p == pred_val) & (t == true_val)).sum()

    return conf_matrices