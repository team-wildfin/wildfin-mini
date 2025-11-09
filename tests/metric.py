from fish_benchmark.utils.export import *
from scripts.export import compute
tolerance = 2
probs = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.3, 0.4]])
targets = torch.tensor([[0, 1, 0], [1, 0, 0]])

aggregate_metrics = {
            #average metrics
            "f1_micro": F1Micro(tolerance),
            "f1_macro": F1Macro(tolerance),
            "precision_micro": PrecisionMicro(tolerance),
            "precision_macro": PrecisionMacro(tolerance),
            "recall_micro": RecallMicro(tolerance),
            "recall_macro": RecallMacro(tolerance),
            "acc": Accuracy(tolerance),
}
per_class_metrics = {
    #per-class metrics
    "f1_per_class": F1PerClass(tolerance),
    "precision_per_class": PrecisionPerClass(tolerance),
    "recall_per_class": RecallPerClass(tolerance),
    "positive_per_class": PositivePerClass(),
}
results = compute(probs, targets, aggregate_metrics | per_class_metrics, device=torch.device('cpu'))