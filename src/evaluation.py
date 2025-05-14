
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np  # Added import for np

def evaluate_model(model, X_test, y_test, y_prob_ml, y_prob_hybrid):
    """Evaluate the model and generate plots."""
    # Compute metrics
    auc_ml = roc_auc_score(y_test, y_prob_ml)
    auc_hybrid = roc_auc_score(y_test, y_prob_hybrid)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, (y_prob_hybrid > 0.5).astype(int), average="binary")
    
    metrics = {
        "auc_ml": auc_ml,
        "auc_hybrid": auc_hybrid,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # SHAP analysis
    try:
        explainer = shap.TreeExplainer(model)
        print("X_test shape:", X_test.shape)  # Debug print
        shap_values = explainer.shap_values(X_test)
        print("shap_values[1] shape:", shap_values[1].shape)  # Debug print
        shap.summary_plot(shap_values[1], X_test, plot_type="bar")
        plt.savefig("figures/shap_summary.png")
        plt.close()
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        # Fallback to feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]  # Now np is defined
        plt.figure(figsize=(10, 6))
        plt.bar(range(X_test.shape[1]), importances[indices])
        plt.xticks(range(X_test.shape[1]), X_test.columns[indices], rotation=45)
        plt.tight_layout()
        plt.savefig("figures/feature_importances.png")
        plt.close()
    
    # ROC curve
    from sklearn.metrics import roc_curve
    fpr_ml, tpr_ml, _ = roc_curve(y_test, y_prob_ml)
    fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, y_prob_hybrid)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ml, tpr_ml, label=f"ML Model (AUC = {auc_ml:.2f})")
    plt.plot(fpr_hybrid, tpr_hybrid, label=f"Hybrid Model (AUC = {auc_hybrid:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("figures/roc_curve.png")
    plt.close()
    
    # Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    prec_ml, rec_ml, _ = precision_recall_curve(y_test, y_prob_ml)
    prec_hybrid, rec_hybrid, _ = precision_recall_curve(y_test, y_prob_hybrid)
    plt.figure(figsize=(8, 6))
    plt.plot(rec_ml, prec_ml, label="ML Model")
    plt.plot(rec_hybrid, prec_hybrid, label="Hybrid Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("figures/pr_curve.png")
    plt.close()
    
    return metrics