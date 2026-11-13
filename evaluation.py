"""
Sports Performance Analytics: Predicting IPL Match Outcomes
File: evaluation.py

This module loads the trained models, runs evaluations on the test dataset, 
computes classification metrics and compares models with a training-only majority
baseline before saving diagnostic visualization plots.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_curve,
    roc_auc_score, classification_report
)

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Feature list and target
FEATURES = [
    'team1_win_pct_h', 'team2_win_pct_h', 
    'team1_run_rate_h', 'team2_run_rate_h', 
    'team1_avg_wickets_h', 'team2_avg_wickets_h',
    'home_advantage', 'toss_advantage', 'toss_decision_bat'
]
TARGET = 'team1_win'

# Premium color variables
COLOR_PRIMARY = "#2C3E50"  # Dark Slate Blue
COLOR_SECONDARY = "#18BC9C"  # Turquoise/Teal
COLOR_ACCENT = "#E74C3C"  # Crimson/Coral

def load_test_data(filepath="preprocessed_matches.csv"):
    """
    Loads the chronological test data and the training-set majority class.
    """
    df = pd.read_csv(filepath)
    train_seasons = ['2007/08', '2009', '2009/10', '2011', '2012', '2013', '2014',
                     '2015', '2016', '2017', '2018', '2019', '2020/21', '2021', '2022']
    test_seasons = ['2023', '2024', '2025']
    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'].isin(test_seasons)]
    
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    baseline_class = int(train_df[TARGET].mode().iloc[0])
    baseline_probability = float(train_df[TARGET].mean())
    return X_test, y_test, baseline_class, baseline_probability

def load_models():
    """
    Loads the saved model pipelines from models/ folder.
    """
    print("[*] Loading trained models...")
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression_model.joblib',
        'Decision Tree': 'models/decision_tree_model.joblib',
        'Random Forest': 'models/random_forest_model.joblib'
    }
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
            print(f"[+] Loaded {name} successfully.")
        else:
            raise FileNotFoundError(f"Model file '{filepath}' not found. Please train models first.")
            
    return models

def evaluate_models(models, X_test, y_test, baseline_class, baseline_probability):
    """
    Evaluates models, prints comparisons, and saves metrics.
    """
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    results = []
    
    for name, model in models.items():
        # Predict classes and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Balanced Accuracy': balanced_acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Brier Score': brier
        })
        
        print(f"\n--- Classification Report: {name} ---")
        print(classification_report(y_test, y_pred, target_names=['Team 2 Wins', 'Team 1 Wins']))

    baseline_pred = np.full(len(y_test), baseline_class)
    baseline_prob = np.full(len(y_test), baseline_probability)
    results.append({
        'Model': 'Training Majority Baseline',
        'Accuracy': accuracy_score(y_test, baseline_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, baseline_pred),
        'Precision': precision_score(y_test, baseline_pred, zero_division=0),
        'Recall': recall_score(y_test, baseline_pred, zero_division=0),
        'F1-Score': f1_score(y_test, baseline_pred, zero_division=0),
        'ROC-AUC': 0.5,
        'Brier Score': brier_score_loss(y_test, baseline_prob)
    })
        
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("                      COMPARATIVE METRICS TABLE")
    print("="*70)
    print(results_df.to_string(index=False, formatters={
        'Accuracy': '{:.4f}'.format,
        'Balanced Accuracy': '{:.4f}'.format,
        'Precision': '{:.4f}'.format,
        'Recall': '{:.4f}'.format,
        'F1-Score': '{:.4f}'.format,
        'ROC-AUC': '{:.4f}'.format,
        'Brier Score': '{:.4f}'.format
    }))
    print("="*70 + "\n")
    
    return results_df

def plot_confusion_matrices(models, X_test, y_test):
    """
    Plots side-by-side confusion matrices for all models.
    """
    print("[*] Generating confusion matrices plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            ax=axes[idx], 
            cbar=False,
            annot_kws={"size": 14},
            xticklabels=['Team 2 Win', 'Team 1 Win'],
            yticklabels=['Team 2 Win', 'Team 1 Win']
        )
        axes[idx].set_title(f"{name} Confusion Matrix")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        
    plt.suptitle("Model Evaluation: Confusion Matrices", y=1.05)
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[+] Saved confusion matrices plot.")

def plot_roc_curves(models, X_test, y_test):
    """
    Plots ROC-AUC curves for all models in a single plot.
    """
    print("[*] Generating ROC curves plot...")
    plt.figure(figsize=(9, 7))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)
        
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.50)", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/roc_curves.png", dpi=150)
    plt.close()
    print("[+] Saved ROC curves plot.")

def plot_feature_importances(models):
    """
    Plots feature importances for the Random Forest model.
    """
    print("[*] Generating feature importances plot...")
    rf_model = models.get('Random Forest')
    
    if rf_model is not None:
        # Extract features and importances
        classifier = rf_model.named_steps['classifier']
        importances = classifier.feature_importances_
        
        feat_imp = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, color=COLOR_SECONDARY)
        plt.title("Random Forest Model: Feature Importance analysis")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png", dpi=150)
        plt.close()
        print("[+] Saved feature importances plot.")
    else:
        print("[!] Random Forest model not loaded, skipping feature importance plot.")

def main():
    """
    Executes the evaluation pipeline.
    """
    print("=== IPL MATCH OUTCOME EVALUATION PIPELINE ===")
    X_test, y_test, baseline_class, baseline_probability = load_test_data()
    models = load_models()
    evaluate_models(models, X_test, y_test, baseline_class, baseline_probability)
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_feature_importances(models)
    print("[+] Evaluation completed. All diagnostics saved in 'plots/'.")

if __name__ == "__main__":
    main()
