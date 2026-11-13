"""
Sports Performance Analytics: Predicting IPL Match Outcomes
File: model_training.py

This module splits the preprocessed match-level dataset into training and 
testing sets chronologically, trains three machine learning classifiers 
(Logistic Regression, Decision Tree, and Random Forest), and saves the 
trained model pipelines in the 'models/' folder.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Select features and target variable
FEATURES = [
    'team1_win_pct_h', 'team2_win_pct_h', 
    'team1_run_rate_h', 'team2_run_rate_h', 
    'team1_avg_wickets_h', 'team2_avg_wickets_h',
    'home_advantage', 'toss_advantage', 'toss_decision_bat'
]
TARGET = 'team1_win'

def load_and_split_data(filepath="preprocessed_matches.csv"):
    """
    Loads preprocessed dataset and splits it into train and test sets chronologically.
    Train set: Seasons 2008 to 2022.
    Test set: Seasons 2023 to 2025.
    """
    print(f"[*] Loading data from '{filepath}'...")
    df = pd.read_csv(filepath)
    
    # Chronological Split
    train_seasons = ['2007/08', '2009', '2009/10', '2011', '2012', '2013', '2014', 
                     '2015', '2016', '2017', '2018', '2019', '2020/21', '2021', '2022']
    test_seasons = ['2023', '2024', '2025']
    
    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'].isin(test_seasons)]
    
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    
    print(f"[+] Chronological Split Complete:")
    print(f"    Train Set (2008-2022): {len(train_df)} matches (Seasons: {train_df['season'].unique().tolist()})")
    print(f"    Test Set  (2023-2025): {len(test_df)} matches (Seasons: {test_df['season'].unique().tolist()})")
    
    return X_train, X_test, y_train, y_test

def build_model_pipelines():
    """
    Creates modeling pipelines for the three classifiers.
    """
    print("[*] Creating model pipelines...")
    
    # 1. Logistic Regression Pipeline (incorporating standard scaling)
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # 2. Decision Tree Pipeline (incorporating standard scaling for symmetry, though not strictly required)
    dt_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42))
    ])
    
    # 3. Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42))
    ])
    
    pipelines = {
        'Logistic Regression': lr_pipeline,
        'Decision Tree': dt_pipeline,
        'Random Forest': rf_pipeline
    }
    return pipelines

def train_and_evaluate_models(pipelines, X_train, X_test, y_train, y_test):
    """
    Trains each pipeline and evaluates simple accuracies to compare them.
    """
    print("\n" + "="*50)
    print("           MODEL TRAINING & HOLDOUT ACCURACY")
    print("="*50)
    
    trained_models = {}
    best_accuracy = 0
    best_name = None
    
    for name, pipeline in pipelines.items():
        print(f"[*] Training {name}...")
        pipeline.fit(X_train, y_train)
        
        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        
        print(f"[+] {name} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
        trained_models[name] = pipeline
        
        # Save model file
        model_filename = f"models/{name.lower().replace(' ', '_')}_model.joblib"
        joblib.dump(pipeline, model_filename)
        print(f"    Saved model checkpoint to '{model_filename}'")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_name = name
            
    # Save best model separately
    best_model_pipeline = trained_models[best_name]
    joblib.dump(best_model_pipeline, "models/best_model.joblib")
    print("\n" + "="*50)
    print(f"Best classifier: {best_name} (Test Accuracy: {best_accuracy*100:.2f}%)")
    print("Saved as 'models/best_model.joblib'")
    print("="*50 + "\n")
    
    return trained_models

def main():
    """
    Executes the model training pipeline.
    """
    print("=== IPL MATCH OUTCOME MODEL TRAINING PIPELINE ===")
    X_train, X_test, y_train, y_test = load_and_split_data()
    pipelines = build_model_pipelines()
    train_and_evaluate_models(pipelines, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
