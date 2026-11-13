"""
Sports Performance Analytics: Predicting IPL Match Outcomes
File: main.py

This is the main orchestrator script that runs the entire machine learning 
pipeline end-to-end: Preprocessing, Exploratory Data Analysis, Model Training, 
and Evaluation.
"""

import time
import sys
import os

# Import pipeline components
import data_preprocessing
import eda
import model_training
import evaluation

def run_pipeline():
    """
    Executes all pipeline scripts sequentially and measures total time.
    """
    start_time = time.time()
    
    print("="*60)
    print("   IPL MATCH OUTCOME PREDICTION - DATA SCIENCE PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Preprocessing
        print("\n[STEP 1/4] Running Data Preprocessing...")
        data_preprocessing.main()
        
        # Step 2: Exploratory Data Analysis
        print("\n[STEP 2/4] Running Exploratory Data Analysis...")
        eda.main()
        
        # Step 3: Model Training
        print("\n[STEP 3/4] Running Model Training...")
        model_training.main()
        
        # Step 4: Model Evaluation
        print("\n[STEP 4/4] Running Model Evaluation...")
        evaluation.main()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print("\n" + "="*60)
        print("   PIPELINE EXECUTED SUCCESSFULLY!")
        print(f"   Total Execution Time: {elapsed:.2f} seconds")
        print("   All plots saved in:  plots/")
        print("   All models saved in: models/")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[!] ERROR occurred during pipeline execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
