"""
Sports Performance Analytics: Predicting IPL Match Outcomes
File: predict.py

This module provides a Command-Line Interface (CLI) to predict IPL match outcomes 
using the trained champion model (models/best_model.joblib). It supports both 
direct command-line arguments and an interactive mode.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Try to import team constants from data_preprocessing
try:
    from data_preprocessing import TEAM_HOME_CITIES, TEAM_MAPPING
except ImportError:
    # Fallback definitions if run independently
    TEAM_MAPPING = {
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
        'Punjab Kings': 'Kings XI Punjab',
        'Delhi Capitals': 'Delhi Daredevils',
        'Rising Pune Supergiants': 'Rising Pune Supergiant'
    }
    TEAM_HOME_CITIES = {
        'Mumbai Indians': ['Mumbai', 'Navi Mumbai'],
        'Chennai Super Kings': ['Chennai'],
        'Kolkata Knight Riders': ['Kolkata'],
        'Royal Challengers Bangalore': ['Bangalore', 'Bengaluru'],
        'Rajasthan Royals': ['Jaipur'],
        'Kings XI Punjab': ['Chandigarh', 'Mohali', 'Dharamsala', 'New Chandigarh'],
        'Delhi Daredevils': ['Delhi'],
        'Sunrisers Hyderabad': ['Hyderabad'],
        'Deccan Chargers': ['Hyderabad'],
        'Gujarat Titans': ['Ahmedabad'],
        'Lucknow Super Giants': ['Lucknow'],
        'Gujarat Lions': ['Rajkot', 'Kanpur'],
        'Pune Warriors': ['Pune'],
        'Rising Pune Supergiant': ['Pune'],
        'Kochi Tuskers Kerala': ['Kochi']
    }

# Features required by the trained model pipeline
FEATURES = [
    'team1_win_pct_h', 'team2_win_pct_h', 
    'team1_run_rate_h', 'team2_run_rate_h', 
    'team1_avg_wickets_h', 'team2_avg_wickets_h',
    'home_advantage', 'toss_advantage', 'toss_decision_bat'
]

def load_champion_model(model_path="models/best_model.joblib"):
    """
    Loads the best trained model pipeline.
    """
    if not os.path.exists(model_path):
        print(f"[!] Model file '{model_path}' not found. Please train models first using main.py.")
        sys.exit(1)
    return joblib.load(model_path)

def calculate_home_advantage(team1, team2, city):
    """
    Calculates home advantage based on standard team-home city mappings.
    +1 if team1 is home, -1 if team2 is home, 0 if neutral.
    """
    t1_clean = TEAM_MAPPING.get(team1, team1)
    t2_clean = TEAM_MAPPING.get(team2, team2)
    
    t1_home = 1 if city in TEAM_HOME_CITIES.get(t1_clean, []) else 0
    t2_home = 1 if city in TEAM_HOME_CITIES.get(t2_clean, []) else 0
    
    return t1_home - t2_home

def predict_winner(model, team1, team2, toss_winner, toss_decision, city, 
                   t1_win_pct, t2_win_pct, t1_run_rate, t2_run_rate, 
                   t1_avg_wickets=5.5, t2_avg_wickets=5.5):
    """
    Formats features, runs predictions through the pipeline, and displays results.
    """
    # 1. Clean team names
    team1 = TEAM_MAPPING.get(team1, team1)
    team2 = TEAM_MAPPING.get(team2, team2)
    toss_winner = TEAM_MAPPING.get(toss_winner, toss_winner)
    
    # 2. Derive features
    home_adv = calculate_home_advantage(team1, team2, city)
    toss_adv = 1 if toss_winner == team1 else 0
    toss_dec_bat = 1 if toss_decision.lower() == 'bat' else 0
    
    # Create feature dataframe
    input_data = pd.DataFrame([{
        'team1_win_pct_h': t1_win_pct,
        'team2_win_pct_h': t2_win_pct,
        'team1_run_rate_h': t1_run_rate,
        'team2_run_rate_h': t2_run_rate,
        'team1_avg_wickets_h': t1_avg_wickets,
        'team2_avg_wickets_h': t2_avg_wickets,
        'home_advantage': home_adv,
        'toss_advantage': toss_adv,
        'toss_decision_bat': toss_dec_bat
    }])
    
    # 3. Predict probabilities
    prob = model.predict_proba(input_data)[0]
    pred = model.predict(input_data)[0]
    
    t1_prob = prob[1]
    t2_prob = prob[0]
    
    winner = team1 if pred == 1 else team2
    winning_prob = t1_prob if pred == 1 else t2_prob
    
    print("\n" + "="*50)
    print("             IPL MATCH OUTCOME PREDICTION")
    print("="*50)
    print(f"Match-up          : {team1} vs {team2}")
    print(f"Venue (City)      : {city}")
    print(f"Toss Winner       : {toss_winner} (Decision: {toss_decision.upper()})")
    print("-"*50)
    print(f"Team 1 Hist Stats : Win %: {t1_win_pct:.3f} | Run Rate: {t1_run_rate:.2f}")
    print(f"Team 2 Hist Stats : Win %: {t2_win_pct:.3f} | Run Rate: {t2_run_rate:.2f}")
    print(f"Calculated Home   : {'Team 1 Home (+1)' if home_adv == 1 else 'Team 2 Home (-1)' if home_adv == -1 else 'Neutral Venue (0)'}")
    print("-"*50)
    print(f"PREDICTED WINNER  : {winner.upper()}")
    print(f"Probability       : {winning_prob * 100:.2f}%")
    print(f"Probabilities     : {team1}: {t1_prob*100:.1f}% | {team2}: {t2_prob*100:.1f}%")
    print("="*50 + "\n")
    
    return winner, t1_prob, t2_prob

def run_interactive_mode(model):
    """
    Runs an interactive command line interface.
    """
    print("=== IPL Match Outcome Predictor: Interactive Mode ===")
    
    try:
        team1 = input("Enter Team 1 (Batting First): ").strip()
        team2 = input("Enter Team 2 (Chasing Team)  : ").strip()
        
        toss_winner = input(f"Enter Toss Winner ({team1} or {team2}): ").strip()
        while toss_winner not in [team1, team2]:
            print(f"[!] Toss winner must be either '{team1}' or '{team2}'.")
            toss_winner = input(f"Enter Toss Winner ({team1} or {team2}): ").strip()
            
        toss_decision = input("Enter Toss Decision (bat or field)     : ").strip().lower()
        while toss_decision not in ['bat', 'field']:
            print("[!] Decision must be 'bat' or 'field'.")
            toss_decision = input("Enter Toss Decision (bat or field)     : ").strip().lower()
            
        city = input("Enter Match Venue City (e.g. Mumbai)   : ").strip()
        
        t1_win_pct = float(input(f"Enter {team1} Historical Win % (0.0 to 1.0) : "))
        t2_win_pct = float(input(f"Enter {team2} Historical Win % (0.0 to 1.0) : "))
        
        t1_run_rate = float(input(f"Enter {team1} Historical Run Rate (e.g. 7.85): "))
        t2_run_rate = float(input(f"Enter {team2} Historical Run Rate (e.g. 7.95): "))
        
        predict_winner(
            model=model,
            team1=team1,
            team2=team2,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
            city=city,
            t1_win_pct=t1_win_pct,
            t2_win_pct=t2_win_pct,
            t1_run_rate=t1_run_rate,
            t2_run_rate=t2_run_rate
        )
    except ValueError as ve:
        print(f"\n[!] Input Error: Please enter valid numerical values for percentages and rates. ({ve})")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

def run_examples(model):
    """
    Runs pre-configured example predictions.
    """
    print("=== RUNNING EXAMPLE IPL PREDICTIONS ===")
    
    # Example 1: El Clasico of IPL (MI vs CSK at Wankhede, MI is Team 1 home)
    print("\n[*] Example 1: Mumbai Indians vs Chennai Super Kings at Wankhede Stadium, Mumbai")
    predict_winner(
        model=model,
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        toss_winner="Mumbai Indians",
        toss_decision="field",
        city="Mumbai",
        t1_win_pct=0.565,
        t2_win_pct=0.590,
        t1_run_rate=8.10,
        t2_run_rate=8.05
    )
    
    # Example 2: RCB vs KKR at Eden Gardens, Kolkata (KKR is Team 2 home)
    print("[*] Example 2: Royal Challengers Bangalore vs Kolkata Knight Riders at Eden Gardens")
    predict_winner(
        model=model,
        team1="Royal Challengers Bangalore",
        team2="Kolkata Knight Riders",
        toss_winner="Kolkata Knight Riders",
        toss_decision="field",
        city="Kolkata",
        t1_win_pct=0.485,
        t2_win_pct=0.515,
        t1_run_rate=8.20,
        t2_run_rate=8.15
    )

def main():
    parser = argparse.ArgumentParser(description="Predict IPL match outcomes using the trained Random Forest model.")
    
    parser.add_argument("--interactive", action="store_true", help="Run in interactive prompt mode.")
    parser.add_argument("--example", action="store_true", help="Run pre-configured match prediction examples.")
    
    # Prediction arguments
    parser.add_argument("--team1", type=str, help="Team 1 Name (Bat First)")
    parser.add_argument("--team2", type=str, help="Team 2 Name (Chasing)")
    parser.add_argument("--toss-winner", type=str, help="Toss Winner Name")
    parser.add_argument("--toss-decision", type=str, choices=['bat', 'field'], help="Toss Decision ('bat' or 'field')")
    parser.add_argument("--city", type=str, help="Venue City (for home advantage check)")
    parser.add_argument("--t1-win-pct", type=float, help="Team 1 Historical Win Percentage (0 to 1)")
    parser.add_argument("--t2-win-pct", type=float, help="Team 2 Historical Win Percentage (0 to 1)")
    parser.add_argument("--t1-run-rate", type=float, help="Team 1 Historical Run Rate (runs per over)")
    parser.add_argument("--t2-run-rate", type=float, help="Team 2 Historical Run Rate (runs per over)")
    
    args = parser.parse_args()
    model = load_champion_model()
    
    # Trigger examples
    if args.example:
        run_examples(model)
        return
        
    # Trigger interactive if explicitly set OR if no arguments are passed
    if args.interactive or len(sys.argv) == 1:
        run_interactive_mode(model)
        return
        
    # Standard command line execution
    required_args = [args.team1, args.team2, args.toss_winner, args.toss_decision, 
                     args.city, args.t1_win_pct, args.t2_win_pct, args.t1_run_rate, args.t2_run_rate]
                     
    if any(a is None for a in required_args):
        print("[!] Error: Missing required prediction arguments.")
        parser.print_help()
        sys.exit(1)
        
    predict_winner(
        model=model,
        team1=args.team1,
        team2=args.team2,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
        city=args.city,
        t1_win_pct=args.t1_win_pct,
        t2_win_pct=args.t2_win_pct,
        t1_run_rate=args.t1_run_rate,
        t2_run_rate=args.t2_run_rate
    )

if __name__ == "__main__":
    main()
