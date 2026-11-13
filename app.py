import os
import joblib
import random
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Constants and Mappings
TEAM_MAPPING = {
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
    'Punjab Kings': 'Kings XI Punjab',
    'Delhi Capitals': 'Delhi Daredevils',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
}

# Mapping of teams to their home cities/venues to calculate Home Advantage
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

# Color mapping and logo mapping for active teams
TEAMS_INFO = [
    {"name": "Mumbai Indians", "abbr": "MI", "logo": "https://upload.wikimedia.org/wikipedia/en/c/cd/Mumbai_Indians_Logo.svg", "colors": ["#004f9f", "#002b5c"]},
    {"name": "Chennai Super Kings", "abbr": "CSK", "logo": "https://upload.wikimedia.org/wikipedia/en/2/2b/Chennai_Super_Kings_Logo.svg", "colors": ["#fdb913", "#fcaa05"]},
    {"name": "Kolkata Knight Riders", "abbr": "KKR", "logo": "https://upload.wikimedia.org/wikipedia/en/4/4c/Kolkata_Knight_Riders_Logo.svg", "colors": ["#3a225d", "#271147"]},
    {"name": "Royal Challengers Bengaluru", "abbr": "RCB", "logo": "https://upload.wikimedia.org/wikipedia/en/2/2a/Royal_Challengers_Bangalore_Logo.svg", "colors": ["#2b2b2c", "#ec1c24"]},
    {"name": "Punjab Kings", "abbr": "PBKS", "logo": "https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.svg", "colors": ["#e31e26", "#b01016"]},
    {"name": "Delhi Capitals", "abbr": "DC", "logo": "https://upload.wikimedia.org/wikipedia/en/f/f5/Delhi_Capitals_Logo.svg", "colors": ["#00008b", "#1e90ff"]},
    {"name": "Rajasthan Royals", "abbr": "RR", "logo": "https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg", "colors": ["#ea1b85", "#004b87"]},
    {"name": "Sunrisers Hyderabad", "abbr": "SRH", "logo": "https://upload.wikimedia.org/wikipedia/en/8/81/Sunrisers_Hyderabad_Logo.svg", "colors": ["#ff822a", "#b03a00"]},
    {"name": "Gujarat Titans", "abbr": "GT", "logo": "https://upload.wikimedia.org/wikipedia/en/0/09/Gujarat_Titans_Logo.svg", "colors": ["#1b2133", "#0b1122"]},
    {"name": "Lucknow Super Giants", "abbr": "LSG", "logo": "https://upload.wikimedia.org/wikipedia/en/a/a9/Lucknow_Super_Giants_Logo.svg", "colors": ["#0057b8", "#e30613"]}
]

FEATURES = [
    'team1_win_pct_h', 'team2_win_pct_h', 
    'team1_run_rate_h', 'team2_run_rate_h', 
    'team1_avg_wickets_h', 'team2_avg_wickets_h',
    'home_advantage', 'toss_advantage', 'toss_decision_bat'
]
TARGET = 'team1_win'

def normalize_venue_label(venue, city):
    venue_name = str(venue).strip() if pd.notna(venue) else ''
    city_name = str(city).strip() if pd.notna(city) else ''
    if not venue_name or not city_name:
        return venue_name
    if venue_name.casefold().endswith(f", {city_name}".casefold()) or venue_name.casefold().endswith(city_name.casefold()):
        return venue_name
    return f"{venue_name}, {city_name}"

# Load Dataset
print("[*] Loading preprocessed matches dataset...")
df = pd.read_csv('preprocessed_matches.csv')
df['date'] = pd.to_datetime(df['date'])
df['venue_label'] = df.apply(lambda row: normalize_venue_label(row['venue'], row['city']), axis=1)

# Load Models
print("[*] Loading trained models...")
models = {
    'Random Forest': joblib.load('models/random_forest_model.joblib'),
    'Logistic Regression': joblib.load('models/logistic_regression_model.joblib'),
    'Decision Tree': joblib.load('models/decision_tree_model.joblib')
}

test_seasons = ['2023', '2024', '2025']
test_df = df[df['season'].astype(str).isin(test_seasons)]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]
MODEL_METRICS = {}
for name, model in models.items():
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    MODEL_METRICS[name] = {
        'accuracy': accuracy_score(y_test, predictions),
        'balancedAccuracy': balanced_accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'rocAuc': roc_auc_score(y_test, probabilities),
        'brierScore': brier_score_loss(y_test, probabilities)
    }

train_seasons = ['2007/08', '2009', '2009/10', '2011', '2012', '2013', '2014',
                 '2015', '2016', '2017', '2018', '2019', '2020/21', '2021', '2022']
train_target = df[df['season'].astype(str).isin(train_seasons)][TARGET]
baseline_class = int(train_target.mode().iloc[0])
baseline_probability = float(train_target.mean())
baseline_predictions = np.full(len(y_test), baseline_class)
baseline_probabilities = np.full(len(y_test), baseline_probability)
MODEL_METRICS['Training Majority Baseline'] = {
    'accuracy': accuracy_score(y_test, baseline_predictions),
    'balancedAccuracy': balanced_accuracy_score(y_test, baseline_predictions),
    'precision': precision_score(y_test, baseline_predictions, zero_division=0),
    'recall': recall_score(y_test, baseline_predictions, zero_division=0),
    'f1': f1_score(y_test, baseline_predictions, zero_division=0),
    'rocAuc': 0.5,
    'brierScore': brier_score_loss(y_test, baseline_probabilities)
}

def clean_team_name(name):
    """Standardizes name to match dataset values."""
    if not name:
        return name
    return TEAM_MAPPING.get(name.strip(), name.strip())

def get_team_latest_stats(team):
    """Computes overall latest stats for a team across all history in dataset."""
    team_clean = clean_team_name(team)
    # Matches where team played
    team_matches = df[(df['team1'] == team_clean) | (df['team2'] == team_clean)]
    played_count = len(team_matches)
    
    if played_count == 0:
        return 0.5, 7.5, 5.5
        
    wins = (team_matches['winner'] == team_clean).sum()
    win_pct = wins / played_count
    
    total_runs = 0
    total_balls = 0
    total_wickets = 0
    
    for _, match in team_matches.iterrows():
        if match['team1'] == team_clean:
            total_runs += match['team1_runs']
            total_balls += match['team1_balls']
            total_wickets += match['team1_wickets']
        else:
            total_runs += match['team2_runs']
            total_balls += match['team2_balls']
            total_wickets += match['team2_wickets']
            
    run_rate = (total_runs / (total_balls / 6)) if total_balls > 0 else 7.5
    avg_wickets = total_wickets / played_count
    
    return float(win_pct), float(run_rate), float(avg_wickets)

@app.route('/teams', methods=['GET'])
def get_teams():
    return jsonify(TEAMS_INFO)

@app.route('/venues', methods=['GET'])
def get_venues():
    venues_df = df[['venue_label', 'city']].dropna().drop_duplicates().sort_values('venue_label')
    venues_list = []
    for _, row in venues_df.iterrows():
        venues_list.append({
            "venue": row['venue_label'],
            "city": str(row['city']).strip()
        })
    return jsonify(venues_list)

@app.route('/historical-stats', methods=['GET'])
def get_historical_stats():
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    venue = request.args.get('venue')
    venue_city = request.args.get('venue_city')
    
    if not team1 or not team2:
        return jsonify({"error": "team1 and team2 parameters are required"}), 400
        
    t1_clean = clean_team_name(team1)
    t2_clean = clean_team_name(team2)
    
    # Head-to-Head matches overall
    h2h = df[((df['team1'] == t1_clean) & (df['team2'] == t2_clean)) | 
             ((df['team1'] == t2_clean) & (df['team2'] == t1_clean))]
    
    matches_played = len(h2h)
    t1_wins = int((h2h['winner'] == t1_clean).sum())
    t2_wins = int((h2h['winner'] == t2_clean).sum())
    draws = int(matches_played - t1_wins - t2_wins)
    
    t1_win_pct = round((t1_wins / matches_played * 100), 1) if matches_played > 0 else 50.0
    t2_win_pct = round((t2_wins / matches_played * 100), 1) if matches_played > 0 else 50.0
    
    # Venue statistics
    venue_stats = {}
    if venue:
        v_df = df[df['venue_label'] == venue]
        if venue_city:
            v_df = v_df[v_df['city'].astype(str).str.strip() == venue_city.strip()]
        if len(v_df) > 0:
            avg_1st = float(v_df['team1_runs'].mean())
            high = float(max(v_df['team1_runs'].max(), v_df['team2_runs'].max()))
            low = float(min(v_df['team1_runs'].min(), v_df['team2_runs'].min()))
            
            winning_runs = v_df.apply(lambda r: r['team1_runs'] if r['winner'] == r['team1'] else r['team2_runs'], axis=1)
            avg_win = float(winning_runs.mean())
            
            toss_counts = v_df['toss_decision'].value_counts().to_dict()
            bat_first_wins = int((v_df['winner'] == v_df['team1']).sum())
            chasing_wins = int((v_df['winner'] == v_df['team2']).sum())
            
            venue_stats = {
                "venueName": venue,
                "avgFirstInningsScore": round(avg_1st, 1),
                "highestScore": int(high),
                "lowestScore": int(low),
                "avgWinningScore": round(avg_win, 1),
                "tossDecisionStats": {
                    "bat": int(toss_counts.get('bat', 0)),
                    "field": int(toss_counts.get('field', 0))
                },
                "winSplit": {
                    "battingFirst": bat_first_wins,
                    "chasing": chasing_wins
                }
            }
            
    response_data = {
        "team1WinPercentage": t1_win_pct,
        "team2WinPercentage": t2_win_pct,
        "matchesPlayed": matches_played,
        "headToHead": {
            "team1": team1,
            "team1Wins": t1_wins,
            "team2": team2,
            "team2Wins": t2_wins,
            "draws": draws
        }
    }
    
    if venue:
        response_data["venueStats"] = venue_stats
        
    return jsonify(response_data)

@app.route('/recent-form', methods=['GET'])
def get_recent_form():
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    
    if not team1 or not team2:
        return jsonify({"error": "team1 and team2 parameters are required"}), 400
        
    t1_clean = clean_team_name(team1)
    t2_clean = clean_team_name(team2)
    
    def calculate_form(team_clean):
        # Sort chronologically, filter matches for team
        t_matches = df[(df['team1'] == team_clean) | (df['team2'] == team_clean)].sort_values('date')
        last_5 = t_matches.tail(5)
        form = []
        for _, match in last_5.iterrows():
            if match['winner'] == team_clean:
                form.append('W')
            else:
                form.append('L')
        # If less than 5 matches played (unlikely for main teams, but possible)
        # Pad with 'W' or 'L' based on team overall performance
        while len(form) < 5:
            form.insert(0, 'W')
        return form
        
    return jsonify({
        "team1Form": calculate_form(t1_clean),
        "team2Form": calculate_form(t2_clean)
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
        
    team1 = data.get('team1')
    team2 = data.get('team2')
    venue = data.get('venue')
    toss_winner = data.get('toss_winner')
    toss_decision = data.get('toss_decision')
    venue_city = data.get('venue_city')
    model_name = data.get('model_name', 'Random Forest')
    
    required = [team1, team2, venue, toss_winner, toss_decision]
    if any(r is None for r in required):
        return jsonify({"error": "Missing required fields"}), 400
        
    # Standarize inputs
    t1_clean = clean_team_name(team1)
    t2_clean = clean_team_name(team2)
    toss_w_clean = clean_team_name(toss_winner)
    
    # 1. Automatically calculate historical features from dataset
    t1_win_pct, t1_run_rate, t1_avg_wickets = get_team_latest_stats(t1_clean)
    t2_win_pct, t2_run_rate, t2_avg_wickets = get_team_latest_stats(t2_clean)
    
    # 2. Derive Home Advantage
    venue_city_row = df[df['venue_label'] == venue]
    if venue_city:
        venue_city_row = venue_city_row[venue_city_row['city'].astype(str).str.strip() == str(venue_city).strip()]
    if venue_city_row.empty:
        return jsonify({"error": "Unknown venue or venue city"}), 400
    city = venue_city_row['city'].dropna().iloc[0]
    
    t1_home = 1 if city in TEAM_HOME_CITIES.get(t1_clean, []) else 0
    t2_home = 1 if city in TEAM_HOME_CITIES.get(t2_clean, []) else 0
    home_adv = t1_home - t2_home
    
    # 3. Derive Toss Advantage & Toss Decision Bat
    toss_adv = 1 if toss_w_clean == t1_clean else 0
    toss_dec_bat = 1 if toss_decision.lower() == 'bat' else 0
    
    # Prepare feature DataFrame
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
    
    # Select model
    model = models.get(model_name, models['Random Forest'])
    
    # Predict probabilities
    prob = model.predict_proba(input_data)[0]
    pred = model.predict(input_data)[0]
    
    t1_prob = float(prob[1])
    t2_prob = float(prob[0])
    
    # Winner
    predicted_winner = team1 if pred == 1 else team2
    winning_prob = t1_prob if pred == 1 else t2_prob
    
    # Model Accuracies
    model_metrics = MODEL_METRICS.get(model_name, MODEL_METRICS['Random Forest'])
    accuracy = round(model_metrics['accuracy'] * 100, 2)
    
    # Predicted margin simulation
    if pred == 1: # Team 1 (Batting First) wins
        if winning_prob > 0.58:
            margin = "18-28 runs"
        elif winning_prob > 0.53:
            margin = "8-15 runs"
        else:
            margin = "2-6 runs"
    else: # Team 2 (Chasing) wins
        if winning_prob > 0.58:
            margin = "6-8 wickets"
        elif winning_prob > 0.53:
            margin = "4-5 wickets"
        else:
            margin = "1-3 wickets"
            
    # Confidence Score is based on probability and model accuracy
    confidence_score = round(winning_prob * 100, 1)
    
    # Head-to-Head overall for prediction details
    h2h = df[((df['team1'] == t1_clean) & (df['team2'] == t2_clean)) | 
             ((df['team1'] == t2_clean) & (df['team2'] == t1_clean))]
    h2h_played = len(h2h)
    
    return jsonify({
        "predictedWinner": predicted_winner,
        "winningProbability": round(winning_prob * 100, 1),
        "team1Probability": round(t1_prob * 100, 1),
        "team2Probability": round(t2_prob * 100, 1),
        "confidenceScore": confidence_score,
        "modelAccuracy": accuracy,
        "modelRocAuc": round(model_metrics['rocAuc'], 4),
        "modelBrierScore": round(model_metrics['brierScore'], 4),
        "baselineBrierScore": round(MODEL_METRICS['Training Majority Baseline']['brierScore'], 4),
        "predictedMargin": margin,
        "modelUsed": model_name,
        "calculatedStats": {
            "team1": team1,
            "team2": team2,
            "team1WinPct": round(t1_win_pct * 100, 1),
            "team2WinPct": round(t2_win_pct * 100, 1),
            "team1RunRate": round(t1_run_rate, 2),
            "team2RunRate": round(t2_run_rate, 2),
            "homeAdvantage": int(home_adv),
            "tossAdvantage": int(toss_adv),
            "h2hMatchesPlayed": h2h_played
        }
    })

@app.route('/analytics', methods=['GET'])
def get_analytics():
    # 1. Summary
    unique_venues = int(df['venue'].nunique())
    # count unique teams
    unique_teams_list = list(set(df['team1'].unique().tolist() + df['team2'].unique().tolist()))
    unique_teams = len(unique_teams_list)
    seasons = sorted(df['season'].unique().tolist())
    years_covered = f"{seasons[0]} - {seasons[-1]}"
    
    summary = {
        "matchesCount": len(df),
        "venuesCount": unique_venues,
        "teamsCount": len(TEAMS_INFO),
        "yearsCovered": years_covered
    }
    
    # 2. Overall Win Percentage Chart (for active teams)
    win_pct_chart = []
    for team_info in TEAMS_INFO:
        t_clean = clean_team_name(team_info['name'])
        team_matches = df[(df['team1'] == t_clean) | (df['team2'] == t_clean)]
        played = len(team_matches)
        if played > 0:
            wins = int((team_matches['winner'] == t_clean).sum())
            win_pct_chart.append({
                "name": team_info['name'],
                "abbr": team_info['abbr'],
                "winPct": round((wins / played * 100), 1),
                "wins": wins,
                "played": played
            })
    win_pct_chart = sorted(win_pct_chart, key=lambda x: x['winPct'], reverse=True)
    
    # 3. Toss Impact
    toss_winner_won = int((df['toss_winner'] == df['winner']).sum())
    toss_winner_lost = len(df) - toss_winner_won
    toss_impact = [
        {"name": "Toss Winner Wins", "value": toss_winner_won},
        {"name": "Toss Loser Wins", "value": toss_winner_lost}
    ]
    
    # 4. Venue-wise wins (Top 10 venues)
    top_venues = df['venue'].value_counts().head(10).index.tolist()
    venue_wins = []
    for v in top_venues:
        v_matches = df[df['venue'] == v]
        bat_wins = int((v_matches['winner'] == v_matches['team1']).sum())
        chasing_wins = int((v_matches['winner'] == v_matches['team2']).sum())
        display_name = v.split(',')[0]
        if len(display_name) > 22:
            display_name = display_name[:20] + "..."
        venue_wins.append({
            "name": display_name,
            "battingFirst": bat_wins,
            "chasing": chasing_wins,
            "total": len(v_matches)
        })
        
    # 5. Most Successful Teams
    successful_teams = []
    for t in df['winner'].dropna().unique():
        # Display modern name if it matches
        display_name = t
        for m_name, db_name in TEAM_MAPPING.items():
            if db_name == t:
                display_name = m_name
                break
        # Filter only active teams or all teams
        wins = int((df['winner'] == t).sum())
        successful_teams.append({
            "name": display_name,
            "wins": wins
        })
    successful_teams = sorted(successful_teams, key=lambda x: x['wins'], reverse=True)[:10]
    
    # 6. Correlation Heatmap
    corr = df[FEATURES].corr().round(2)
    corr_list = []
    feature_labels = {
        'team1_win_pct_h': 'T1 Win %',
        'team2_win_pct_h': 'T2 Win %',
        'team1_run_rate_h': 'T1 Run Rate',
        'team2_run_rate_h': 'T2 Run Rate',
        'team1_avg_wickets_h': 'T1 Avg Wickets',
        'team2_avg_wickets_h': 'T2 Avg Wickets',
        'home_advantage': 'Home Adv',
        'toss_advantage': 'Toss Adv',
        'toss_decision_bat': 'Toss Bat'
    }
    for col in corr.columns:
        for idx in corr.index:
            corr_list.append({
                "x": feature_labels.get(col, col),
                "y": feature_labels.get(idx, idx),
                "value": float(corr.loc[idx, col])
            })
            
    # 7. Feature Importance (Random Forest Model)
    rf_model = models['Random Forest']
    importances = rf_model.named_steps['classifier'].feature_importances_
    feature_importance = []
    for feat, imp in zip(FEATURES, importances):
        feature_importance.append({
            "feature": feature_labels.get(feat, feat),
            "importance": round(float(imp) * 100, 2)
        })
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
    
    # 8. Confusion Matrix, ROC, and PR curves dynamically on test set
    test_df = df[df['season'].astype(str).isin(test_seasons)]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    
    # RF Predictions
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_matrix_data = {
        "labels": ["Chasing Win (0)", "Batting First Win (1)"],
        "matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ]
    }
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curve_data = []
    # Downsample
    step = max(1, len(fpr) // 15)
    for i in range(0, len(fpr), step):
        roc_curve_data.append({
            "fpr": round(float(fpr[i]), 3),
            "tpr": round(float(tpr[i]), 3)
        })
    if len(roc_curve_data) == 0 or roc_curve_data[-1]["fpr"] != 1.0:
        roc_curve_data.append({"fpr": 1.0, "tpr": 1.0})
        
    # Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    pr_curve_data = []
    step = max(1, len(precisions) // 15)
    for i in range(0, len(precisions), step):
        pr_curve_data.append({
            "recall": round(float(recalls[i]), 3),
            "precision": round(float(precisions[i]), 3)
        })
        
    # 9. Model Comparison
    model_comparison = [
        {"name": name, **{key: round(value * 100, 2) if key not in ('rocAuc', 'brierScore') else round(value, 4)
                          for key, value in metrics.items()}}
        for name, metrics in MODEL_METRICS.items()
    ]
    
    return jsonify({
        "summary": summary,
        "winPercentageChart": win_pct_chart,
        "tossImpact": toss_impact,
        "venueWins": venue_wins,
        "mostSuccessfulTeams": successful_teams,
        "correlationHeatmap": corr_list,
        "featureImportance": feature_importance,
        "modelComparison": model_comparison,
        "confusionMatrix": confusion_matrix_data,
        "rocCurve": roc_curve_data,
        "precisionRecallCurve": pr_curve_data
    })


# ─────────────────────────────────────────────────────────────────────────────
# LIVE MATCH SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

TEAM_SQUADS = {
    'Mumbai Indians': {
        'batters': ['Rohit Sharma', 'Ishan Kishan', 'Suryakumar Yadav', 'Hardik Pandya', 'Tilak Varma', 'Tim David'],
        'bowlers': ['Jasprit Bumrah', 'Mohammad Siraj', 'Piyush Chawla', 'Suryakumar Yadav', 'Hardik Pandya']
    },
    'Chennai Super Kings': {
        'batters': ['Ruturaj Gaikwad', 'Devon Conway', 'Ajinkya Rahane', 'Shivam Dube', 'Ambati Rayudu', 'MS Dhoni'],
        'bowlers': ['Deepak Chahar', 'Tushar Deshpande', 'Ravindra Jadeja', 'Maheesh Theekshana', 'Matheesha Pathirana']
    },
    'Royal Challengers Bengaluru': {
        'batters': ['Virat Kohli', 'Faf du Plessis', 'Glenn Maxwell', 'Dinesh Karthik', 'Mahipal Lomror', 'Anuj Rawat'],
        'bowlers': ['Mohammed Siraj', 'Harshal Patel', 'Wanindu Hasaranga', 'Karn Sharma', 'Josh Hazlewood']
    },
    'Kolkata Knight Riders': {
        'batters': ['Shreyas Iyer', 'Jason Roy', 'Nitish Rana', 'Rinku Singh', 'Andre Russell', 'Sam Billings'],
        'bowlers': ['Pat Cummins', 'Varun Chakravarthy', 'Sunil Narine', 'Shardul Thakur', 'Tim Southee']
    },
    'Rajasthan Royals': {
        'batters': ['Sanju Samson', 'Jos Buttler', 'Yashasvi Jaiswal', 'Shimron Hetmyer', 'Devdutt Padikkal', 'Dhruv Jurel'],
        'bowlers': ['Trent Boult', 'Yuzvendra Chahal', 'Ravichandran Ashwin', 'Jason Holder', 'Sandeep Sharma']
    },
    'Delhi Capitals': {
        'batters': ['David Warner', 'Prithvi Shaw', 'Mitchell Marsh', 'Ricky Bhui', 'Rishabh Pant', 'Axar Patel'],
        'bowlers': ['Anrich Nortje', 'Kuldeep Yadav', 'Axar Patel', 'Khaleel Ahmed', 'Mukesh Kumar']
    },
    'Punjab Kings': {
        'batters': ['Shikhar Dhawan', 'Prabhsimran Singh', 'Sam Curran', 'Jonny Bairstow', 'Liam Livingstone', 'Jitesh Sharma'],
        'bowlers': ['Arshdeep Singh', 'Nathan Ellis', 'Kagiso Rabada', 'Rahul Chahar', 'Sam Curran']
    },
    'Sunrisers Hyderabad': {
        'batters': ['Abhishek Sharma', 'Rahul Tripathi', 'Aiden Markram', 'Heinrich Klaasen', 'Abdul Samad', 'Mayank Agarwal'],
        'bowlers': ['Bhuvneshwar Kumar', 'T Natarajan', 'Fazalhaq Farooqi', 'Umran Malik', 'Washington Sundar']
    },
    'Gujarat Titans': {
        'batters': ['Shubman Gill', 'Wriddhiman Saha', 'Vijay Shankar', 'David Miller', 'Hardik Pandya', 'Abhinav Manohar'],
        'bowlers': ['Mohammed Shami', 'Rashid Khan', 'Lockie Ferguson', 'Yash Dayal', 'R Sai Kishore']
    },
    'Lucknow Super Giants': {
        'batters': ['KL Rahul', 'Kyle Mayers', 'Quinton de Kock', 'Deepak Hooda', 'Marcus Stoinis', 'Krunal Pandya'],
        'bowlers': ['Avesh Khan', 'Mark Wood', 'Ravi Bishnoi', 'Mohsin Khan', 'Krunal Pandya']
    }
}

SQUAD_NAME_ALIASES = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Rising Pune Supergiant': 'Mumbai Indians',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Gujarat Titans',
}

def get_squad(team_name):
    display = SQUAD_NAME_ALIASES.get(team_name, team_name)
    return TEAM_SQUADS.get(display, {
        'batters': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6'],
        'bowlers': ['Bowler A', 'Bowler B', 'Bowler C', 'Bowler D', 'Bowler E']
    })

# In-memory live match store
_live_matches = {}
_sim_lock = threading.Lock()

def _make_key(t1, t2):
    return f"{t1}||{t2}"

def _init_simulation(team1, team2):
    batting_squad = get_squad(team2)   # team2 always chases in simulation
    bowling_squad = get_squad(team1)
    target = random.randint(158, 205)
    return {
        'team1': team1,
        'team2': team2,
        'innings': 2,
        'batting_team': team2,
        'bowling_team': team1,
        'target': target,
        'score': 0,
        'wickets': 0,
        'balls': 0,
        'total_overs': 20,
        'batters': [batting_squad['batters'][0], batting_squad['batters'][1]],
        'batter_runs': [0, 0],
        'batter_balls': [0, 0],
        'current_bowler': bowling_squad['bowlers'][0],
        'last_ball': 'Match started — let the game begin!',
        'status': 'Live',
        'winner': None,
        'updated_at': time.time(),
    }

def _simulate_ball(state):
    if state['winner']:
        return state
    balls = state['balls']
    score = state['score']
    target = state['target']
    wickets = state['wickets']
    balls_remaining = 120 - balls
    runs_needed = target - score

    if balls_remaining <= 0 or wickets >= 10:
        if score >= target:
            state['winner'] = state['batting_team']
            state['status'] = f"{state['batting_team']} won by {10 - wickets} wickets!"
        else:
            deficit = target - score - 1
            state['winner'] = state['bowling_team']
            state['status'] = f"{state['bowling_team']} won by {deficit} runs!"
        return state

    pressure = runs_needed / max(balls_remaining, 1)
    if pressure > 2.5:
        weights = [5, 12, 14, 14, 22, 18, 15]
    elif pressure > 1.5:
        weights = [10, 20, 18, 12, 18, 12, 10]
    else:
        weights = [18, 28, 20, 12, 12, 6, 4]

    outcomes = [0, 1, 2, 3, 4, 6, 'W']
    result = random.choices(outcomes, weights=weights, k=1)[0]
    striker = balls % 2

    state['batter_balls'][striker] += 1
    state['balls'] += 1
    desc = ''

    if result == 'W':
        state['wickets'] += 1
        out_batter = state['batters'][striker]
        desc = f"WICKET! {out_batter} is out! {state['wickets']} wickets down."
        bowling_squad = get_squad(state['bowling_team'])
        batting_squad = get_squad(state['batting_team'])
        next_bat_idx = state['wickets'] + 1
        if next_bat_idx < len(batting_squad['batters']):
            state['batters'][striker] = batting_squad['batters'][next_bat_idx]
            state['batter_runs'][striker] = 0
            state['batter_balls'][striker] = 0
        if state['wickets'] >= 10:
            deficit = target - score - 1
            state['winner'] = state['bowling_team']
            state['status'] = f"{state['bowling_team']} won by {deficit} runs!"
    else:
        state['score'] += result
        state['batter_runs'][striker] += result
        batter_name = state['batters'][striker]
        if result == 6:
            desc = f"SIX! {batter_name} launches it over the boundary!"
        elif result == 4:
            desc = f"FOUR! {batter_name} finds the gap beautifully!"
        elif result == 0:
            desc = f"Dot ball. {batter_name} plays defensively."
        else:
            desc = f"{result} run{'s' if result > 1 else ''}. {batter_name} keeps the scoreboard ticking."
        if state['score'] >= target:
            state['winner'] = state['batting_team']
            wkts_left = 10 - state['wickets']
            state['status'] = f"{state['batting_team']} won by {wkts_left} wicket{'s' if wkts_left != 1 else ''}!"

    if state['balls'] % 6 == 0 and not state['winner']:
        over_num = state['balls'] // 6
        bowling_squad = get_squad(state['bowling_team'])
        state['current_bowler'] = bowling_squad['bowlers'][over_num % len(bowling_squad['bowlers'])]
        desc += f" | End of over {over_num}."

    if state['balls'] >= 120 and not state['winner']:
        if state['score'] >= target:
            state['winner'] = state['batting_team']
            state['status'] = f"{state['batting_team']} won!"
        else:
            deficit = target - state['score'] - 1
            state['winner'] = state['bowling_team']
            state['status'] = f"{state['bowling_team']} won by {deficit} runs!"

    state['last_ball'] = desc or state['last_ball']
    state['updated_at'] = time.time()
    return state

def _advance_sim(team1, team2, n=4):
    key = _make_key(team1, team2)
    with _sim_lock:
        if key not in _live_matches:
            _live_matches[key] = _init_simulation(team1, team2)
        st = _live_matches[key]
        for _ in range(n):
            st = _simulate_ball(st)
            if st['winner']:
                break
        _live_matches[key] = st
    return _live_matches[key]

def _get_state(team1, team2, simulate=False):
    key = _make_key(team1, team2)
    with _sim_lock:
        exists = key in _live_matches
    if simulate:
        return _advance_sim(team1, team2, n=random.randint(2, 5))
    return _live_matches.get(key) if exists else None

def _fmt_overs(balls):
    return f"{balls // 6}.{balls % 6}"

def _crr(score, balls):
    return round((score / balls) * 6, 2) if balls > 0 else 0.0

def _rrr(target, score, balls):
    rem = 120 - balls
    needed = target - score
    if rem <= 0:
        return 99.0
    return round((needed / rem) * 6, 2)

def _state_to_api(state):
    if not state:
        return None
    balls = state['balls']
    score = state['score']
    target = state['target']
    wickets = state['wickets']
    batters = [
        {'name': state['batters'][0], 'runs': state['batter_runs'][0],
         'balls': state['batter_balls'][0],
         'sr': round((state['batter_runs'][0] / max(state['batter_balls'][0], 1)) * 100, 1)},
        {'name': state['batters'][1], 'runs': state['batter_runs'][1],
         'balls': state['batter_balls'][1],
         'sr': round((state['batter_runs'][1] / max(state['batter_balls'][1], 1)) * 100, 1)}
    ]
    return {
        'isLive': not bool(state['winner']),
        'status': state['status'],
        'winner': state.get('winner'),
        'team1': state['team1'],
        'team2': state['team2'],
        'battingTeam': state['batting_team'],
        'bowlingTeam': state['bowling_team'],
        'innings': state['innings'],
        'score': score,
        'wickets': wickets,
        'overs': _fmt_overs(balls),
        'target': target,
        'runsNeeded': max(0, target - score),
        'ballsRemaining': max(0, 120 - balls),
        'crr': _crr(score, balls),
        'rrr': _rrr(target, score, balls),
        'batters': batters,
        'currentBowler': state['current_bowler'],
        'lastBall': state['last_ball'],
        'lastUpdated': datetime.fromtimestamp(state['updated_at']).isoformat()
    }

def _blend_probs(pre_t1, pre_t2, live_state):
    if not live_state:
        return round(pre_t1 * 100, 1), round(pre_t2 * 100, 1)
    balls = live_state.get('balls', 0)
    score = live_state.get('score', 0)
    target = live_state.get('target', 165)
    wickets = live_state.get('wickets', 0)

    live_wt = min(1.0, balls / 120)
    pre_wt = 1.0 - live_wt

    rrr_val = _rrr(target, score, balls)
    crr_val = _crr(score, balls)
    wickets_left = max(1, 10 - wickets)

    rate_ratio = crr_val / max(rrr_val, 0.1)
    wicket_factor = wickets_left / 10.0

    raw_batting = rate_ratio * wicket_factor
    live_batting_p = max(0.05, min(0.95, raw_batting / (raw_batting + 1)))
    live_bowling_p = 1.0 - live_batting_p

    # team2 is batting (chasing), team1 is bowling (defending)
    t1_blend = pre_wt * pre_t1 + live_wt * live_bowling_p
    t2_blend = pre_wt * pre_t2 + live_wt * live_batting_p
    total = t1_blend + t2_blend
    return round((t1_blend / total) * 100, 1), round((t2_blend / total) * 100, 1)


@app.route('/live-match-status', methods=['GET'])
def live_match_status():
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    simulate = request.args.get('simulate', 'false').lower() == 'true'
    reset = request.args.get('reset', 'false').lower() == 'true'

    if not team1 or not team2:
        return jsonify({'error': 'team1 and team2 are required'}), 400

    if reset:
        key = _make_key(team1, team2)
        with _sim_lock:
            _live_matches.pop(key, None)

    state = _get_state(team1, team2, simulate=simulate)
    if state is None:
        return jsonify({'isLive': False, 'status': 'No live match found', 'team1': team1, 'team2': team2})

    return jsonify(_state_to_api(state))


@app.route('/predict-live', methods=['POST'])
def predict_live():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    team1 = data.get('team1')
    team2 = data.get('team2')
    venue = data.get('venue', '')
    venue_city = data.get('venue_city')
    toss_winner = data.get('toss_winner', team1)
    toss_decision = data.get('toss_decision', 'bat')
    model_name = data.get('model_name', 'Random Forest')
    simulate = data.get('simulate', False)

    if not team1 or not team2:
        return jsonify({'error': 'team1 and team2 are required'}), 400

    t1_clean = clean_team_name(team1)
    t2_clean = clean_team_name(team2)
    toss_w_clean = clean_team_name(toss_winner)

    t1_wp, t1_rr, t1_aw = get_team_latest_stats(t1_clean)
    t2_wp, t2_rr, t2_aw = get_team_latest_stats(t2_clean)

    vc_row = df[df['venue_label'] == venue]
    if venue_city:
        vc_row = vc_row[vc_row['city'].astype(str).str.strip() == str(venue_city).strip()]
    city = vc_row['city'].dropna().iloc[0] if len(vc_row) > 0 else None
    home_adv = (1 if city in TEAM_HOME_CITIES.get(t1_clean, []) else 0) - (1 if city in TEAM_HOME_CITIES.get(t2_clean, []) else 0)
    toss_adv = 1 if toss_w_clean == t1_clean else 0
    toss_dec_bat = 1 if toss_decision.lower() == 'bat' else 0

    input_data = pd.DataFrame([{
        'team1_win_pct_h': t1_wp, 'team2_win_pct_h': t2_wp,
        'team1_run_rate_h': t1_rr, 'team2_run_rate_h': t2_rr,
        'team1_avg_wickets_h': t1_aw, 'team2_avg_wickets_h': t2_aw,
        'home_advantage': home_adv, 'toss_advantage': toss_adv,
        'toss_decision_bat': toss_dec_bat
    }])

    model = models.get(model_name, models['Random Forest'])
    prob = model.predict_proba(input_data)[0]
    pre_t1 = float(prob[1])
    pre_t2 = float(prob[0])

    live_state = _get_state(team1, team2, simulate=simulate)
    if live_state is None and simulate:
        live_state = _advance_sim(team1, team2, n=4)

    live_resp = _state_to_api(live_state)
    t1_prob, t2_prob = _blend_probs(pre_t1, pre_t2, live_state)
    predicted_winner = team1 if t1_prob >= t2_prob else team2

    return jsonify({
        'predictedWinner': predicted_winner,
        'winningProbability': max(t1_prob, t2_prob),
        'team1Probability': t1_prob,
        'team2Probability': t2_prob,
        'confidenceScore': max(t1_prob, t2_prob),
        'modelUsed': model_name,
        'isLiveMode': bool(live_resp),
        'liveMatch': live_resp,
        'preMatchProbabilities': {'team1': round(pre_t1 * 100, 1), 'team2': round(pre_t2 * 100, 1)},
        'lastUpdated': datetime.now().isoformat()
    })


def overs_to_balls(overs):
    """Converts a standard cricket over format (e.g. 10.2 or 10) to total balls."""
    try:
        overs_float = float(overs)
        overs_int = int(overs_float)
        balls_frac = round((overs_float - overs_int) * 10)
        return overs_int * 6 + min(balls_frac, 5)
    except (ValueError, TypeError):
        return 0

@app.route('/predict-live-custom', methods=['POST'])
def predict_live_custom():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    team1 = data.get('team1')
    team2 = data.get('team2')
    venue = data.get('venue', '')
    venue_city = data.get('venue_city')
    toss_winner = data.get('toss_winner', team1)
    toss_decision = data.get('toss_decision', 'bat')
    model_name = data.get('model_name', 'Random Forest')

    # Custom inputs
    first_innings_runs = float(data.get('first_innings_runs', 160))
    first_innings_wickets = int(data.get('first_innings_wickets', 5))
    first_innings_overs = float(data.get('first_innings_overs', 20))
    
    second_innings_runs = float(data.get('second_innings_runs', 0))
    second_innings_wickets = int(data.get('second_innings_wickets', 0))
    second_innings_overs = float(data.get('second_innings_overs', 0))

    if not team1 or not team2:
        return jsonify({'error': 'team1 and team2 are required'}), 400

    t1_clean = clean_team_name(team1)
    t2_clean = clean_team_name(team2)
    toss_w_clean = clean_team_name(toss_winner)

    # 1. Run pre-match prediction
    t1_wp, t1_rr, t1_aw = get_team_latest_stats(t1_clean)
    t2_wp, t2_rr, t2_aw = get_team_latest_stats(t2_clean)

    vc_row = df[df['venue_label'] == venue]
    if venue_city:
        vc_row = vc_row[vc_row['city'].astype(str).str.strip() == str(venue_city).strip()]
    city = vc_row['city'].dropna().iloc[0] if len(vc_row) > 0 else None
    home_adv = (1 if city in TEAM_HOME_CITIES.get(t1_clean, []) else 0) - (1 if city in TEAM_HOME_CITIES.get(t2_clean, []) else 0)
    toss_adv = 1 if toss_w_clean == t1_clean else 0
    toss_dec_bat = 1 if toss_decision.lower() == 'bat' else 0

    input_data = pd.DataFrame([{
        'team1_win_pct_h': t1_wp, 'team2_win_pct_h': t2_wp,
        'team1_run_rate_h': t1_rr, 'team2_run_rate_h': t2_rr,
        'team1_avg_wickets_h': t1_aw, 'team2_avg_wickets_h': t2_aw,
        'home_advantage': home_adv, 'toss_advantage': toss_adv,
        'toss_decision_bat': toss_dec_bat
    }])

    model = models.get(model_name, models['Random Forest'])
    prob = model.predict_proba(input_data)[0]
    pre_t1 = float(prob[1])
    pre_t2 = float(prob[0])

    # 2. Process live custom state
    target = int(first_innings_runs) + 1
    second_innings_balls = overs_to_balls(second_innings_overs)
    balls_remaining = max(0, 120 - second_innings_balls)
    runs_needed = max(0, target - second_innings_runs)

    live_state = {
        'team1': team1,
        'team2': team2,
        'innings': 2,
        'balls': second_innings_balls,
        'score': second_innings_runs,
        'target': target,
        'wickets': second_innings_wickets,
        'batting_team': team2,
        'bowling_team': team1,
        'status': 'Live',
        'winner': None,
        'updated_at': time.time(),
        'current_bowler': 'Opposition Bowler',
        'last_ball': 'Custom setup initialized.',
        'batters': ['Striker', 'Non-Striker'],
        'batter_runs': [0, 0],
        'batter_balls': [0, 0],
    }

    # Resolve win/loss states
    if second_innings_runs >= target:
        live_state['winner'] = team2
        live_state['status'] = f"{team2} won by {10 - second_innings_wickets} wickets!"
    elif balls_remaining <= 0 or second_innings_wickets >= 10:
        live_state['winner'] = team1
        deficit = target - second_innings_runs - 1
        if deficit == 0:
            live_state['status'] = "Match Tied!"
            live_state['winner'] = "Tie"
        else:
            live_state['status'] = f"{team1} won by {deficit} runs!"

    live_resp = _state_to_api(live_state)
    
    # Overwrite custom stats that aren't in the default state_to_api
    if live_resp:
        live_resp['firstInningsRuns'] = first_innings_runs
        live_resp['firstInningsWickets'] = first_innings_wickets
        live_resp['firstInningsOvers'] = first_innings_overs
        live_resp['firstInningsRunRate'] = round(first_innings_runs / first_innings_overs, 2) if first_innings_overs > 0 else 0.0

    t1_prob, t2_prob = _blend_probs(pre_t1, pre_t2, live_state)
    
    # Force 100% probabilities if match has finished
    if live_state['winner'] == team2:
        t1_prob = 0.0
        t2_prob = 100.0
    elif live_state['winner'] == team1:
        t1_prob = 100.0
        t2_prob = 0.0

    predicted_winner = team1 if t1_prob >= t2_prob else team2

    return jsonify({
        'predictedWinner': predicted_winner,
        'winningProbability': max(t1_prob, t2_prob),
        'team1Probability': t1_prob,
        'team2Probability': t2_prob,
        'confidenceScore': max(t1_prob, t2_prob),
        'modelUsed': model_name,
        'isLiveMode': True,
        'liveMatch': live_resp,
        'preMatchProbabilities': {'team1': round(pre_t1 * 100, 1), 'team2': round(pre_t2 * 100, 1)},
        'lastUpdated': datetime.now().isoformat()
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
