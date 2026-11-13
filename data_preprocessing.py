"""
Sports Performance Analytics: Predicting IPL Match Outcomes
File: data_preprocessing.py

This module handles the loading, cleaning, and preprocessing of the raw IPL 
ball-by-ball dataset. It aggregates the data into a match-level format and 
engineers historical performance features chronologically to prevent data leakage.
"""

import os
import pandas as pd
import numpy as np

# Mapping of historical/alternative team names to standardized, modern names
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

def load_raw_data(filepath="IPL.csv"):
    """
    Loads the raw IPL dataset.
    """
    print(f"[*] Loading raw dataset from '{filepath}'...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"[+] Loaded raw dataset. Shape: {df.shape}")
    return df

def clean_data(df):
    """
    Cleans the dataset by removing duplicates, standardizing team names,
    resolving tied matches using super overs, and filling missing cities.
    """
    print("[*] Cleaning dataset...")
    
    # 1. Remove duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"[+] Removed duplicates. Shape went from {initial_shape} to {df.shape}")
    
    # 2. Standardize team names
    print("[*] Standardizing team names...")
    team_cols = ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by', 'superover_winner']
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_MAPPING)
            
    # 3. Resolve winners for tied matches using super overs
    # If winner is 'Unknown' and result_type is 'tie', map it to superover_winner
    print("[*] Resolving tied match winners...")
    
    def resolve_winner(row):
        w = row['match_won_by']
        if w == 'Unknown':
            if row['result_type'] == 'tie':
                return row['superover_winner']
            else:
                return None  # Represents 'no result' matches, to be dropped
        return w
        
    df['winner'] = df.apply(resolve_winner, axis=1)
    
    # Drop rows where winner is None (no-result matches)
    df = df.dropna(subset=['winner'])
    print(f"[+] Tied matches resolved. Active deliveries shape: {df.shape}")
    
    # 4. Handle missing/Unknown cities
    print("[*] Cleaning city and venue information...")
    df['city'] = df['city'].replace({'Unknown': None})
    # Fill based on venue mapping for Dubai and Sharjah
    venue_city_map = {
        'Dubai International Cricket Stadium': 'Dubai',
        'Sharjah Cricket Stadium': 'Sharjah'
    }
    df['city'] = df['city'].fillna(df['venue'].map(venue_city_map))
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def aggregate_to_matches(df):
    """
    Aggregates ball-by-ball delivery data to match-level records.
    """
    print("[*] Aggregating delivery data to match-level...")
    
    # Keep only standard innings 1 and 2 for run/wicket/ball aggregation
    df_std = df[df['innings'].isin([1, 2])]
    
    # Group by match_id to get match-level properties (take the first row's attributes)
    # We sort by date to ensure chronological order is maintained
    match_grouped = df.sort_values('date').groupby('match_id')
    
    matches = match_grouped.first().reset_index()
    
    # Identify Team 1 (bats first in Innings 1) and Team 2 (bowls first in Innings 1)
    inn1_first = df_std[df_std['innings'] == 1].sort_values(['date', 'over', 'ball']).groupby('match_id').first().reset_index()
    team1_2 = inn1_first[['match_id', 'batting_team', 'bowling_team']].rename(
        columns={'batting_team': 'team1', 'bowling_team': 'team2'}
    )
    
    # Merge team definitions into matches
    matches = matches.drop(columns=['team1', 'team2'], errors='ignore')
    matches = matches.merge(team1_2, on='match_id', how='inner')
    
    # Calculate Runs, Balls, and Wickets for Innings 1 (Team 1)
    inn1_stats = df_std[df_std['innings'] == 1].groupby('match_id').agg(
        team1_runs=('runs_total', 'sum'),
        team1_balls=('valid_ball', 'sum'),
        team1_wickets=('player_out', lambda x: x.notna().sum())
    ).reset_index()
    
    # Calculate Runs, Balls, and Wickets for Innings 2 (Team 2)
    inn2_stats = df_std[df_std['innings'] == 2].groupby('match_id').agg(
        team2_runs=('runs_total', 'sum'),
        team2_balls=('valid_ball', 'sum'),
        team2_wickets=('player_out', lambda x: x.notna().sum())
    ).reset_index()
    
    # Merge stats into matches
    matches = matches.merge(inn1_stats, on='match_id', how='left')
    matches = matches.merge(inn2_stats, on='match_id', how='left')
    
    # Fill any NaNs with 0 (e.g., if innings was not played or cut short)
    stats_cols = ['team1_runs', 'team1_balls', 'team1_wickets', 'team2_runs', 'team2_balls', 'team2_wickets']
    matches[stats_cols] = matches[stats_cols].fillna(0)
    
    # Standardize column order and keep only match-level relevant columns
    cols_to_keep = [
        'match_id', 'date', 'season', 'venue', 'city', 'team1', 'team2',
        'toss_winner', 'toss_decision', 'winner', 'result_type',
        'team1_runs', 'team1_balls', 'team1_wickets',
        'team2_runs', 'team2_balls', 'team2_wickets', 'player_of_match'
    ]
    matches = matches[cols_to_keep]
    print(f"[+] Match-level dataset created. Total matches: {len(matches)}")
    
    return matches

def engineer_features(matches):
    """
    Engineers match-level features chronologically to prevent data leakage.
    Features:
    - Run Rate: average runs scored per 6 valid balls (overs) historically.
    - Winning Percentage: team win rate historically.
    - Home Advantage: 1 if team1 is home, -1 if team2 is home, 0 if neutral.
    - Toss Advantage: 1 if team1 won toss, else 0.
    """
    print("[*] Engineering historical features (chronologically)...")
    
    # Sort chronologically
    matches = matches.sort_values(by=['date', 'match_id']).reset_index(drop=True)
    
    # Lists to store the engineered historical features
    t1_win_pct_list = []
    t2_win_pct_list = []
    t1_run_rate_list = []
    t2_run_rate_list = []
    t1_avg_wickets_list = []
    t2_avg_wickets_list = []
    
    # We iterate through each match and compute history based on preceding matches
    for i, row in matches.iterrows():
        current_date = row['date']
        t1 = row['team1']
        t2 = row['team2']
        
        # Exclude every match on the current date so same-day results cannot
        # leak into one another's historical features.
        history = matches[matches['date'] < current_date]
        
        # Helper to compute history for a specific team
        def get_team_history(team):
            # Matches where this team played
            team_matches = history[(history['team1'] == team) | (history['team2'] == team)]
            played_count = len(team_matches)
            
            if played_count == 0:
                # Default baseline stats if team has never played before
                return 0.5, 7.5, 5.5  # Win pct, run rate, average wickets
                
            # Wins
            wins = (team_matches['winner'] == team).sum()
            win_pct = wins / played_count
            
            # Total runs and balls faced to calculate run rate
            total_runs = 0
            total_balls = 0
            total_wickets = 0
            
            for _, match in team_matches.iterrows():
                if match['team1'] == team:
                    total_runs += match['team1_runs']
                    total_balls += match['team1_balls']
                    total_wickets += match['team1_wickets']
                else:
                    total_runs += match['team2_runs']
                    total_balls += match['team2_balls']
                    total_wickets += match['team2_wickets']
            
            # Run rate = runs per over (runs / (balls / 6))
            run_rate = (total_runs / (total_balls / 6)) if total_balls > 0 else 7.5
            avg_wickets = total_wickets / played_count
            
            return win_pct, run_rate, avg_wickets
            
        t1_win_pct, t1_run_rate, t1_avg_wickets = get_team_history(t1)
        t2_win_pct, t2_run_rate, t2_avg_wickets = get_team_history(t2)
        
        t1_win_pct_list.append(t1_win_pct)
        t2_win_pct_list.append(t2_win_pct)
        t1_run_rate_list.append(t1_run_rate)
        t2_run_rate_list.append(t2_run_rate)
        t1_avg_wickets_list.append(t1_avg_wickets)
        t2_avg_wickets_list.append(t2_avg_wickets)
        
    # Append engineered features to the matches dataframe
    matches['team1_win_pct_h'] = t1_win_pct_list
    matches['team2_win_pct_h'] = t2_win_pct_list
    matches['team1_run_rate_h'] = t1_run_rate_list
    matches['team2_run_rate_h'] = t2_run_rate_list
    matches['team1_avg_wickets_h'] = t1_avg_wickets_list
    matches['team2_avg_wickets_h'] = t2_avg_wickets_list
    
    # Home Advantage
    # team1_home: 1 if match is in team1's home cities, 0 otherwise
    # team2_home: 1 if match is in team2's home cities, 0 otherwise
    # home_advantage: team1_home - team2_home (1 = team1 home, -1 = team2 home, 0 = neutral)
    def check_home(team, city):
        if pd.isna(city) or team not in TEAM_HOME_CITIES:
            return 0
        return 1 if city in TEAM_HOME_CITIES[team] else 0
        
    matches['team1_home'] = matches.apply(lambda r: check_home(r['team1'], r['city']), axis=1)
    matches['team2_home'] = matches.apply(lambda r: check_home(r['team2'], r['city']), axis=1)
    matches['home_advantage'] = matches['team1_home'] - matches['team2_home']
    
    # Toss Advantage
    matches['toss_advantage'] = (matches['toss_winner'] == matches['team1']).astype(int)
    matches['toss_decision_bat'] = (matches['toss_decision'] == 'bat').astype(int)
    
    # Target Variable: 1 if team1 wins, 0 if team2 wins
    matches['team1_win'] = (matches['winner'] == matches['team1']).astype(int)
    
    print("[+] Feature engineering completed successfully.")
    return matches

def main():
    """
    Executes the preprocessing pipeline.
    """
    print("=== IPL MATCH OUTCOME PREPROCESSING PIPELINE ===")
    raw_df = load_raw_data("IPL.csv")
    cleaned_df = clean_data(raw_df)
    matches_df = aggregate_to_matches(cleaned_df)
    preprocessed_df = engineer_features(matches_df)
    
    # Save to CSV
    output_path = "preprocessed_matches.csv"
    preprocessed_df.to_csv(output_path, index=False)
    print(f"[+] Saved preprocessed match dataset to '{output_path}'.")
    
    # Print shapes and info
    print("\n--- Preprocessed Data Summary ---")
    print(f"Shape: {preprocessed_df.shape}")
    print(f"Columns: {preprocessed_df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(preprocessed_df[['date', 'team1', 'team2', 'winner', 'team1_win', 'team1_win_pct_h', 'team2_win_pct_h', 'home_advantage', 'toss_advantage']].head())

if __name__ == "__main__":
    main()
