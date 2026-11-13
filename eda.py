"""
Sports Performance Analytics: Predicting IPL Match Outcomes
File: eda.py

This module performs Exploratory Data Analysis (EDA) on the preprocessed 
match-level dataset and raw delivery dataset. It generates summary statistics 
and saves professional data visualizations to the 'plots/' directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the plots directory exists
os.makedirs("plots", exist_ok=True)

# Set premium visualization styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18,
    'figure.figsize': (10, 6),
    'figure.dpi': 150
})

# Custom premium palette
COLOR_PRIMARY = "#2C3E50"  # Dark Slate Blue
COLOR_SECONDARY = "#18BC9C"  # Turquoise/Teal
COLOR_ACCENT = "#E74C3C"  # Crimson/Coral
COLOR_MUTED = "#95A5A6"  # Cool Gray

def load_data():
    """
    Loads both raw deliveries and preprocessed matches data.
    """
    print("[*] Loading datasets for EDA...")
    matches = pd.read_csv("preprocessed_matches.csv")
    matches['date'] = pd.to_datetime(matches['date'])
    
    # Load raw delivery data for player stats (runs, wickets)
    raw = pd.read_csv("IPL.csv", low_memory=False)
    # Standardize team names in raw data too
    team_mapping = {
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
        'Punjab Kings': 'Kings XI Punjab',
        'Delhi Capitals': 'Delhi Daredevils',
        'Rising Pune Supergiants': 'Rising Pune Supergiant'
    }
    for col in ['batting_team', 'bowling_team', 'match_won_by', 'superover_winner']:
        raw[col] = raw[col].replace(team_mapping)
        
    print(f"[+] Loaded matches ({matches.shape}) and deliveries ({raw.shape}).")
    return matches, raw

def run_summary_statistics(matches):
    """
    Prints high-level summary statistics of the dataset.
    """
    print("\n" + "="*50)
    print("           SUMMARY STATISTICS & KEY METRICS")
    print("="*50)
    
    total_matches = len(matches)
    print(f"Total Matches Analyzed  : {total_matches}")
    print(f"Seasons Covered         : {matches['season'].nunique()} ({matches['season'].min()} to {matches['season'].max()})")
    print(f"Unique Venues           : {matches['venue'].nunique()}")
    print(f"Unique Cities           : {matches['city'].nunique()}")
    print(f"Average Innings 1 Runs : {matches['team1_runs'].mean():.2f}")
    print(f"Average Innings 2 Runs : {matches['team2_runs'].mean():.2f}")
    
    # Missing values check
    print("\n--- Missing Values in Processed Data ---")
    print(matches.isnull().sum()[matches.isnull().sum() > 0])
    
    print("\n--- Summary Description (Numerical Features) ---")
    num_cols = ['team1_runs', 'team2_runs', 'team1_win_pct_h', 'team2_win_pct_h', 
                'team1_run_rate_h', 'team2_run_rate_h', 'home_advantage', 'toss_advantage']
    print(matches[num_cols].describe().round(3))
    print("="*50 + "\n")

def plot_distributions(matches):
    """
    Generates and saves histograms and boxplots of match scores.
    """
    print("[*] Generating distribution plots...")
    
    # 1. Innings Scores Histograms
    plt.figure(figsize=(12, 5))
    
    sns.histplot(matches['team1_runs'], kde=True, color=COLOR_PRIMARY, label="Innings 1 (Bat First)", alpha=0.7)
    sns.histplot(matches['team2_runs'], kde=True, color=COLOR_SECONDARY, label="Innings 2 (Chase)", alpha=0.5)
    
    plt.title("Distribution of IPL Match Innings Scores")
    plt.xlabel("Total Runs Scored")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/innings_score_distribution.png", dpi=150)
    plt.close()
    
    # 2. Boxplot of Runs by Season
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='season', y='team1_runs', data=matches.sort_values('season'), color=COLOR_SECONDARY)
    plt.title("IPL 1st Innings Scores Distribution Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("1st Innings Runs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/innings_score_boxplot.png", dpi=150)
    plt.close()
    print("[+] Saved score distribution plots.")

def plot_correlation_heatmap(matches):
    """
    Generates and saves the correlation matrix heatmap.
    """
    print("[*] Generating correlation heatmap...")
    corr_cols = [
        'team1_win_pct_h', 'team2_win_pct_h', 
        'team1_run_rate_h', 'team2_run_rate_h', 
        'team1_avg_wickets_h', 'team2_avg_wickets_h',
        'home_advantage', 'toss_advantage', 'team1_win'
    ]
    corr_matrix = matches[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5,
        vmin=-1, 
        vmax=1, 
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title("Correlation Heatmap of Match Prediction Features", pad=20)
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=150)
    plt.close()
    print("[+] Saved correlation heatmap.")

def plot_team_performance(matches):
    """
    Analyzes team wins and saves the team win distribution.
    """
    print("[*] Generating team performance plots...")
    
    # Team Win Count
    win_counts = matches['winner'].value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=win_counts.values, y=win_counts.index, color=COLOR_PRIMARY)
    plt.title("Total Wins by IPL Teams (Standardized)")
    plt.xlabel("Number of Wins")
    plt.ylabel("Team")
    plt.tight_layout()
    plt.savefig("plots/team_win_distribution.png", dpi=150)
    plt.close()
    print("[+] Saved team win distribution plot.")

def plot_toss_impact(matches):
    """
    Analyzes toss impact on match results and saves the plot.
    """
    print("[*] Analyzing toss impact...")
    
    # Count matches where toss winner won the match
    toss_winner_won = (matches['toss_winner'] == matches['winner']).sum()
    toss_winner_lost = len(matches) - toss_winner_won
    
    toss_win_rate = toss_winner_won / len(matches)
    print(f"[+] Toss Winner Won Match: {toss_winner_won} times ({toss_win_rate*100:.2f}%)")
    print(f"[+] Toss Winner Lost Match: {toss_winner_lost} times ({(1-toss_win_rate)*100:.2f}%)")
    
    # Plotting
    plt.figure(figsize=(8, 6))
    labels = ['Toss Winner Wins', 'Toss Winner Loses']
    sizes = [toss_winner_won, toss_winner_lost]
    colors = [COLOR_SECONDARY, COLOR_ACCENT]
    
    plt.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors, 
        explode=(0.05, 0),
        textprops={'fontsize': 13}
    )
    plt.title("Toss Advantage: Toss Winner Match Outcomes", pad=20)
    plt.tight_layout()
    plt.savefig("plots/toss_impact.png", dpi=150)
    plt.close()
    print("[+] Saved toss impact visualization.")

def plot_venue_analysis(matches):
    """
    Analyzes average venue runs and saves venue impact.
    """
    print("[*] Analyzing venue impact...")
    
    # Keep top 15 venues by match count to avoid noise
    top_venues = matches['venue'].value_counts().head(15).index
    venue_df = matches[matches['venue'].isin(top_venues)]
    
    venue_runs = venue_df.groupby('venue')['team1_runs'].mean().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='team1_runs', y='venue', data=venue_runs, color=COLOR_SECONDARY)
    plt.title("Average 1st Innings Score at Top 15 IPL Venues")
    plt.xlabel("Average Runs Scored")
    plt.ylabel("Venue")
    plt.tight_layout()
    plt.savefig("plots/venue_analysis.png", dpi=150)
    plt.close()
    print("[+] Saved venue analysis plot.")

def plot_player_performance(raw_df):
    """
    Extracts top players (batsmen and bowlers) from raw deliveries and saves plots.
    """
    print("[*] Analyzing player performance...")
    
    # 1. Top 10 Batsmen (Runs Scored)
    top_batsmen = raw_df.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(10)
    
    # 2. Top 10 Bowlers (Wickets Taken)
    top_bowlers = raw_df.groupby('bowler')['bowler_wicket'].sum().sort_values(ascending=False).head(10)
    
    # Plotting side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.barplot(x=top_batsmen.values, y=top_batsmen.index, ax=axes[0], color=COLOR_PRIMARY)
    axes[0].set_title("Top 10 IPL Run Scorers")
    axes[0].set_xlabel("Total Runs")
    axes[0].set_ylabel("Batter")
    
    sns.barplot(x=top_bowlers.values, y=top_bowlers.index, ax=axes[1], color=COLOR_SECONDARY)
    axes[1].set_title("Top 10 IPL Wicket Takers")
    axes[1].set_xlabel("Total Wickets")
    axes[1].set_ylabel("Bowler")
    
    plt.suptitle("IPL Historic Top Performers (Standardized Data)")
    plt.tight_layout()
    plt.savefig("plots/top_players.png", dpi=150)
    plt.close()
    print("[+] Saved top players analysis plot.")

def main():
    """
    Executes the EDA pipeline.
    """
    print("=== IPL MATCH OUTCOME EXPLORATORY DATA ANALYSIS ===")
    matches, raw = load_data()
    run_summary_statistics(matches)
    plot_distributions(matches)
    plot_correlation_heatmap(matches)
    plot_team_performance(matches)
    plot_toss_impact(matches)
    plot_venue_analysis(matches)
    plot_player_performance(raw)
    print("[+] EDA completed. All visualizations saved in the 'plots/' folder.")

if __name__ == "__main__":
    main()
