# Sports Performance Analytics: Predicting IPL Match Outcomes

This is a complete Data Science and Machine Learning project designed to analyze historical Indian Premier League (IPL) data and predict the outcome of matches using team and venue metrics. 

The project aggregates ball-by-ball delivery data to match-level statistics, engineers historical performance features chronologically (to prevent data leakage), and trains three different machine learning classifiers to predict match winners.

---

## Directory Structure

```text
├── IPL.csv                  # Raw ball-by-ball delivery dataset
├── preprocessed_matches.csv # Preprocessed match-level dataset (generated)
├── data_preprocessing.py    # Cleans data and engineers features
├── eda.py                   # Generates key insights and EDA visualizations
├── model_training.py        # Trains ML models (LogReg, Decision Tree, RF)
├── evaluation.py            # Evaluates models and plots diagnostic graphs
├── predict.py               # Prediction module CLI for user inputs
├── main.py                  # End-to-end orchestrator script
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation
├── report.md                # Detailed academic project report
├── plots/                   # Directory containing saved visual plots
│   ├── correlation_heatmap.png
│   ├── innings_score_distribution.png
│   ├── innings_score_boxplot.png
│   ├── team_win_distribution.png
│   ├── toss_impact.png
│   ├── venue_analysis.png
│   ├── top_players.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── feature_importance.png
└── models/                  # Directory containing trained joblib models
    ├── logistic_regression_model.joblib
    ├── decision_tree_model.joblib
    ├── random_forest_model.joblib
    └── best_model.joblib
```

---

## Features Engineered

To predict the winner between `team1` (batting first) and `team2` (chasing) without data leakage, we compute team statistics **strictly from matches played prior to the current match's date**:
1. **Winning Percentage (`team1_win_pct_h`, `team2_win_pct_h`)**: Historical win rate of both teams prior to the match.
2. **Run Rate (`team1_run_rate_h`, `team2_run_rate_h`)**: Cumulative runs scored divided by cumulative overs faced by each team in prior matches.
3. **Average Wickets Lost (`team1_avg_wickets_h`, `team2_avg_wickets_h`)**: Cumulative wickets lost divided by cumulative matches played in prior matches.
4. **Home Advantage (`home_advantage`)**: Set to `1` if the match is played in Team 1's home city, `-1` if in Team 2's home city, and `0` if it is a neutral venue.
5. **Toss Advantage (`toss_advantage`)**: Set to `1` if Team 1 won the toss, and `0` otherwise.
6. **Toss Decision (`toss_decision_bat`)**: Set to `1` if the toss-winning team elected to bat first, and `0` if they elected to field.

The target variable is binary: `team1_win` (1 if Team 1 won, 0 if Team 2 won).

Matches on the same calendar date are excluded from one another's historical feature calculations because their results would not be known before that day's matches.
Venue options use normalized display labels while retaining the corresponding city for historical statistics and home-advantage calculations.

## Model Evaluation

The evaluation script uses seasons 2023–2025 as a chronological holdout and reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and Brier score. It also compares the classifiers with a majority-class baseline derived only from the training seasons. A score close to or worse than this baseline indicates that the model should be treated as experimental rather than as a reliable forecast.

---

## Prerequisites & Installation

To set up the Python environment and install the required dependencies:

1. Clone or copy the project files to your workspace.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Run the Entire Pipeline
You can run the entire pipeline end-to-end (Data Cleaning, Aggregation, Feature Engineering, EDA, Model Training, and Model Evaluation) by executing:

```bash
python main.py
```

This will run all stages sequentially and output the preprocessed match dataset (`preprocessed_matches.csv`), model checkpoints in `models/`, and visualizations in `plots/`.

### 2. Predict Match Outcomes
Use `predict.py` to predict match outcomes using a trained model. It supports three execution modes:

#### A. Interactive CLI Mode (Default)
Simply run the script with no arguments to get step-by-step interactive prompts:
```bash
python predict.py
```

#### B. Example Predictions Mode
Run pre-configured example match scenarios (e.g., El Clasico: Mumbai Indians vs Chennai Super Kings):
```bash
python predict.py --example
```

#### C. Direct Command-Line Arguments Mode
Pass feature values directly using command-line arguments:
```bash
python predict.py \
  --team1 "Mumbai Indians" \
  --team2 "Chennai Super Kings" \
  --toss-winner "Mumbai Indians" \
  --toss-decision "field" \
  --city "Mumbai" \
  --t1-win-pct 0.565 \
  --t2-win-pct 0.590 \
  --t1-run-rate 8.10 \
  --t2-run-rate 8.05
```
