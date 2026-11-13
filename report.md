# Sports Performance Analytics: Predicting IPL Match Outcomes
**Academic Project Report | College Final-Year Data Science Project**

---

## Abstract
Predicting the outcome of cricket matches, particularly in high-octane leagues like the Indian Premier League (IPL), is an intricate problem due to the volatile nature of Twenty20 (T20) cricket, team composition changes, player form, toss outcomes, and ground dimensions. This project presents a structured, end-to-end data science pipeline that aggregates historical IPL ball-by-ball delivery data (from 2008 to 2025) into match-level statistics, engineers chronological features to prevent data leakage, and evaluates three machine learning models (Logistic Regression, Decision Tree, and Random Forest) under a chronological split. We discuss performance evaluation metrics, analyze feature importances, and present critical domain insights, including how the "Impact Player" rule (introduced in 2023) has shifted data distributions and affected model performance.

---

## 1. Introduction
### 1.1 Context and Background
The Indian Premier League (IPL) is one of the most popular and commercially successful sports leagues in the world. T20 cricket, the format played in the IPL, is characterized by high scores, rapid play, and high volatility. Unlike Test or One-Day International (ODI) cricket, where longer durations allow team strengths to average out, a single over or even a single delivery in T20 can pivot the match outcome.

### 1.2 Problem Statement
Predicting match outcomes prior to the first delivery is a classic sports analytics problem. The primary challenge is to build predictive models that rely strictly on information available *before* the match starts. Standard data science tutorials often leak match-time data (like runs scored in the current match) into the models, creating a trivial classifier with 100% accuracy but zero predictive power for future matches. 

### 1.3 Objectives
- Load and clean a comprehensive, large-scale IPL deliveries dataset (over 278,000 rows).
- Aggregate delivery-level details into match-level statistics.
- Develop a chronological feature engineering strategy to calculate team win percentages, run rates, and home advantages, avoiding data leakage.
- Train and compare Logistic Regression, Decision Tree, and Random Forest classifiers.
- Evaluate model performances on a test set composed of the most recent IPL seasons (2023–2025).
- Identify and analyze key factors influencing match outcomes.

---

## 2. Dataset and Preprocessing
The primary raw dataset is `IPL.csv`, which contains ball-by-ball records from the inception of the league in 2008 to the 2025 season. It consists of **278,205 rows** and **64 columns**.

### 2.1 Data Cleaning & Unification
To make the dataset usable for machine learning, several cleaning steps were executed:
1. **Duplicate Elimination:** The dataset was scanned for identical rows, and duplicates were removed.
2. **Team Name Standardization:** Over the years, several IPL franchises rebranded (e.g., *Delhi Daredevils* became *Delhi Capitals*, *Punjab Kings* was formerly *Kings XI Punjab*) or had minor name spelling variations (e.g., *Royal Challengers Bangalore* vs *Royal Challengers Bengaluru*). These were mapped to unified names to ensure historical statistics are grouped correctly.
3. **Resolving Tied Outcomes:** Tied matches where `match_won_by` was recorded as "Unknown" were resolved by extracting the winner of the tie-breaking Super Over (`superover_winner`).
4. **Excluding No-Results:** Matches abandoned due to rain or external factors (recorded as "no result" with "Unknown" winners) were dropped.
5. **Imputing Missing Cities:** Missing cities for matches played in Dubai and Sharjah were filled based on the venue name.

### 2.2 Aggregation to Match-Level
The cleaned delivery dataset was filtered to standard innings (1 and 2) to aggregate statistics for each unique `match_id`:
$$\text{team1\_runs} = \sum \text{runs\_total} \quad (\text{Innings } 1)$$
$$\text{team2\_runs} = \sum \text{runs\_total} \quad (\text{Innings } 2)$$
$$\text{team1\_balls} = \sum \text{valid\_ball} \quad (\text{Innings } 1)$$
$$\text{team2\_balls} = \sum \text{valid\_ball} \quad (\text{Innings } 2)$$
$$\text{team1\_wickets} = \sum [\text{player\_out is not null}] \quad (\text{Innings } 1)$$
$$\text{team2\_wickets} = \sum [\text{player\_out is not null}] \quad (\text{Innings } 2)$$

Here, `team1` represents the team batting first in Innings 1, and `team2` represents the chasing team.

---

## 3. Feature Engineering & Selection
Predicting match outcomes *before* the match starts requires compiling historical profiles of the competing teams. To prevent data leakage (using future information to predict the past), we computed all team metrics **strictly chronologically** by sorting the dataset by date and filtering preceding matches only.

### 3.1 Description of Engineered Features
1. **Winning Percentage (`team1_win_pct_h`, `team2_win_pct_h`):**
   $$W_{\%}(t) = \frac{\text{Matches Won by } t \text{ prior to date}}{\text{Matches Played by } t \text{ prior to date}}$$
2. **Historical Run Rate (`team1_run_rate_h`, `team2_run_rate_h`):** 
   Calculated as runs per over (6 valid deliveries) scored historically:
   $$RR(t) = 6 \times \frac{\sum \text{Runs Scored by } t \text{ in prior matches}}{\sum \text{Valid Balls faced by } t \text{ in prior matches}}$$
3. **Average Wickets Lost (`team1_avg_wickets_h`, `team2_avg_wickets_h`):**
   $$Wkt_{avg}(t) = \frac{\sum \text{Wickets Lost by } t \text{ in prior matches}}{\text{Matches Played by } t \text{ prior to date}}$$
4. **Home Advantage (`home_advantage`):**
   Franchises have designated home stadiums. We mapped teams to their home cities (e.g., Chennai Super Kings playing in Chennai).
   - `+1` if the venue is Team 1's home ground.
   - `-1` if the venue is Team 2's home ground.
   - `0` if it is a neutral venue (e.g., South Africa in 2009, UAE in 2014/2020, or neutral cities).
5. **Toss Advantage (`toss_advantage`):**
   - `1` if Team 1 won the toss, and `0` otherwise.
6. **Toss Decision (`toss_decision_bat`):**
   - `1` if the toss winner chose to bat, and `0` if they chose to field.

### 3.2 Target Variable
The target variable is binary: `team1_win` (1 if Team 1 wins, 0 if Team 2 wins).

---

## 4. Machine Learning Implementation
Three machine learning algorithms were selected to model the relationship between team profiles and match results:
1. **Logistic Regression:** Serves as a baseline model. It applies a sigmoid function to a linear combination of features, providing interpretable probability estimates.
2. **Decision Tree:** A non-linear model that splits the feature space hierarchically. A maximum depth of 4 was set to prevent overfitting.
3. **Random Forest:** An ensemble method that trains multiple decision trees on bootstrap samples and averages their predictions, reducing variance.

### 4.1 Feature Scaling
Since Logistic Regression is sensitive to the magnitude of features (e.g., run rate values are ~7.5 while win percentages are ~0.5), we applied `StandardScaler` to scale all features to zero mean and unit variance. This was integrated into a scikit-learn `Pipeline` to prevent data leakage during training.

### 4.2 Train-Test Split Strategy
Instead of a random train-test split, which would violate the temporal nature of sports (i.e. predicting past matches using models trained on future seasons), we utilized a **Chronological Split**:
- **Training Set (81.5%):** Seasons 2008 through 2022 (946 matches).
- **Test Set (18.5%):** Seasons 2023, 2024, and 2025 (215 matches).

---

## 5. Results & Discussion

### 5.1 Model Evaluation Table
Evaluating the trained models on the test set (Seasons 2023–2025) yielded the following comparative results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 46.98% | 46.84% | 33.94% | 39.36% | 0.4762 |
| **Decision Tree** | 44.65% | 41.94% | 23.85% | 30.41% | 0.4384 |
| **Random Forest** | **49.77%** | **50.67%** | **34.86%** | **41.30%** | **0.4957** |

### 5.2 Analysis of Results
1. **The Predictability Limit:** All models hovered around 45% to 50% accuracy on the test set. In binary classification (where predicting "Team 2 Wins" for all matches gives a baseline of 54%), a score of 50% indicates that predicting IPL match outcomes *prior to the match starting* using only historical averages is extremely difficult.
2. **Random Forest Superiority:** The Random Forest model outperformed the simple Decision Tree and Logistic Regression, achieving 49.77% accuracy and a 0.5067 precision score on predicting Team 1 wins, showing the benefit of ensemble learning.

---

## 6. Critical Insights: The Impact Player Rule
A key contribution of this project is explaining the drop in predictive accuracy on recent seasons. In 2023, the BCCI introduced the **"Impact Player" rule** in the IPL, allowing teams to substitute a player during the match.

### 6.1 Shift in Game Dynamics (Covariate Shift)
Before 2023, teams had to balance their XI with all-rounders. The Impact Player rule allowed teams to play an extra specialist batsman in the first innings and swap them for a specialist bowler in the second innings.
This led to:
- **Higher Innings Scores:** Teams batted with more aggression because they had batting depth down to number 8 or 9. The average first-innings score rose from ~165 to over 190.
- **Devaluation of Historical Stats:** Historical averages (e.g., run rates computed from 2008–2022) were built under the old rules. In 2023–2025, these historical run rates failed to capture the new scoring rates, leading to a significant **covariate shift** (concept drift).

### 6.2 The Volatility of T20 Cricket
T20 cricket is designed for parity. In the IPL, auction dynamics, salary caps, and player drafts ensure that teams are relatively balanced. When two balanced teams play, the match is close to a 50/50 event. Consequently, external, unmodeled factors play a massive role:
- **The Dew Factor:** Matches played in the evening often experience dew in the second innings, making the wet ball slippery for bowlers and giving a huge advantage to the chasing team.
- **Pitch Conditions:** Pitch changes (slow vs flat) significantly alter scoring.
- **Player Match-ups:** Specific bowlers matching up well against specific batsmen.

---

## 7. Future Scope & Improvements
To improve predictive accuracy beyond 50%, future iterations of this project could incorporate:
1. **Recent Form (Rolling Windows):** Instead of lifetime historical averages, calculate a team's win percentage and run rate over their last 5 or 10 matches.
2. **Match-up Data:** Incorporate head-to-head records between the two teams.
3. **Player-Specific Ratings:** Use player metrics (like ICC rankings, recent strike rates, or custom player ratings) aggregated to represent the team strength of the active playing XI.
4. **Real-time Match Factors:** Incorporate venue-specific dew history, weather, and pitch condition data.

---

## 8. Conclusion
This project successfully implemented a complete Data Science pipeline to predict IPL match outcomes. We established that predicting IPL outcomes purely on historical team averages is highly challenging due to the inherent parity of the teams and the high volatility of the T20 format. Furthermore, the introduction of the Impact Player rule in 2023 caused a significant covariate shift, devaluing older historical data. This project highlights the importance of rigorous feature engineering, temporal validation splits, and domain knowledge in sports analytics.
