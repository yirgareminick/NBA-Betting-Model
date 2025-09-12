import joblib
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

# Load model and data
model = joblib.load('models/nba_model_latest.joblib')
df = pl.read_parquet('data/processed/nba_features.parquet').to_pandas()

# Prepare features
exclude_cols = ['game_id', 'game_date', 'team_name', 'opponent', 'target_win', 'venue']
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
if 'is_home' in df.columns:
    feature_cols.append('is_home')

X = df[feature_cols].copy()
y = df['target_win'].copy()

# Clean data
for col in X.select_dtypes(include=['bool']).columns:
    X[col] = X[col].astype(int)
X = X.fillna(X.mean(numeric_only=True))
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean(numeric_only=True))

# Get predictions and probabilities
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# Simulate betting strategy
min_edge = 0.03  # Minimum 3% edge to bet
assumed_odds = 1.91  # Typical NBA moneyline odds (around -110)
implied_prob = 1/assumed_odds  # ~52.4%

# Calculate edge and simulate bets
edges = probabilities - implied_prob
bet_mask = np.abs(edges) > min_edge  # Bet when we have >3% edge

if bet_mask.sum() > 0:
    bet_predictions = predictions[bet_mask]
    bet_actuals = y[bet_mask]
    bet_probabilities = probabilities[bet_mask]
    bet_edges = edges[bet_mask]
    
    # Calculate betting performance
    bet_accuracy = accuracy_score(bet_actuals, bet_predictions)
    num_bets = bet_mask.sum()
    avg_edge = np.mean(np.abs(bet_edges))
    
    # Simulate Kelly criterion bet sizing (fractional)
    kelly_fraction = 0.25  # Conservative 25% of Kelly
    bankroll = 10000
    avg_bet_size = bankroll * kelly_fraction * avg_edge
    
    # Simulate wins/losses (assuming we bet on our prediction)
    correct_bets = (bet_predictions == bet_actuals).sum()
    incorrect_bets = num_bets - correct_bets
    
    # Calculate P&L (simplified)
    win_payout = correct_bets * (assumed_odds - 1) * avg_bet_size  # Profit on wins
    loss_amount = incorrect_bets * avg_bet_size  # Amount lost
    net_profit = win_payout - loss_amount
    roi = (net_profit / (num_bets * avg_bet_size)) * 100
    
    print('BETTING SIMULATION RESULTS')
    print('========================')
    print(f'Total games analyzed: {len(df):,}')
    print(f'Overall model accuracy: {accuracy_score(y, predictions):.1%}')
    print('')
    print('BETTING PERFORMANCE:')
    print(f'Games meeting betting criteria (>{min_edge:.0%} edge): {num_bets:,}')
    print(f'Betting accuracy: {bet_accuracy:.1%}')
    print(f'Average edge: {avg_edge:.1%}')
    print(f'Average bet size: ${avg_bet_size:.0f}')
    print('')
    print('SIMULATED P&L:')
    print(f'Correct bets: {correct_bets}, Incorrect bets: {incorrect_bets}')
    print(f'Net profit: ${net_profit:,.0f}')
    print(f'ROI: {roi:+.1f}%')
    
    # Show distribution of edges
    high_edge_mask = np.abs(bet_edges) > 0.1  # >10% edge
    if high_edge_mask.sum() > 0:
        high_edge_accuracy = accuracy_score(bet_actuals[high_edge_mask], bet_predictions[high_edge_mask])
        print(f'High-edge bets (>10% edge): {high_edge_mask.sum()} games')
        print(f'High-edge accuracy: {high_edge_accuracy:.1%}')
        
else:
    print('No betting opportunities found with current criteria')
