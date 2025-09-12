"""
Performance Monitoring and Tracking for NBA Betting Model

This module tracks prediction accuracy, betting performance, and model drift
over time. It provides analytics and triggers for model retraining.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import yaml
from dataclasses import dataclass, asdict
import sqlite3


@dataclass
class PredictionRecord:
    """Record of a single game prediction."""
    date: str
    game_id: str
    home_team: str
    away_team: str
    predicted_winner: str
    predicted_prob: float
    confidence: float
    actual_winner: Optional[str] = None
    correct_prediction: Optional[bool] = None
    bet_recommended: bool = False
    bet_amount: float = 0.0
    bet_odds: float = 0.0
    bet_result: Optional[str] = None  # 'win', 'loss', 'push'
    bet_payout: float = 0.0


@dataclass
class DailyPerformance:
    """Daily performance metrics."""
    date: str
    total_games: int
    predictions_made: int
    correct_predictions: int
    accuracy: float
    bets_placed: int
    total_stake: float
    total_payout: float
    net_profit: float
    roi: float


class PerformanceTracker:
    """Tracks and analyzes model and betting performance over time."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.db_path = db_path or self.project_root / "data" / "performance.db"
        self.db_path.parent.mkdir(exist_ok=True)
        
        self.reports_dir = self.project_root / "reports" / "performance"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for performance tracking."""
        with sqlite3.connect(self.db_path) as conn:
            # Predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    predicted_winner TEXT NOT NULL,
                    predicted_prob REAL NOT NULL,
                    confidence REAL NOT NULL,
                    actual_winner TEXT,
                    correct_prediction INTEGER,
                    bet_recommended INTEGER DEFAULT 0,
                    bet_amount REAL DEFAULT 0.0,
                    bet_odds REAL DEFAULT 0.0,
                    bet_result TEXT,
                    bet_payout REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, game_id)
                )
            """)
            
            # Daily performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_games INTEGER NOT NULL,
                    predictions_made INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    bets_placed INTEGER NOT NULL,
                    total_stake REAL NOT NULL,
                    total_payout REAL NOT NULL,
                    net_profit REAL NOT NULL,
                    roi REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    cv_accuracy REAL NOT NULL,
                    log_loss REAL,
                    feature_count INTEGER NOT NULL,
                    training_samples INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def record_predictions(self, predictions_df: pd.DataFrame, bet_data: pd.DataFrame = None):
        """Record daily predictions in the database."""
        print(f"📊 Recording {len(predictions_df)} predictions...")
        
        with sqlite3.connect(self.db_path) as conn:
            for _, pred in predictions_df.iterrows():
                # Check if bet was recommended
                bet_recommended = False
                bet_amount = 0.0
                bet_odds = 0.0
                
                if bet_data is not None:
                    bet_row = bet_data[bet_data['game_id'] == pred['game_id']]
                    if not bet_row.empty and bet_row.iloc[0].get('recommended_bet', False):
                        bet_recommended = True
                        bet_amount = bet_row.iloc[0].get('stake_amount', 0.0)
                        bet_odds = bet_row.iloc[0].get('best_bet_odds', 0.0)
                
                conn.execute("""
                    INSERT OR REPLACE INTO predictions 
                    (date, game_id, home_team, away_team, predicted_winner, 
                     predicted_prob, confidence, bet_recommended, bet_amount, bet_odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pred['game_date'].strftime('%Y-%m-%d') if hasattr(pred['game_date'], 'strftime') else str(pred['game_date']),
                    pred['game_id'],
                    pred['home_team'],
                    pred['away_team'],
                    pred['predicted_winner'],
                    float(pred['confidence']),
                    float(pred['confidence']),
                    int(bet_recommended),
                    float(bet_amount),
                    float(bet_odds)
                ))
            
            conn.commit()
        
        print("✓ Predictions recorded successfully")
    
    def update_actual_results(self, game_results: List[Dict]):
        """Update predictions with actual game results."""
        print(f"🎯 Updating {len(game_results)} game results...")
        
        with sqlite3.connect(self.db_path) as conn:
            for result in game_results:
                game_id = result['game_id']
                actual_winner = result['winner']
                
                # Update prediction accuracy
                conn.execute("""
                    UPDATE predictions 
                    SET actual_winner = ?,
                        correct_prediction = (predicted_winner = ?)
                    WHERE game_id = ?
                """, (actual_winner, actual_winner, game_id))
                
                # Update betting results if applicable
                if 'bet_result' in result:
                    bet_payout = result.get('bet_payout', 0.0)
                    conn.execute("""
                        UPDATE predictions 
                        SET bet_result = ?, bet_payout = ?
                        WHERE game_id = ?
                    """, (result['bet_result'], bet_payout, game_id))
            
            conn.commit()
        
        print("✓ Results updated successfully")
    
    def calculate_daily_performance(self, target_date: date) -> Optional[DailyPerformance]:
        """Calculate daily performance metrics."""
        date_str = target_date.strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            # Get daily prediction stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN correct_prediction = 1 THEN 1 ELSE 0 END) as correct_predictions,
                    COUNT(CASE WHEN actual_winner IS NOT NULL THEN 1 END) as games_with_results,
                    SUM(CASE WHEN bet_recommended = 1 THEN 1 ELSE 0 END) as bets_placed,
                    SUM(bet_amount) as total_stake,
                    SUM(bet_payout) as total_payout
                FROM predictions 
                WHERE date = ?
            """, (date_str,))
            
            row = cursor.fetchone()
            if not row or row[0] == 0:
                return None
            
            total_predictions = row[0]
            correct_predictions = row[1] or 0
            games_with_results = row[2] or 0
            bets_placed = row[3] or 0
            total_stake = row[4] or 0.0
            total_payout = row[5] or 0.0
            
            # Calculate metrics
            accuracy = correct_predictions / games_with_results if games_with_results > 0 else 0.0
            net_profit = total_payout - total_stake
            roi = net_profit / total_stake if total_stake > 0 else 0.0
            
            daily_perf = DailyPerformance(
                date=date_str,
                total_games=total_predictions,
                predictions_made=total_predictions,
                correct_predictions=correct_predictions,
                accuracy=accuracy,
                bets_placed=bets_placed,
                total_stake=total_stake,
                total_payout=total_payout,
                net_profit=net_profit,
                roi=roi
            )
            
            # Save to database
            conn.execute("""
                INSERT OR REPLACE INTO daily_performance 
                (date, total_games, predictions_made, correct_predictions, accuracy,
                 bets_placed, total_stake, total_payout, net_profit, roi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str, total_predictions, total_predictions, correct_predictions,
                accuracy, bets_placed, total_stake, total_payout, net_profit, roi
            ))
            
            conn.commit()
            
            return daily_perf
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for the last N days."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(correct_predictions) as total_correct,
                    AVG(accuracy) as avg_accuracy,
                    SUM(bets_placed) as total_bets,
                    SUM(total_stake) as total_stake,
                    SUM(total_payout) as total_payout,
                    SUM(net_profit) as total_profit
                FROM daily_performance 
                WHERE date BETWEEN ? AND ?
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            
            row = cursor.fetchone()
            
            if not row or row[0] == 0:
                return {'error': 'No data available for the specified period'}
            
            total_predictions = row[0] or 0
            total_correct = row[1] or 0
            avg_accuracy = row[2] or 0.0
            total_bets = row[3] or 0
            total_stake = row[4] or 0.0
            total_payout = row[5] or 0.0
            total_profit = row[6] or 0.0
            
            # Calculate ROI
            total_roi = total_profit / total_stake if total_stake > 0 else 0.0
            
            # Get recent trend
            cursor = conn.execute("""
                SELECT date, accuracy, roi 
                FROM daily_performance 
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
                LIMIT 7
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            
            recent_performance = cursor.fetchall()
            
            return {
                'period_days': days,
                'total_predictions': total_predictions,
                'total_correct': total_correct,
                'overall_accuracy': total_correct / total_predictions if total_predictions > 0 else 0.0,
                'average_daily_accuracy': avg_accuracy,
                'total_bets_placed': total_bets,
                'total_stake': total_stake,
                'total_payout': total_payout,
                'total_profit': total_profit,
                'total_roi': total_roi,
                'recent_performance': [
                    {'date': row[0], 'accuracy': row[1], 'roi': row[2]} 
                    for row in recent_performance
                ]
            }
    
    def check_model_drift(self, accuracy_threshold: float = 0.55, days: int = 7) -> Dict:
        """Check for model performance drift."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT AVG(accuracy) as recent_accuracy
                FROM daily_performance 
                WHERE date BETWEEN ? AND ?
                AND accuracy > 0
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            
            row = cursor.fetchone()
            recent_accuracy = row[0] if row and row[0] else 0.0
            
            # Get baseline accuracy (last 30 days before the recent period)
            baseline_start = start_date - timedelta(days=30)
            baseline_end = start_date
            
            cursor = conn.execute("""
                SELECT AVG(accuracy) as baseline_accuracy
                FROM daily_performance 
                WHERE date BETWEEN ? AND ?
                AND accuracy > 0
            """, (baseline_start.strftime('%Y-%m-%d'), baseline_end.strftime('%Y-%m-%d')))
            
            row = cursor.fetchone()
            baseline_accuracy = row[0] if row and row[0] else 0.0
            
            # Calculate drift
            accuracy_drop = baseline_accuracy - recent_accuracy if baseline_accuracy > 0 else 0.0
            needs_retraining = recent_accuracy < accuracy_threshold or accuracy_drop > 0.05
            
            return {
                'recent_accuracy': recent_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'accuracy_drop': accuracy_drop,
                'below_threshold': recent_accuracy < accuracy_threshold,
                'significant_drop': accuracy_drop > 0.05,
                'needs_retraining': needs_retraining,
                'recommendation': 'Retrain model' if needs_retraining else 'Continue monitoring'
            }
    
    def generate_performance_report(self, days: int = 30) -> Path:
        """Generate comprehensive performance report."""
        print(f"📈 Generating performance report for last {days} days...")
        
        summary = self.get_performance_summary(days)
        drift_analysis = self.check_model_drift()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': days,
            'summary': summary,
            'model_drift': drift_analysis,
            'recommendations': self._generate_recommendations(summary, drift_analysis)
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"performance_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Performance report saved: {report_file}")
        return report_file
    
    def _generate_recommendations(self, summary: Dict, drift_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on performance."""
        recommendations = []
        
        if 'error' in summary:
            recommendations.append("Insufficient data for analysis. Continue collecting performance data.")
            return recommendations
        
        # Accuracy recommendations
        if summary['overall_accuracy'] < 0.55:
            recommendations.append("Overall accuracy is below 55%. Consider retraining with more features or different algorithms.")
        elif summary['overall_accuracy'] > 0.65:
            recommendations.append("Excellent accuracy performance. Consider increasing bet sizes within risk limits.")
        
        # ROI recommendations
        if summary['total_roi'] < 0:
            recommendations.append("Negative ROI detected. Review betting strategy and edge calculations.")
        elif summary['total_roi'] > 0.1:
            recommendations.append("Strong ROI performance. Consider increasing bankroll allocation.")
        
        # Model drift recommendations
        if drift_analysis['needs_retraining']:
            recommendations.append("Model drift detected. Schedule retraining with recent data.")
        
        # Betting volume recommendations
        if summary['total_bets_placed'] == 0:
            recommendations.append("No bets placed recently. Check edge calculation and minimum thresholds.")
        
        if not recommendations:
            recommendations.append("Performance is stable. Continue current strategy.")
        
        return recommendations


def track_daily_performance(predictions_df: pd.DataFrame, betting_df: pd.DataFrame = None) -> Dict:
    """Track daily performance for given predictions and betting data."""
    tracker = PerformanceTracker()
    
    # Record predictions
    tracker.record_predictions(predictions_df, betting_df)
    
    # Calculate daily performance (will be 0 accuracy until results are updated)
    target_date = date.today()
    daily_perf = tracker.calculate_daily_performance(target_date)
    
    if daily_perf:
        return asdict(daily_perf)
    else:
        return {'message': 'Performance tracking initialized. Results will be available after games complete.'}


if __name__ == "__main__":
    # Example usage
    tracker = PerformanceTracker()
    summary = tracker.get_performance_summary(30)
    print(f"30-day performance: {summary.get('overall_accuracy', 0):.1%} accuracy")
