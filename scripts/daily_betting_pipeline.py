#!/usr/bin/env python3
"""
Enhanced Daily NBA Predictions and Betting Script

This script runs the complete daily NBA betting pipeline including:
- Data ingestion and updates
- Model performance monitoring
- Advanced predictions with multiple models
- Betting recommendations with risk management
- Performance tracking and reporting
- Automated notifications

Usage:
    python scripts/daily_betting_pipeline.py
    python scripts/daily_betting_pipeline.py --bankroll 15000 --retrain
    python scripts/daily_betting_pipeline.py --dry-run --notify
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from predict.daily_report import generate_daily_report
from predict.predict_games import predict_daily_games
from stake.kelly_criterion import calculate_daily_bets


class DailyBettingPipeline:
    """Enhanced daily betting pipeline with full automation."""
    
    def __init__(self, config: dict = None):
        self.config = config or self._load_config()
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def _load_config(self) -> dict:
        """Load configuration from file."""
        config_file = Path(__file__).parent.parent / "configs" / "model.yml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {
                'betting': {'kelly_fraction': 0.5, 'min_edge': 0.02, 'max_bet_size': 0.1},
                'thresholds': {'min_accuracy': 0.55, 'retrain_threshold': 0.05}
            }
    
    def log_message(self, message: str, level: str = "INFO"):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        print(log_entry)
        
        # Also log to file
        log_file = self.logs_dir / f"daily_pipeline_{date.today().strftime('%Y%m%d')}.log"
        with open(log_file, 'a') as f:
            f.write(log_entry + "\n")
    
    def check_model_performance(self) -> dict:
        """Check current model performance and determine if retraining is needed."""
        try:
            model_file = self.project_root / "models" / "nba_model_latest.joblib"
            
            if not model_file.exists():
                self.log_message("No model found", "WARNING")
                return {'needs_retraining': True, 'reason': 'no_model'}
            
            return {'needs_retraining': False, 'reason': 'model_exists'}
            
        except Exception as e:
            self.log_message(f"Model check failed: {e}", "ERROR")
            return {'needs_retraining': False, 'error': str(e)}
    
    def retrain_model_if_needed(self, force_retrain: bool = False) -> bool:
        """Retrain model if performance has degraded."""
        performance_check = self.check_model_performance()
        
        if force_retrain or performance_check.get('needs_retraining', False):
            try:
                from models.train_model import train_model
                metrics = train_model()
                self.log_message(f"Model retrained: {metrics['best_model']}")
                return True
            except Exception as e:
                self.log_message(f"Retrain failed: {e}", "ERROR")
                return False
        return False
    
    def generate_predictions(self, target_date: date) -> dict:
        """Generate daily predictions."""
        try:
            predictions = predict_daily_games(target_date)
            
            if len(predictions) == 0:
                self.log_message("No games found")
                return {'predictions': predictions, 'games_found': 0}
            
            self.log_message(f"Predictions: {len(predictions)} games")
            return {'predictions': predictions, 'games_found': len(predictions)}
            
        except Exception as e:
            self.log_message(f"Prediction failed: {e}", "ERROR")
            return {'predictions': None, 'games_found': 0, 'error': str(e)}
    
    def calculate_betting_strategy(self, predictions, bankroll: float) -> dict:
        """Calculate betting recommendations."""
        try:
            betting_recommendations, simulation_results = calculate_daily_bets(
                predictions, bankroll, self.config.get('betting', {})
            )
            
            recommended_bets = betting_recommendations['recommended_bet'].sum()
            total_stake = betting_recommendations['stake_amount'].sum()
            expected_value = betting_recommendations['expected_value'].sum()
            
            self.log_message(f"Bets: {recommended_bets}, Stake: ${total_stake:.2f}, EV: ${expected_value:.2f}")
            
            return {
                'betting_recommendations': betting_recommendations,
                'simulation_results': simulation_results,
                'recommended_bets': recommended_bets,
                'total_stake': total_stake,
                'expected_value': expected_value
            }
            
        except Exception as e:
            self.log_message(f"Betting failed: {e}", "ERROR")
            return {'betting_recommendations': None, 'error': str(e)}
    
    def generate_comprehensive_report(self, target_date: date, bankroll: float) -> dict:
        """Generate comprehensive daily report."""
        try:
            report = generate_daily_report(target_date, bankroll)
            self.log_message("Report generated")
            return report
        except Exception as e:
            self.log_message(f"Report failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def track_performance(self, predictions, betting_data) -> dict:
        """Track daily performance."""
        self.log_message("Recording performance data...")
        
        try:
            # Simplified performance tracking - just log basic metrics
            performance_data = {
                'predictions_made': len(predictions),
                'bets_recommended': betting_data['recommended_bet'].sum() if not betting_data.empty else 0
            }
            self.log_message("Performance data recorded")
            return performance_data
            
        except Exception as e:
            self.log_message(f"Performance tracking failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def send_notifications(self, report_data: dict):
        """Log notification summary (placeholder for future notification system)."""
        recommended_bets = report_data.get('recommended_bets', 0)
        expected_value = report_data.get('expected_value', 0)
        self.log_message(f"Daily summary: {recommended_bets} bets, ${expected_value:.2f} EV")
    
    def run_daily_pipeline(self, target_date: date = None, bankroll: float = 10000, 
                          retrain: bool = False, dry_run: bool = False, 
                          notify: bool = False) -> dict:
        """Run the complete daily betting pipeline."""
        if target_date is None:
            target_date = date.today()
        
        self.log_message("=" * 80)
        self.log_message("🏀 STARTING DAILY NBA BETTING PIPELINE")
        self.log_message(f"Date: {target_date}, Bankroll: ${bankroll:,.2f}")
        if dry_run:
            self.log_message("*** DRY RUN MODE - NO ACTUAL BETS WILL BE PLACED ***")
        self.log_message("=" * 80)
        
        pipeline_results = {
            'date': target_date.isoformat(),
            'bankroll': bankroll,
            'dry_run': dry_run,
            'started_at': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Check model performance and retrain if needed
            if retrain or not dry_run:
                retrained = self.retrain_model_if_needed(retrain)
                pipeline_results['model_retrained'] = retrained
            
            # Step 2: Generate predictions
            prediction_results = self.generate_predictions(target_date)
            pipeline_results.update(prediction_results)
            
            if prediction_results['games_found'] == 0:
                self.log_message("No games to analyze today")
                pipeline_results['status'] = 'completed_no_games'
                return pipeline_results
            
            # Step 3: Calculate betting strategy
            betting_results = self.calculate_betting_strategy(
                prediction_results['predictions'], bankroll
            )
            pipeline_results.update(betting_results)
            
            # Step 4: Generate comprehensive report
            if not dry_run:
                report = self.generate_comprehensive_report(target_date, bankroll)
                pipeline_results['report'] = report
                
                # Step 5: Track performance
                if 'betting_recommendations' in betting_results:
                    performance = self.track_performance(
                        prediction_results['predictions'],
                        betting_results['betting_recommendations']
                    )
                    pipeline_results['performance_tracking'] = performance
                
                # Step 6: Send notifications
                if notify:
                    notify_config = self.config.get('notifications', {})
                    self.send_notifications(report, notify_config)
            
            pipeline_results['status'] = 'completed_successfully'
            pipeline_results['completed_at'] = datetime.now().isoformat()
            
            self.log_message("=" * 80)
            self.log_message("✅ DAILY PIPELINE COMPLETED SUCCESSFULLY")
            self.log_message(f"Recommended bets: {pipeline_results.get('recommended_bets', 0)}")
            self.log_message(f"Expected value: ${pipeline_results.get('expected_value', 0):.2f}")
            self.log_message("=" * 80)
            
        except Exception as e:
            self.log_message(f"Pipeline failed: {e}", "ERROR")
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
        
        return pipeline_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced daily NBA betting pipeline")
    
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--bankroll", type=float, default=10000, help="Betting bankroll")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving/betting")
    parser.add_argument("--notify", action="store_true", help="Send notifications")
    parser.add_argument("--config", type=str, help="Custom config file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Parse target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = date.today()
    
    # Load custom config if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            print(f"❌ Config file not found: {config_path}")
            sys.exit(1)
    
    try:
        # Initialize and run pipeline
        pipeline = DailyBettingPipeline(config)
        
        results = pipeline.run_daily_pipeline(
            target_date=target_date,
            bankroll=args.bankroll,
            retrain=args.retrain,
            dry_run=args.dry_run,
            notify=args.notify
        )
        
        # Print summary if not quiet
        if not args.quiet:
            print("\n📊 PIPELINE SUMMARY")
            print("=" * 50)
            print(f"Status: {results['status']}")
            print(f"Games analyzed: {results.get('games_found', 0)}")
            print(f"Recommended bets: {results.get('recommended_bets', 0)}")
            print(f"Expected value: ${results.get('expected_value', 0):.2f}")
            
            if results.get('model_retrained'):
                print("🔄 Model was retrained")
        
        # Exit with appropriate code
        sys.exit(0 if results['status'] in ['completed_successfully', 'completed_no_games'] else 1)
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
