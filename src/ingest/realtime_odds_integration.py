#!/usr/bin/env python3
"""
Real-Time Odds Integration for NBA Predictions

This script integrates current betting odds with live game data for real-time predictions.
It combines free NBA schedule data with your existing paid odds API.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, date
from typing import Dict, List, Optional
from pathlib import Path


class RealTimeOddsIntegrator:
    """Integrates real-time odds with live game schedule."""
    
    def __init__(self, odds_api_key: str = None):
        self.odds_api_key = odds_api_key or os.getenv('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        
    def get_current_nba_odds(self) -> pd.DataFrame:
        """Get current NBA odds from The Odds API."""
        if not self.odds_api_key:
            print("⚠️  No odds API key found, using simulated odds")
            return pd.DataFrame()
        
        print("💰 Fetching current NBA odds...")
        
        try:
            url = f"{self.base_url}/sports/basketball_nba/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h',  # head-to-head (moneyline)
                'oddsFormat': 'decimal',
                'bookmakers': 'draftkings,fanduel'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            odds_data = []
            
            for game in data:
                game_date = datetime.strptime(game['commence_time'][:10], '%Y-%m-%d').date()
                
                # Extract team names and odds
                teams = [game['home_team'], game['away_team']]
                bookmaker_odds = {}
                
                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            outcomes = market['outcomes']
                            for outcome in outcomes:
                                team_name = outcome['name']
                                odds = outcome['price']
                                
                                if bookmaker['key'] not in bookmaker_odds:
                                    bookmaker_odds[bookmaker['key']] = {}
                                bookmaker_odds[bookmaker['key']][team_name] = odds
                
                # Create odds record
                if bookmaker_odds:
                    # Use average odds across bookmakers
                    home_odds = []
                    away_odds = []
                    
                    for bm_data in bookmaker_odds.values():
                        if game['home_team'] in bm_data:
                            home_odds.append(bm_data[game['home_team']])
                        if game['away_team'] in bm_data:
                            away_odds.append(bm_data[game['away_team']])
                    
                    odds_data.append({
                        'game_date': game_date,
                        'home_team': self._normalize_team_name(game['home_team']),
                        'away_team': self._normalize_team_name(game['away_team']),
                        'home_odds': np.mean(home_odds) if home_odds else None,
                        'away_odds': np.mean(away_odds) if away_odds else None,
                        'last_updated': datetime.now()
                    })
            
            odds_df = pd.DataFrame(odds_data)
            print(f"✓ Fetched odds for {len(odds_df)} games")
            return odds_df
            
        except Exception as e:
            print(f"❌ Error fetching odds: {e}")
            return pd.DataFrame()
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Convert full team names to NBA abbreviations."""
        name_mapping = {
            # Eastern Conference - Atlantic Division
            'Boston Celtics': 'BOS',
            'Brooklyn Nets': 'BKN',
            'New York Knicks': 'NYK',
            'Philadelphia 76ers': 'PHI',
            'Toronto Raptors': 'TOR',
            
            # Eastern Conference - Central Division
            'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE',
            'Detroit Pistons': 'DET',
            'Indiana Pacers': 'IND',
            'Milwaukee Bucks': 'MIL',
            
            # Eastern Conference - Southeast Division
            'Atlanta Hawks': 'ATL',
            'Charlotte Hornets': 'CHA',
            'Miami Heat': 'MIA',
            'Orlando Magic': 'ORL',
            'Washington Wizards': 'WAS',
            
            # Western Conference - Northwest Division
            'Denver Nuggets': 'DEN',
            'Minnesota Timberwolves': 'MIN',
            'Oklahoma City Thunder': 'OKC',
            'Portland Trail Blazers': 'POR',
            'Utah Jazz': 'UTA',
            
            # Western Conference - Pacific Division
            'Golden State Warriors': 'GSW',
            'Los Angeles Clippers': 'LAC',
            'Los Angeles Lakers': 'LAL',
            'Phoenix Suns': 'PHX',
            'Sacramento Kings': 'SAC',
            
            # Western Conference - Southwest Division
            'Dallas Mavericks': 'DAL',
            'Houston Rockets': 'HOU',
            'Memphis Grizzlies': 'MEM',
            'New Orleans Pelicans': 'NOP',
            'San Antonio Spurs': 'SAS'
        }
        
        return name_mapping.get(team_name, team_name[:3].upper())
    
    def merge_games_with_odds(self, games_df: pd.DataFrame, odds_df: pd.DataFrame = None) -> pd.DataFrame:
        """Merge game schedule with current odds."""
        if odds_df is None or len(odds_df) == 0:
            # Use simulated realistic odds
            return self._add_simulated_odds(games_df)
        
        print("🔗 Merging games with current odds...")
        
        # Merge on date and team matchup
        merged = games_df.merge(
            odds_df,
            on=['game_date', 'home_team', 'away_team'],
            how='left'
        )
        
        # Fill missing odds with simulated values
        missing_odds = merged['home_odds'].isna()
        if missing_odds.any():
            print(f"⚠️  {missing_odds.sum()} games missing odds, using simulated values")
            merged = self._add_simulated_odds(merged, missing_only=True)
        
        print(f"✓ Merged {len(merged)} games with odds")
        return merged
    
    def _add_simulated_odds(self, games_df: pd.DataFrame, missing_only: bool = False) -> pd.DataFrame:
        """Add simulated realistic odds for development/fallback."""
        games_with_odds = games_df.copy()
        
        for idx, game in games_with_odds.iterrows():
            # Skip if odds already exist and we're only filling missing
            if missing_only and not pd.isna(game.get('home_odds')):
                continue
                
            # Simulate realistic NBA odds
            base_odds = np.random.uniform(1.7, 2.3)
            home_advantage = 0.1  # Home teams typically favored
            
            games_with_odds.loc[idx, 'home_odds'] = base_odds - home_advantage
            games_with_odds.loc[idx, 'away_odds'] = base_odds + home_advantage
        
        return games_with_odds\n    \n    def get_games_with_odds(self, target_date: date = None) -> pd.DataFrame:\n        """Get complete game data with current odds."""\n        from ingest.live_data_fetcher import LiveNBADataFetcher\n        \n        if target_date is None:\n            target_date = date.today()\n        \n        print(f"🏀 Getting complete game data for {target_date}...")\n        \n        # Get game schedule\n        fetcher = LiveNBADataFetcher()\n        games = fetcher.get_todays_games(target_date)\n        \n        if len(games) == 0:\n            print("📭 No games found for today")\n            return pd.DataFrame()\n        \n        # Get current odds\n        odds = self.get_current_nba_odds()\n        \n        # Merge games with odds\n        complete_data = self.merge_games_with_odds(games, odds)\n        \n        print(f"✅ Complete data ready for {len(complete_data)} games")\n        return complete_data\n\n\ndef test_real_time_integration():\n    """Test the real-time odds integration."""\n    print("🧪 Testing Real-Time Odds Integration")\n    print("=" * 50)\n    \n    integrator = RealTimeOddsIntegrator()\n    \n    # Test odds fetching (will use simulated since no API key in demo)\n    print("\\n1. Testing odds fetching:")\n    odds = integrator.get_current_nba_odds()\n    print(f"Found odds for {len(odds)} games")\n    \n    # Test complete integration\n    print("\\n2. Testing complete integration:")\n    try:\n        complete_data = integrator.get_games_with_odds()\n        print(f"Complete data for {len(complete_data)} games")\n        \n        if len(complete_data) > 0:\n            print("\\nSample game data:")\n            sample_cols = ['home_team', 'away_team', 'home_odds', 'away_odds']\n            available_cols = [col for col in sample_cols if col in complete_data.columns]\n            print(complete_data[available_cols].head())\n            \n    except Exception as e:\n        print(f"Integration test failed: {e}")\n    \n    print("\\n✅ Real-time integration test completed!")\n\n\nif __name__ == "__main__":\n    test_real_time_integration()
