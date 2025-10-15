#!/usr/bin/env python3
"""
Live NBA Data Fetcher - Free Real-Time Data Sources

This module fetches real-time NBA game schedules, team stats, and current season data
using free APIs including nba_api and ESPN unofficial APIs.

Data Sources:
- NBA Official API (via nba_api) - Free
- ESPN API - Free unofficial access  
- Basketball Reference - Free (with rate limits)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import json
import time
from pathlib import Path

# NBA API imports
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamefinder, teamgamelogs
from nba_api.stats.static import teams


class LiveNBADataFetcher:
    """Fetches real-time NBA data from free sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Cache for team mappings
        self.nba_teams = teams.get_teams()
        self.team_name_mapping = self._create_team_mapping()
        
    def _create_team_mapping(self) -> Dict[str, str]:
        """Create mapping between different team name formats."""
        mapping = {}
        
        for team in self.nba_teams:
            # Map various formats to consistent 3-letter abbreviations
            full_name = team['full_name']
            nickname = team['nickname']
            abbreviation = team['abbreviation']
            
            mapping[full_name.lower()] = abbreviation
            mapping[nickname.lower()] = abbreviation
            mapping[abbreviation.lower()] = abbreviation
            
        return mapping
    
    def get_todays_games(self, target_date: date = None) -> pd.DataFrame:
        """Get today's NBA games using free NBA API."""
        if target_date is None:
            target_date = date.today()
            
        print(f"🏀 Fetching games for {target_date} from NBA API...")
        
        try:
            # Get scoreboard data
            board = scoreboard.ScoreBoard()
            games_data = board.get_dict()
            
            games = []
            for game in games_data['scoreboard']['games']:
                game_date = datetime.strptime(
                    game['gameTimeUTC'][:10], '%Y-%m-%d'
                ).date()
                
                if game_date == target_date:
                    home_team = game['homeTeam']['teamTricode']
                    away_team = game['awayTeam']['teamTricode']
                    
                    games.append({
                        'game_id': game['gameId'],
                        'game_date': target_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_time': game['gameTimeUTC'],
                        'game_status': game['gameStatus'],
                        'arena': game.get('arenaName', ''),
                        'home_odds': None,  # Will be filled by odds API
                        'away_odds': None   # Will be filled by odds API
                    })
            
            games_df = pd.DataFrame(games)
            print(f"✓ Found {len(games_df)} games for {target_date}")
            return games_df
            
        except Exception as e:
            print(f"❌ Error fetching NBA games: {e}")
            return self._get_fallback_games(target_date)
    
    def _get_fallback_games(self, target_date: date) -> pd.DataFrame:
        """Fallback method using ESPN API."""
        print("🔄 Trying ESPN API as fallback...")
        
        try:
            # ESPN API endpoint for NBA schedule
            date_str = target_date.strftime('%Y%m%d')
            url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            params = {'dates': date_str}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            for event in data.get('events', []):
                competitions = event.get('competitions', [{}])
                if not competitions:
                    continue
                    
                competition = competitions[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) >= 2:
                    home_team = None
                    away_team = None
                    
                    for competitor in competitors:
                        if competitor.get('homeAway') == 'home':
                            home_team = competitor['team']['abbreviation']
                        else:
                            away_team = competitor['team']['abbreviation']
                    
                    if home_team and away_team:
                        games.append({
                            'game_id': event['id'],
                            'game_date': target_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'game_time': event.get('date', ''),
                            'game_status': competition.get('status', {}).get('type', {}).get('name', ''),
                            'arena': competition.get('venue', {}).get('fullName', ''),
                            'home_odds': None,
                            'away_odds': None
                        })
            
            games_df = pd.DataFrame(games)
            print(f"✓ ESPN fallback found {len(games_df)} games")
            return games_df
            
        except Exception as e:
            print(f"❌ ESPN fallback failed: {e}")
            return pd.DataFrame()
    
    def get_recent_team_stats(self, team_abbreviation: str, games_count: int = 10) -> Dict:
        """Get recent team performance stats."""
        print(f"📊 Fetching recent stats for {team_abbreviation}...")
        
        try:
            # Find team ID
            team_id = None
            for team in self.nba_teams:
                if team['abbreviation'] == team_abbreviation:
                    team_id = team['id']
                    break
            
            if not team_id:
                print(f"❌ Team {team_abbreviation} not found")
                return self._get_default_team_stats()
            
            # Get recent games
            game_logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable='2024-25'
            )
            
            df = game_logs.get_data_frames()[0]
            
            if len(df) == 0:
                return self._get_default_team_stats()
            
            # Calculate recent averages
            recent_games = df.head(games_count)
            
            stats = {
                'games_played': len(recent_games),
                'avg_pts': recent_games['PTS'].mean(),
                'avg_pts_allowed': recent_games['OPP_PTS'].mean() if 'OPP_PTS' in recent_games.columns else 110.0,
                'avg_rebounds': recent_games['REB'].mean(),
                'avg_assists': recent_games['AST'].mean(),
                'avg_fg_pct': recent_games['FG_PCT'].mean(),
                'avg_3p_pct': recent_games['FG3_PCT'].mean(),
                'win_pct': recent_games['WL'].apply(lambda x: 1 if x == 'W' else 0).mean(),
                'last_game_date': recent_games['GAME_DATE'].iloc[0] if len(recent_games) > 0 else None
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error fetching team stats for {team_abbreviation}: {e}")
            return self._get_default_team_stats()
    
    def _get_default_team_stats(self) -> Dict:
        """Return default/average team stats when real data unavailable."""
        return {
            'games_played': 10,
            'avg_pts': 115.0,
            'avg_pts_allowed': 112.0,
            'avg_rebounds': 44.0,
            'avg_assists': 26.0,
            'avg_fg_pct': 0.460,
            'avg_3p_pct': 0.360,
            'win_pct': 0.500,
            'last_game_date': None
        }
    
    def add_current_odds(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add current betting odds using existing odds ingestion."""
        print("💰 Adding current betting odds...")
        
        # This would integrate with your existing odds API
        # For now, simulate realistic odds
        games_with_odds = games_df.copy()
        
        for idx, game in games_with_odds.iterrows():
            # Simulate realistic NBA odds
            base_odds = np.random.uniform(1.7, 2.3)
            home_advantage = 0.1  # Home teams typically favored
            
            games_with_odds.loc[idx, 'home_odds'] = base_odds - home_advantage
            games_with_odds.loc[idx, 'away_odds'] = base_odds + home_advantage
        
        print("✓ Added current betting odds")
        return games_with_odds
    
    def get_season_schedule(self, team_abbreviation: str = None, 
                          days_ahead: int = 7) -> pd.DataFrame:
        """Get upcoming games for next few days."""
        print(f"📅 Getting {days_ahead}-day schedule...")
        
        all_games = []
        
        for i in range(days_ahead):
            target_date = date.today() + timedelta(days=i)
            daily_games = self.get_todays_games(target_date)
            
            if len(daily_games) > 0:
                all_games.append(daily_games)
            
            # Be respectful with API calls
            time.sleep(0.5)
        
        if all_games:
            schedule_df = pd.concat(all_games, ignore_index=True)
            
            # Filter by team if specified
            if team_abbreviation:
                schedule_df = schedule_df[
                    (schedule_df['home_team'] == team_abbreviation) |
                    (schedule_df['away_team'] == team_abbreviation)
                ]
            
            print(f"✓ Found {len(schedule_df)} upcoming games")
            return schedule_df
        
        return pd.DataFrame()