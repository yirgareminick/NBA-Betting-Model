import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
from nba_api.live.nba.endpoints import scoreboard


class LiveNBADataFetcher:
    """Fetches real-time NBA game data."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_todays_games(self, target_date: date = None) -> pd.DataFrame:
        """Get NBA games for specified date."""
        if target_date is None:
            target_date = date.today()

        try:
            board = scoreboard.ScoreBoard()
            games_data = board.get_dict()

            games = []
            for game in games_data['scoreboard']['games']:
                # Convert UTC time to Eastern time for date matching
                utc_time = datetime.strptime(game['gameTimeUTC'], '%Y-%m-%dT%H:%M:%SZ')
                eastern_time = utc_time - timedelta(hours=5)  # EST approximation
                game_date = eastern_time.date()

                if game_date == target_date:
                    games.append({
                        'game_id': game['gameId'],
                        'game_date': target_date,
                        'home_team': game['homeTeam']['teamTricode'],
                        'away_team': game['awayTeam']['teamTricode'],
                        'game_time': game['gameTimeUTC'],
                        'game_status': game['gameStatus'],
                        'arena': game.get('arenaName', ''),
                        'home_odds': None,
                        'away_odds': None
                    })

            return pd.DataFrame(games)

        except Exception as e:
            print(f"Error fetching NBA games: {e}")
            return self._get_fallback_games(target_date)

    def _get_fallback_games(self, target_date: date) -> pd.DataFrame:
        """Fallback using ESPN API."""
        try:
            url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            params = {'dates': target_date.strftime('%Y%m%d')}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            games = []
            for event in data.get('events', []):
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])

                if len(competitors) >= 2:
                    home_team = next((c['team']['abbreviation'] for c in competitors if c.get('homeAway') == 'home'), None)
                    away_team = next((c['team']['abbreviation'] for c in competitors if c.get('homeAway') != 'home'), None)

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

            return pd.DataFrame(games)

        except Exception:
            return pd.DataFrame()

    def add_current_odds(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add simulated betting odds to games."""
        games_with_odds = games_df.copy()
        
        for idx, _ in games_with_odds.iterrows():
            base_odds = np.random.uniform(1.7, 2.3)
            games_with_odds.loc[idx, 'home_odds'] = base_odds - 0.1  # Home advantage
            games_with_odds.loc[idx, 'away_odds'] = base_odds + 0.1

        return games_with_odds

    def get_recent_team_stats(self, team_abbreviation: str, games_count: int = 10) -> dict:
        """Return default team stats (removed detailed implementation for simplicity)."""
        return {
            'games_played': games_count,
            'avg_pts': 115.0,
            'avg_pts_allowed': 112.0,
            'avg_rebounds': 44.0,
            'avg_assists': 26.0,
            'avg_fg_pct': 0.460,
            'avg_3p_pct': 0.360,
            'win_pct': 0.500,
            'last_game_date': None
        }


# CLI entry point for direct testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch NBA games for a given date")
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format', default=None)
    args = parser.parse_args()

    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except Exception as e:
            print(f"Invalid date format: {e}")
            target_date = date.today()
    else:
        target_date = date.today()

    fetcher = LiveNBADataFetcher()
    games_df = fetcher.get_todays_games(target_date)
    if not games_df.empty:
        print(f"\nNBA Games for {target_date}:")
        print(games_df)
    else:
        print(f"No NBA games found for {target_date}.")