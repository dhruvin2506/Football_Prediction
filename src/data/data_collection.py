import pandas as pd
import requests
from typing import Dict, List
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballDataCollector:
    def __init__(self):
        """
        Initialize the data collector with RapidAPI credentials
        """
        self.api_key = os.getenv('RAPID_API_KEY')
        self.api_host = os.getenv('RAPID_API_HOST')
        
        if not self.api_key or not self.api_host:
            raise ValueError("API credentials not found in .env file")
        
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.api_host
        }
        self.base_url = f"https://{self.api_host}/v3"

    def fetch_leagues(self) -> List[Dict]:
        """
        Fetch available leagues/competitions
        """
        try:
            url = f"{self.base_url}/leagues"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching leagues: {e}")
            return []

    def fetch_matches(self, league_id: int, season: int) -> List[Dict]:
        """
        Fetch match data for a specific league and season
        
        Args:
            league_id (int): League ID (e.g., 39 for Premier League)
            season (int): Season year (e.g., 2023)
        """
        try:
            url = f"{self.base_url}/fixtures"
            params = {
                "league": league_id,
                "season": season
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            return response.json()["response"]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching match data: {e}")
            return []

    def process_matches(self, matches: List[Dict]) -> pd.DataFrame:
        """
        Process raw match data into a structured DataFrame
        """
        processed_matches = []
        
        for match in matches:
            fixture = match['fixture']
            teams = match['teams']
            goals = match['goals']
            score = match['score']
            
            processed_match = {
                'match_id': fixture['id'],
                'date': fixture['date'],
                'timestamp': fixture['timestamp'],
                'venue': fixture['venue']['name'],
                'home_team': teams['home']['name'],
                'away_team': teams['away']['name'],
                'home_score': goals['home'],
                'away_score': goals['away'],
                'status': fixture['status']['long'],
                'home_win': teams['home']['winner'],
                'away_win': teams['away']['winner'],
                'halftime_home': score['halftime']['home'],
                'halftime_away': score['halftime']['away'],
                'fulltime_home': score['fulltime']['home'],
                'fulltime_away': score['fulltime']['away']
            }
            processed_matches.append(processed_match)
        
        return pd.DataFrame(processed_matches)

    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Save the processed data to a CSV file
        """
        output_dir = os.path.join('data', 'raw')
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")

def main():
    # Initialize collector
    collector = FootballDataCollector()
    
    # Example usage - Premier League (ID: 39)
    league_id = 39  # Premier League
    season = 2023
    
    # Fetch and process data
    matches = collector.fetch_matches(league_id, season)
    if matches:
        df = collector.process_matches(matches)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"matches_league_{league_id}_{season}_{timestamp}.csv"
        collector.save_data(df, filename)
        
        logger.info(f"Successfully collected {len(df)} matches")
    else:
        logger.warning("No matches data retrieved")

if __name__ == "__main__":
    main()