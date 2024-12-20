import pandas as pd
import numpy as np
from typing import Dict, List, Union
import logging
from datetime import datetime, timedelta
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def calculate_roi(predictions: pd.DataFrame, actual_results: pd.DataFrame, 
                 stake: float = 100.0) -> Dict[str, float]:
    """
    Calculate Return on Investment for predictions
    Args:
        predictions: DataFrame with predicted probabilities
        actual_results: DataFrame with actual match results
        stake: Amount bet on each game
    """
    roi_data = {
        'total_bets': len(predictions),
        'winning_bets': 0,
        'total_stake': stake * len(predictions),
        'total_returns': 0.0
    }
    
    for pred, actual in zip(predictions.itertuples(), actual_results.itertuples()):
        if pred.predicted_outcome == actual.result:
            roi_data['winning_bets'] += 1
            roi_data['total_returns'] += stake * pred.odds
            
    roi_data['roi_percentage'] = ((roi_data['total_returns'] - roi_data['total_stake']) 
                                 / roi_data['total_stake'] * 100)
    
    return roi_data

def create_betting_odds(probabilities: Dict[str, float]) -> Dict[str, float]:
    """Convert probabilities to betting odds"""
    odds = {}
    for outcome, prob in probabilities.items():
        # Convert probability to decimal odds (including margin)
        margin = 1.1  # 10% margin
        if prob > 0:
            odds[outcome] = round((1 / (prob / 100) * margin), 2)
        else:
            odds[outcome] = 0
    return odds

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame contains required columns"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def get_recent_matches(df: pd.DataFrame, team: str, n_matches: int = 5) -> pd.DataFrame:
    """Get the most recent matches for a specific team"""
    team_matches = df[
        (df['home_team'] == team) | (df['away_team'] == team)
    ].sort_values('date', ascending=False).head(n_matches)
    
    return team_matches

def create_performance_summary(team_matches: pd.DataFrame, team: str) -> Dict:
    """Create a summary of team's recent performance"""
    summary = {
        'matches_played': len(team_matches),
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'goals_scored': 0,
        'goals_conceded': 0
    }
    
    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            goals_scored = match['home_score']
            goals_conceded = match['away_score']
        else:
            goals_scored = match['away_score']
            goals_conceded = match['home_score']
            
        summary['goals_scored'] += goals_scored
        summary['goals_conceded'] += goals_conceded
        
        if goals_scored > goals_conceded:
            summary['wins'] += 1
        elif goals_scored < goals_conceded:
            summary['losses'] += 1
        else:
            summary['draws'] += 1
            
    return summary

def save_predictions_report(predictions: List[Dict], filepath: str):
    """Save detailed predictions report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{filepath}_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("Football Match Predictions Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for idx, pred in enumerate(predictions, 1):
            f.write(f"Match {idx}:\n")
            for key, value in pred.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
    logger.info(f"Predictions report saved to {report_path}")

def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)