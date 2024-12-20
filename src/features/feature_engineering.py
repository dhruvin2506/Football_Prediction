import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def calculate_team_stats(self, df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """Calculate rolling team statistics"""
        df = df.sort_values('date')
        stats_df = df.copy()
        
        # Calculate rolling stats for each team
        for team_type in ['home', 'away']:
            # Goals scored and conceded
            stats_df[f'{team_type}_rolling_goals_scored'] = df.groupby(f'{team_type}_team')[f'{team_type}_score'].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
            stats_df[f'{team_type}_rolling_goals_conceded'] = df.groupby(f'{team_type}_team')[f'{"away" if team_type == "home" else "home"}_score'].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
            
            # Win rate
            stats_df[f'{team_type}_rolling_win_rate'] = df.groupby(f'{team_type}_team')[f'{team_type}_win'].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
            
        return stats_df

    def create_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on historical head-to-head matches"""
        df = df.sort_values('date')
        h2h_df = df.copy()
        
        def get_h2h_stats(group):
            if len(group) < 2:
                return pd.Series({
                    'h2h_home_wins': 0,
                    'h2h_away_wins': 0,
                    'h2h_draws': 0,
                    'h2h_avg_goals': 0
                })
            
            h2h_home_wins = sum((group['home_score'] > group['away_score']).astype(int))
            h2h_away_wins = sum((group['home_score'] < group['away_score']).astype(int))
            h2h_draws = sum((group['home_score'] == group['away_score']).astype(int))
            h2h_avg_goals = (group['home_score'] + group['away_score']).mean()
            
            return pd.Series({
                'h2h_home_wins': h2h_home_wins,
                'h2h_away_wins': h2h_away_wins,
                'h2h_draws': h2h_draws,
                'h2h_avg_goals': h2h_avg_goals
            })
        
        # Create head-to-head features
        h2h_stats = df.groupby(['home_team', 'away_team']).apply(get_h2h_stats)
        
        # Merge stats back to original dataframe
        h2h_df = h2h_df.merge(h2h_stats, left_on=['home_team', 'away_team'], right_index=True, how='left')
        
        return h2h_df

    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on time components"""
        time_df = df.copy()
        
        # Convert date to datetime if it's not already
        time_df['date'] = pd.to_datetime(time_df['date'])
        
        # Extract time components
        time_df['day_of_week'] = time_df['date'].dt.dayofweek
        time_df['month'] = time_df['date'].dt.month
        time_df['is_weekend'] = time_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate days since last match for each team
        for team_type in ['home', 'away']:
            time_df[f'{team_type}_days_since_last_match'] = time_df.groupby(f'{team_type}_team')['date'].diff().dt.days
            
        return time_df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and combine all relevant features for model training"""
        feature_df = df.copy()
        
        self.feature_names = [
            'home_rolling_goals_scored', 'home_rolling_goals_conceded',
            'away_rolling_goals_scored', 'away_rolling_goals_conceded',
            'home_rolling_win_rate', 'away_rolling_win_rate',
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_avg_goals',
            'is_weekend', 'month',
            'home_days_since_last_match', 'away_days_since_last_match'
        ]
        
        return feature_df[self.feature_names]

def main():
    # Example usage
    engineer = FeatureEngineer()
    
    # Load preprocessed data
    df = pd.read_csv('data/processed/preprocessed_matches.csv')
    
    # Create features
    df = engineer.calculate_team_stats(df)
    df = engineer.create_head_to_head_features(df)
    df = engineer.create_time_based_features(df)
    
    # Select features
    features_df = engineer.select_features(df)
    
    # Save engineered features
    features_df.to_csv('data/processed/engineered_features.csv', index=False)
    logger.info(f"Created {len(engineer.feature_names)} features")

if __name__ == "__main__":
    main()