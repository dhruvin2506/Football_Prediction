import joblib
import pandas as pd
import numpy as np
from typing import Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the raw match data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} matches from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def convert_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date strings to datetime and extract useful features"""
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday_period'] = df['date'].dt.month.isin([12, 1]).astype(int)
        df['match_number_in_season'] = df.groupby(df['date'].dt.year).cumcount()
        return df

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable (1: Home win, 0: Draw, -1: Away win)"""
        df['target'] = np.where(df['home_score'] > df['away_score'], 1,
                              np.where(df['home_score'] < df['away_score'], -1, 0))
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        return df

    def create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team form features based on recent matches"""
        df = df.sort_values('date')
        
        for team_type in ['home', 'away']:
            # Basic form features
            df[f'{team_type}_team_last_5_goals'] = df.groupby(f'{team_type}_team')[f'{team_type}_score'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            df[f'{team_type}_team_last_5_conceded'] = df.groupby(f'{team_type}_team')[f'{"away" if team_type == "home" else "home"}_score'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Win rate and streaks
            df[f'{team_type}_team_last_5_winrate'] = df.groupby(f'{team_type}_team')[f'{team_type}_win'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            df[f'{team_type}_team_streak'] = df.groupby(f'{team_type}_team')[f'{team_type}_win'].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum()
            )
            
            # Clean sheets
            df[f'{team_type}_clean_sheets'] = (df[f'{"away" if team_type == "home" else "home"}_score'] == 0).astype(int)
            df[f'{team_type}_clean_sheet_rate'] = df.groupby(f'{team_type}_team')[f'{team_type}_clean_sheets'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Goals per game
            df[f'{team_type}_goals_per_game'] = df.groupby(f'{team_type}_team')[f'{team_type}_score'].transform('mean')
            
            # Points calculation (3 for win, 1 for draw)
            df[f'{team_type}_points'] = np.where(df[f'{team_type}_win'] == True, 3, 
                                               np.where(df[f'{team_type}_win'].isna(), 1, 0))
            df[f'{team_type}_points_per_game'] = df.groupby(f'{team_type}_team')[f'{team_type}_points'].transform('mean')

        return df

    def create_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create head-to-head features for teams"""
        df = df.sort_values('date')
        
        def get_h2h_stats(group):
            if len(group) < 1:
                return pd.Series({
                    'h2h_home_wins': 0, 
                    'h2h_away_wins': 0, 
                    'h2h_draws': 0,
                    'h2h_total_goals': 0,
                    'h2h_avg_goals': 0
                })
            
            h2h_home_wins = sum((group['home_score'] > group['away_score']).astype(int))
            h2h_away_wins = sum((group['home_score'] < group['away_score']).astype(int))
            h2h_draws = sum((group['home_score'] == group['away_score']).astype(int))
            h2h_total_goals = group['home_score'].sum() + group['away_score'].sum()
            h2h_avg_goals = h2h_total_goals / len(group)
            
            return pd.Series({
                'h2h_home_wins': h2h_home_wins,
                'h2h_away_wins': h2h_away_wins,
                'h2h_draws': h2h_draws,
                'h2h_total_goals': h2h_total_goals,
                'h2h_avg_goals': h2h_avg_goals
            })
        
        h2h_stats = df.groupby(['home_team', 'away_team']).apply(get_h2h_stats)
        df = df.merge(h2h_stats, left_on=['home_team', 'away_team'], right_index=True, how='left')
        
        return df

    def create_team_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features indicating team quality"""
        for team_type in ['home', 'away']:
            # Goal difference
            df[f'{team_type}_goal_difference'] = df.groupby(f'{team_type}_team').apply(
                lambda x: x[f'{team_type}_score'].sum() - x[f'{"away" if team_type == "home" else "home"}_score'].sum()
            )
            
            # League position (approximation based on points)
            df[f'{team_type}_league_position'] = df.groupby(f'{team_type}_team')[f'{team_type}_points'].transform(
                lambda x: x.rank(ascending=False)
            )

        return df

    def scale_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                      feature_columns: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features using StandardScaler"""
        self.scaler.fit(train_data[feature_columns])
        
        train_scaled = pd.DataFrame(
            self.scaler.transform(train_data[feature_columns]),
            columns=feature_columns,
            index=train_data.index
        )
        
        test_scaled = pd.DataFrame(
            self.scaler.transform(test_data[feature_columns]),
            columns=feature_columns,
            index=test_data.index
        )
        
        return train_scaled, test_scaled

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training"""
        # Define features to use
        feature_columns = [
            'home_team_last_5_goals', 'home_team_last_5_conceded',
            'away_team_last_5_goals', 'away_team_last_5_conceded',
            'home_team_last_5_winrate', 'away_team_last_5_winrate',
            'home_team_streak', 'away_team_streak',
            'home_clean_sheet_rate', 'away_clean_sheet_rate',
            'home_goals_per_game', 'away_goals_per_game',
            'home_points_per_game', 'away_points_per_game',
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_avg_goals', 'h2h_total_goals',
            'home_goal_difference', 'away_goal_difference',
            'home_league_position', 'away_league_position',
            'is_weekend', 'month', 'is_holiday_period',
            'match_number_in_season'
        ]

        # Split data chronologically
        train_size = int(len(df) * (1 - test_size))
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(train_data, test_data, feature_columns)
        y_train = train_data['target']
        y_test = test_data['target']
        
        # Save the scaler
        os.makedirs('data/processed', exist_ok=True)
        joblib.dump(self.scaler, 'data/processed/scaler.pkl')
    
        return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/raw/matches_league_39_2023_20231218.csv')
    
    # Preprocess
    df = preprocessor.convert_date(df)
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.create_target_variable(df)
    df = preprocessor.create_form_features(df)
    df = preprocessor.create_head_to_head_features(df)
    df = preprocessor.create_team_quality_features(df)
    
    # Prepare for training
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Features used: {X_train.columns.tolist()}")

if __name__ == "__main__":
    main()