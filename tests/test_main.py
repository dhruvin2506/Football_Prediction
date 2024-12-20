import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.utils.helpers import validate_data, calculate_roi

@pytest.fixture
def sample_match_data():
    """Create sample match data for testing"""
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-08', '2023-01-15'],
        'home_team': ['Team A', 'Team B', 'Team C'],
        'away_team': ['Team B', 'Team C', 'Team A'],
        'home_score': [2, 1, 0],
        'away_score': [1, 1, 2],
        'status': ['FINISHED', 'FINISHED', 'FINISHED']
    })

def test_data_preprocessor(sample_match_data):
    """Test the DataPreprocessor class"""
    preprocessor = DataPreprocessor()
    
    # Test create_target_variable
    df_with_target = preprocessor.create_target_variable(sample_match_data)
    assert 'target' in df_with_target.columns
    assert df_with_target['target'].iloc[0] == 1  # Home win
    assert df_with_target['target'].iloc[1] == 0  # Draw
    assert df_with_target['target'].iloc[2] == -1  # Away win

def test_feature_engineer(sample_match_data):
    """Test the FeatureEngineer class"""
    engineer = FeatureEngineer()
    
    # Test time-based features
    df_with_time = engineer.create_time_based_features(sample_match_data)
    assert 'day_of_week' in df_with_time.columns
    assert 'month' in df_with_time.columns
    assert 'is_weekend' in df_with_time.columns

def test_model_trainer():
    """Test the ModelTrainer class"""
    trainer = ModelTrainer()
    
    # Create dummy data
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10]
    })
    y = pd.Series([1, -1, 0, 1, -1])
    
    # Test model training
    trainer.train_model(X, y)
    assert trainer.model is not None
    
    # Test predictions
    predictions = trainer.predict_proba(X)
    assert predictions.shape[0] == len(X)
    assert predictions.shape[1] == 3  # Three classes (win, draw, loss)

def test_data_validation(sample_match_data):
    """Test data validation helper function"""
    required_columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    
    # Test with valid data
    assert validate_data(sample_match_data, required_columns) == True
    
    # Test with missing columns
    invalid_data = sample_match_data.drop('home_score', axis=1)
    assert validate_data(invalid_data, required_columns) == False

def test_roi_calculation():
    """Test ROI calculation helper function"""
    predictions = pd.DataFrame({
        'predicted_outcome': ['home_win', 'draw', 'away_win'],
        'odds': [1.5, 2.0, 2.5]
    })
    
    actual_results = pd.DataFrame({
        'result': ['home_win', 'draw', 'away_win']
    })
    
    roi_data = calculate_roi(predictions, actual_results, stake=100)
    
    assert roi_data['total_bets'] == 3
    assert roi_data['winning_bets'] == 3
    assert roi_data['total_stake'] == 300
    assert roi_data['total_returns'] == 600  # (1.5 + 2.0 + 2.5) * 100

if __name__ == '__main__':
    pytest.main([__file__])