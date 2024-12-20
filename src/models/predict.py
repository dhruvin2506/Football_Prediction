import pandas as pd
import joblib
import logging
from typing import Dict, List
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self, model_path: str):
        """Initialize the predictor with a trained model"""
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def predict_match(self, features: pd.DataFrame) -> Dict:
        """
        Predict the outcome of a match
        Returns probabilities for home win, draw, and away win
        """
        try:
            # Get probability predictions
            probabilities = self.model.predict_proba(features)
            
            # Create prediction dictionary
            prediction = {
                'home_win_probability': round(probabilities[0][2] * 100, 2),
                'draw_probability': round(probabilities[0][1] * 100, 2),
                'away_win_probability': round(probabilities[0][0] * 100, 2)
            }
            
            # Get predicted outcome
            prediction['predicted_outcome'] = max(prediction.items(), key=lambda x: x[1])[0]
            
            return prediction
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
            
    def predict_multiple_matches(self, features_df: pd.DataFrame) -> List[Dict]:
        """Predict outcomes for multiple matches"""
        predictions = []
        
        for idx, row in features_df.iterrows():
            match_features = pd.DataFrame([row])
            prediction = self.predict_match(match_features)
            predictions.append(prediction)
            
        return predictions

    def save_predictions(self, predictions: List[Dict], filepath: str):
        """Save predictions to CSV"""
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(filepath, index=False)
        logger.info(f"Predictions saved to {filepath}")

def main():
    # Example usage
    model_path = 'models/random_forest_latest.pkl'
    predictor = MatchPredictor(model_path)
    
    # Load new match features
    new_matches = pd.read_csv('data/processed/new_matches_features.csv')
    
    # Make predictions
    predictions = predictor.predict_multiple_matches(new_matches)
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'data/predictions/predictions_{timestamp}.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictor.save_predictions(predictions, output_path)

if __name__ == "__main__":
    main()