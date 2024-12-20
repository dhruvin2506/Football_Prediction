import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_params: dict = None):
        """Initialize the model trainer with optional parameters"""
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        self.model = RandomForestClassifier(**self.model_params)
        self.scaler = None  # We'll get this from the preprocessing step
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model"""
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate model and create visualizations"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Create plots directory if it doesn't exist
        os.makedirs('models/plots', exist_ok=True)
        plt.savefig('models/plots/confusion_matrix.png')
        plt.close()
        
        # Feature importance plot
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('models/plots/feature_importance.png')
        plt.close()
        
        return feature_importance
    
    def save_model(self, filepath: str, scaler):
        """Save both the model and scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save the scaler
        scaler_filepath = filepath.replace('.pkl', '_scaler.pkl')
        joblib.dump(scaler, scaler_filepath)
        logger.info(f"Scaler saved to {scaler_filepath}")

def main():
    try:
        # Load preprocessed data
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').iloc[:, 0]
        y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]
        
        # Load the scaler used in preprocessing
        scaler = joblib.load('data/processed/scaler.pkl')
        
        logger.info(f"Loaded training data: {X_train.shape} samples")
        logger.info(f"Loaded test data: {X_test.shape} samples")
        
        # Initialize and train model
        trainer = ModelTrainer()
        trainer.train_model(X_train, y_train)
        
        # Evaluate model
        feature_importance = trainer.evaluate_model(X_test, y_test)
        
        # Save model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/football_predictor_{timestamp}.pkl'
        trainer.save_model(model_path, scaler)
        
        # Save as latest model
        trainer.save_model('models/football_predictor_latest.pkl', scaler)
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()