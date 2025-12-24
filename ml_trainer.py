# -*- coding: utf-8 -*-
"""
ML Model Trainer for Sniper Bot Pro.
Trains a simple machine learning model to predict trade outcomes.

Requirements:
    pip install scikit-learn pandas numpy

Usage:
    python ml_trainer.py                    # Train model
    python ml_trainer.py --evaluate         # Evaluate only
    python ml_trainer.py --predict          # Test predictions
"""

import csv
import json
import os
import pickle
from datetime import datetime

# Check for ML libraries
try:
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not installed. Run: pip install xgboost pandas scikit-learn")


class MLTrainer:
    """Trains and evaluates ML models for trading prediction."""
    
    def __init__(self, data_file: str = "ml_training_data.csv"):
        self.data_file = data_file
        self.model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.feature_columns = [
            'rsi', 'stoch_rsi', 'macd_hist', 'atr', 
            'obi', 'cvd', 'vol_ratio', 'vpin', 'liq_vol', 
            'funding', 'oi', 'sentiment', 'score'
        ]
    
    def load_data(self) -> tuple:
        """Load and prepare training data."""
        if not os.path.exists(self.data_file):
            # Try alternative file
            if os.path.exists("training_data.csv"):
                self.data_file = "training_data.csv"
                # Use same standard features for all files
            else:
                print(f"⚠️ Data file not found: {self.data_file}")
                return None, None
        
        X = []
        y = []
        
        with open(self.data_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    features = []
                    for col in self.feature_columns:
                        val = row.get(col, 0)
                        features.append(float(val) if val else 0)
                    
                    # Get label
                    label = int(row.get('label', row.get('outcome', 0)))
                    
                    X.append(features)
                    y.append(label)
                except (ValueError, KeyError) as e:
                    continue
        
        if not X:
            print("⚠️ No valid data found")
            return None, None
        
        print(f"✅ Loaded {len(X)} samples")
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the ML model."""
        if not ML_AVAILABLE:
            return {"error": "ML libraries not available"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        print("🔄 Training XGBoost (Super Intelligence)...")
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate with ROC AUC
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv)
        
        # Feature importance
        importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        results = {
            'accuracy': round(accuracy * 100, 2),
            'auc': round(auc, 4),
            'cv_mean': round(cv_scores.mean() * 100, 2),
            'cv_std': round(cv_scores.std() * 100, 2),
            'feature_importance': importance,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return results
    
    def save_model(self, filename: str = "trading_model.pkl"):
        """Save trained model to file."""
        if self.model is None:
            print("⚠️ No model to save")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.feature_columns,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filename}")
    
    def load_model(self, filename: str = "trading_model.pkl") -> bool:
        """Load model from file."""
        if not os.path.exists(filename):
            print(f"⚠️ Model file not found: {filename}")
            return False
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['features']
        
        print(f"✅ Model loaded from {filename}")
        return True
    
    def predict(self, features: dict) -> dict:
        """Make prediction with trained model."""
        if self.model is None:
            return {"error": "No model loaded"}
        
        # Extract features in correct order
        X = []
        for col in self.feature_columns:
            X.append(float(features.get(col, 0)))
        
        X = np.array([X])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': round(max(probability) * 100, 2),
            'prob_up': round(probability[1] * 100, 2) if len(probability) > 1 else 0,
            'prob_down': round(probability[0] * 100, 2)
        }
    
    def print_results(self, results: dict):
        """Print training results."""
        print("\n" + "="*50)
        print("🤖 ML MODEL TRAINING RESULTS")
        print("="*50)
        
        print(f"\n📊 ACCURACY & CONFIDENCE:")
        print(f"   Accuracy: {results['accuracy']}%")
        print(f"   ROC AUC Core: {results.get('auc', 0)}")
        print(f"   CV Mean: {results['cv_mean']}% (±{results['cv_std']}%)")
        
        print(f"\n🔑 FEATURE IMPORTANCE:")
        for feature, importance in results['feature_importance'].items():
            bar = "█" * int(importance * 50)
            print(f"   {feature:12s}: {importance:.3f} {bar}")
        
        print(f"\n📋 CONFUSION MATRIX:")
        cm = results['confusion_matrix']
        print(f"   Predicted:  DOWN   UP")
        print(f"   Actual DOWN: {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"   Actual UP:   {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        report = results['classification_report']
        print(f"\n📈 CLASSIFICATION REPORT:")
        print(f"   Precision (UP): {report.get('1', {}).get('precision', 0)*100:.1f}%")
        print(f"   Recall (UP):    {report.get('1', {}).get('recall', 0)*100:.1f}%")
        print(f"   F1-Score (UP):  {report.get('1', {}).get('f1-score', 0)*100:.1f}%")
        
        print("\n" + "="*50)


def main():
    import sys
    
    if not ML_AVAILABLE:
        print("\n❌ Cannot run without ML libraries.")
        print("   Install with: pip install scikit-learn numpy")
        return
    
    trainer = MLTrainer()
    
    if '--predict' in sys.argv:
        # Test prediction with sample data
        if trainer.load_model():
            sample = {
                'rsi': 35,
                'stoch_rsi': 20,
                'macd_hist': 0.5,
                'atr': 300,
                'bb_position': 1,
                'trend': 1,
                'vol_ratio': 1.5
            }
            result = trainer.predict(sample)
            print(f"\n🔮 Prediction: {result['direction']} ({result['confidence']}% confidence)")
    
    elif '--evaluate' in sys.argv:
        # Load and evaluate existing model
        if trainer.load_model():
            X, y = trainer.load_data()
            if X is not None:
                # Quick evaluation
                X_scaled = trainer.scaler.transform(X)
                y_pred = trainer.model.predict(X_scaled)
                acc = accuracy_score(y, y_pred)
                print(f"\n📊 Model accuracy on full dataset: {acc*100:.2f}%")
    
    else:
        # Train new model
        X, y = trainer.load_data()
        
        if X is not None:
            results = trainer.train(X, y)
            trainer.print_results(results)
            trainer.save_model()
            
            # Save results to JSON
            with open("ml_training_results.json", 'w') as f:
                # Convert numpy types to Python types for JSON
                json_results = {
                    'accuracy': results['accuracy'],
                    'cv_mean': results['cv_mean'],
                    'cv_std': results['cv_std'],
                    'feature_importance': {k: float(v) for k, v in results['feature_importance'].items()},
                }
                json.dump(json_results, f, indent=2)
            print("✅ Results saved to ml_training_results.json")


if __name__ == "__main__":
    main()
