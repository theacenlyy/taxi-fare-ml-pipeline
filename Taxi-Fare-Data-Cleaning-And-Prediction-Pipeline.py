# ==============================================================================
# Cellula Technologies ML Track - Task 1 & 2
# Robust End-to-End Fare Prediction Architecture
# Engineer: Nagham
# ==============================================================================

import os
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FarePredictionPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.pipeline = None

    def load_and_sanitize_data(self):
        """Loads dataset and standardizes column schemas dynamically."""
        if not os.path.exists(self.data_path):
            logging.warning(f"File {self.data_path} not found. Synthesizing benchmark dataset...")
            self.raw_df = self._generate_synthetic_data()
        else:
            self.raw_df = pd.read_csv(self.data_path, encoding='utf-8-sig')

        # Header Normalization Map
        self.raw_df.columns = self.raw_df.columns.str.strip()
        column_mapping = {}
        for col in self.raw_df.columns:
            c = col.lower().replace('_', ' ').replace('-', ' ').strip()
            if 'passenger' in c: column_mapping[col] = 'Passenger_Count'
            elif 'fare' in c: column_mapping[col] = 'Fare_Amount'
            elif 'car' in c: column_mapping[col] = 'Car_Condition'
            elif 'traffic' in c: column_mapping[col] = 'Traffic_Conditions'
            elif 'hour' in c: column_mapping[col] = 'Hour'
            elif 'weather' in c: column_mapping[col] = 'Weather'
            elif 'jfk' in c: column_mapping[col] = 'JFK_Dist'
            elif 'ewr' in c: column_mapping[col] = 'EWR_Dist'
            elif 'lga' in c: column_mapping[col] = 'LGA_Dist'
            elif 'sol' in c: column_mapping[col] = 'SOL_Dist'
            elif 'dist' in c or 'distance' in c: column_mapping[col] = 'Distance'

        self.raw_df = self.raw_df.rename(columns=column_mapping)
        self.raw_df = self.raw_df.loc[:, ~self.raw_df.columns.duplicated()]
        
        # Deduplication
        initial_count = len(self.raw_df)
        self.raw_df.drop_duplicates(inplace=True)
        logging.info(f"Data Loaded: {len(self.raw_df)} rows ({initial_count - len(self.raw_df)} duplicates removed).")

    def _generate_synthetic_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            'Passenger Count': np.random.choice([1, 2, 3, 4, 5, np.nan], n),
            'Fare-Amount': np.random.uniform(5.0, 100.0, n),
            'Car_Condition': np.random.choice(['Good', 'Excellent', 'Fair', np.nan], n),
            'traffic_conditions': np.random.choice(['Low', 'Medium', 'High'], n),
            'Hour': np.random.randint(0, 24, n),
            'Weather': np.random.choice(['Sunny', 'Rainy', 'Snowy'], n),
            'Distance': np.random.uniform(0.5, 30.0, n),
            'JFK Dist': np.random.uniform(5.0, 25.0, n)
        })

    def build_preprocessor(self):
        """Constructs Transformer Pipeline ensuring zero data leakage."""
        num_cols = [c for c in ['Distance', 'JFK_Dist', 'EWR_Dist', 'LGA_Dist', 'SOL_Dist', 'Hour', 'Passenger_Count'] if c in self.X_train.columns]
        cat_cols = [c for c in ['Car_Condition', 'Traffic_Conditions', 'Weather'] if c in self.X_train.columns]

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

    def execute_pipeline(self):
        """Executes splitting, preprocessing, model selection, and hyperparameter tuning."""
        target = 'Fare_Amount'
        X = self.raw_df.drop(columns=[target])
        y = self.raw_df[target]

        # Train/Test Split Prior to Transformation (Prevent Leakage)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        self.build_preprocessor()

        # Full Pipeline Engine
        full_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('feature_selection', SelectFromModel(LassoCV(cv=5, random_state=42))),
            ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])

        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20, None],
            'regressor__min_samples_split': [2, 5],
            'regressor__max_features': ['sqrt', 'log2']
        }

        logging.info("Initiating RandomizedSearchCV Hyperparameter Tuning...")
        search = RandomizedSearchCV(
            full_pipeline,
            param_distributions=param_grid,
            n_iter=6,
            cv=3,
            scoring='neg_root_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(self.X_train, self.y_train)
        self.pipeline = search.best_estimator_

        # Evaluation
        y_pred = self.pipeline.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        logging.info(f"--- Final Evaluation Metrics ---")
        logging.info(f"MAE  : {mae:.4f}")
        logging.info(f"RMSE : {rmse:.4f}")
        logging.info(f"R² Score : {r2:.4f}")

    def export_artifact(self, output_filename: str = 'model_artifact.pkl'):
        """Serializes the complete inference-ready pipeline."""
        joblib.dump(self.pipeline, output_filename)
        logging.info(f"Artifact successfully exported to {output_filename}")

if __name__ == "__main__":
    pipeline_engine = FarePredictionPipeline(data_path='dataset.csv')
    pipeline_engine.load_and_sanitize_data()
    pipeline_engine.execute_pipeline()
    pipeline_engine.export_artifact('final_fare_prediction_pipeline.pkl')