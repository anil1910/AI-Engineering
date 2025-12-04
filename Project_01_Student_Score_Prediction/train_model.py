"""
Student Score Prediction System - Regression Model Training
This script demonstrates linear and polynomial regression with MLflow tracking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class StudentScorePredictor:
    """Regression model for predicting student scores."""
    
    def __init__(self, data_path):
        """Initialize predictor with data path."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Load CSV data and preprocess it for regression."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Use Target_Score from CSV (has quadratic characteristics)
        if 'Target_Score' not in self.df.columns:
            raise ValueError("Target_Score column not found! Run create_target_variable.py first.")
        
        self.df['Score'] = self.df['Target_Score']
        
        # Handle missing values in Scholarship column
        self.df['Scholarship'] = self.df['Scholarship'].fillna('0%')
        
        print("\n" + "="*60)
        print("Exploratory Data Analysis (EDA)")
        print("="*60)
        
        # Display basic statistics
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nTarget variable (Score) statistics:")
        print(self.df['Score'].describe())
        
        print(f"\nTarget Score range: {self.df['Score'].min():.1f} - {self.df['Score'].max():.1f}")
        print(f"Mean: {self.df['Score'].mean():.2f} | Std: {self.df['Score'].std():.2f}")
        
        print("\n💡 Note: Target_Score has QUADRATIC characteristics:")
        print("   • Linear component for basic trends")
        print("   • Quadratic component (squared terms & interactions)")
        print("   • Polynomial (degree 2) should significantly outperform Linear!")
        
        # Check for missing values
        print(f"\nMissing values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
        
        # Convert categorical variables to numerical
        label_encoders = {}
        categorical_cols = ['Sex', 'High_School_Type', 'Scholarship', 'Additional_Work', 
                          'Sports_activity', 'Transportation', 'Reading', 'Notes', 
                          'Listening_in_Class', 'Project_work', 'Attendance']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                label_encoders[col] = le
        
        # Convert age ranges to numerical
        age_mapping = {
            '18': 18,
            '19-22': 20.5,
            '23-27': 25
        }
        self.df['Student_Age'] = self.df['Student_Age'].map(age_mapping)
        
        # Select features for regression
        feature_cols = ['Weekly_Study_Hours', 'Attendance', 'Reading', 
                       'Listening_in_Class', 'Project_work']
        
        X = self.df[feature_cols]
        y = self.df['Score']
        
        self.feature_names = feature_cols
        
        # Show correlation with target
        print(f"\nCorrelation with Score:")
        correlations = self.df[feature_cols + ['Score']].corr()['Score'].sort_values(ascending=False)
        print(correlations)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n" + "="*60)
        print(f"Data loaded: {len(self.df)} samples")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        print("="*60)
        
        return self
    
    def train_linear_regression(self, experiment_name="Student_Score_Prediction"):
        """Train linear regression model with MLflow tracking."""
        print("\n" + "="*60)
        print("Training Linear Regression Model")
        print("="*60)
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="Linear_Regression"):
            # Train model
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # Log parameters
            mlflow.log_param("model_type", "Linear Regression")
            mlflow.log_param("n_features", self.X_train.shape[1])
            mlflow.log_param("train_samples", len(self.X_train))
            mlflow.log_param("test_samples", len(self.X_test))
            
            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_mae", test_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "linear_model")
            
            # Print results
            print(f"\nTraining Metrics:")
            print(f"  RMSE: {train_rmse:.2f}")
            print(f"  MAE: {train_mae:.2f}")
            print(f"  R² Score: {train_r2:.4f}")
            
            print(f"\nTest Metrics:")
            print(f"  RMSE: {test_rmse:.2f}")
            print(f"  MAE: {test_mae:.2f}")
            print(f"  R² Score: {test_r2:.4f}")
            
            # Visualize results
            self._plot_predictions(self.y_test, y_test_pred, "Linear Regression")
            plt.savefig('models/linear_regression_predictions.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('models/linear_regression_predictions.png')
            
            # Feature importance
            self._plot_feature_importance(model.coef_, self.feature_names, "Linear Regression")
            plt.savefig('models/linear_regression_features.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('models/linear_regression_features.png')
            
        return model, test_r2
    
    def train_polynomial_regression(self, degree=2, experiment_name="Student_Score_Prediction"):
        """Train polynomial regression model with MLflow tracking."""
        print("\n" + "="*60)
        print(f"Training Polynomial Regression Model (Degree {degree})")
        print("="*60)
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"Polynomial_Regression_Degree_{degree}"):
            # Create polynomial pipeline
            model = Pipeline([
                ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # Log parameters
            mlflow.log_param("model_type", "Polynomial Regression")
            mlflow.log_param("polynomial_degree", degree)
            mlflow.log_param("n_features", self.X_train.shape[1])
            mlflow.log_param("train_samples", len(self.X_train))
            mlflow.log_param("test_samples", len(self.X_test))
            
            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log model
            mlflow.sklearn.log_model(model, f"polynomial_model_degree_{degree}")
            
            # Print results
            print(f"\nTraining Metrics:")
            print(f"  RMSE: {train_rmse:.2f}")
            print(f"  MAE: {train_mae:.2f}")
            print(f"  R² Score: {train_r2:.4f}")
            
            print(f"\nTest Metrics:")
            print(f"  RMSE: {test_rmse:.2f}")
            print(f"  MAE: {test_mae:.2f}")
            print(f"  R² Score: {test_r2:.4f}")
            
            # Visualize results
            self._plot_predictions(self.y_test, y_test_pred, f"Polynomial Regression (Degree {degree})")
            plt.savefig(f'models/polynomial_degree_{degree}_predictions.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact(f'models/polynomial_degree_{degree}_predictions.png')
            
        return model, test_r2
    
    def _plot_predictions(self, y_true, y_pred, title):
        """Plot actual vs predicted values."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=80)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Score', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Score', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{title}\nActual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=80)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Score', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{title}\nResidual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def _plot_feature_importance(self, coefficients, feature_names, title):
        """Plot feature importance based on coefficients."""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })
        importance_df = importance_df.sort_values('Coefficient', key=abs, ascending=True)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'green' for x in importance_df['Coefficient']]
        plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'{title}\nFeature Importance', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
    
    def compare_models(self):
        """Compare linear and polynomial models visually."""
        print("\n" + "="*60)
        print("Comparing Model Performance")
        print("="*60)
        
        # Train both models
        linear_model, linear_r2 = self.train_linear_regression()
        poly_model_2, poly_r2_2 = self.train_polynomial_regression(degree=2)
        
        # Create comparison plot
        models_comparison = pd.DataFrame({
            'Model': ['Linear (d=1)', 'Polynomial (d=2)'],
            'R² Score': [linear_r2, poly_r2_2]
        })
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models_comparison['Model'], models_comparison['R² Score'], 
                       color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=2)
        plt.ylabel('R² Score', fontsize=12, fontweight='bold')
        plt.title('Model Comparison - Test Set Performance', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nModel comparison plot saved: models/model_comparison.png")
        
        # Create polynomial fitting visualization
        self._plot_polynomial_fitting(linear_model, poly_model_2)
        
        # Print summary
        print("\n" + "="*60)
        print("Model Performance Summary")
        print("="*60)
        print(models_comparison.to_string(index=False))
        print("\nBest Model:", models_comparison.loc[models_comparison['R² Score'].idxmax(), 'Model'])
    
    def _plot_polynomial_fitting(self, linear_model, poly_model_2):
        """Visualize how different polynomial degrees fit the data."""
        # Use the primary feature (study hours) for visualization
        feature_idx = 0  # Weekly_Study_Hours
        X_primary = self.X_test.iloc[:, feature_idx].values.reshape(-1, 1)
        
        # Sort for smooth line plotting
        sort_idx = np.argsort(X_primary.flatten())
        X_sorted = X_primary[sort_idx]
        # Handle both pandas Series and numpy array
        y_test_array = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        y_sorted = y_test_array[sort_idx]
        
        # Create full feature array for predictions
        X_test_sorted = self.X_test.iloc[sort_idx].reset_index(drop=True)
        
        # Get predictions
        y_pred_linear = linear_model.predict(X_test_sorted)
        y_pred_poly2 = poly_model_2.predict(X_test_sorted)
        
        # Create smooth curves for polynomial fits
        # Generate fine-grained data points for smooth curves
        X_range = np.linspace(X_sorted.min(), X_sorted.max(), 300).reshape(-1, 1)
        
        # Create full feature arrays with mean values for other features
        X_smooth = pd.DataFrame(
            np.tile(self.X_test.mean().values, (300, 1)),
            columns=self.X_test.columns
        )
        X_smooth.iloc[:, feature_idx] = X_range.flatten()
        
        # Get smooth predictions
        y_smooth_linear = linear_model.predict(X_smooth)
        y_smooth_poly2 = poly_model_2.predict(X_smooth)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Linear (Degree 1)
        axes[0].scatter(X_sorted, y_sorted, alpha=0.6, s=50, c='gray', edgecolors='k', label='Actual Data')
        axes[0].plot(X_range, y_smooth_linear, 'b-', linewidth=3, label='Linear Fit (d=1)')
        axes[0].set_xlabel('Study Hours per Week', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Student Score', fontsize=11, fontweight='bold')
        axes[0].set_title('Linear Regression (Degree 1)', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 110])
        
        # Plot 2: Polynomial Degree 2
        axes[1].scatter(X_sorted, y_sorted, alpha=0.6, s=50, c='gray', edgecolors='k', label='Actual Data')
        axes[1].plot(X_range, y_smooth_poly2, '-', linewidth=3, label='Polynomial Fit (d=2)', color='#e74c3c')
        axes[1].set_xlabel('Study Hours per Week', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Student Score', fontsize=11, fontweight='bold')
        axes[1].set_title('Polynomial Regression (Degree 2)', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 110])
        
        plt.tight_layout()
        plt.savefig('models/polynomial_fitting_comparison.png', dpi=300, bbox_inches='tight')
        print("\nPolynomial fitting comparison saved: models/polynomial_fitting_comparison.png")
        
        # Create combined comparison with smooth curves
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(X_sorted, y_sorted, alpha=0.6, s=80, c='black', edgecolors='k', 
                  label='Actual Data', zorder=5)
        ax.plot(X_range, y_smooth_linear, 'b-', linewidth=3, label='Linear (d=1)', alpha=0.9)
        ax.plot(X_range, y_smooth_poly2, '-', linewidth=3, label='Polynomial (d=2)', 
               alpha=0.9, color='#e74c3c')
        
        ax.set_xlabel('Study Hours per Week', fontsize=13, fontweight='bold')
        ax.set_ylabel('Student Score', fontsize=13, fontweight='bold')
        ax.set_title('Comparing Polynomial Fits: Linear vs Polynomial (Degree 2)\nSmooth Curves Show Non-Linear Patterns', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 110])
        
        plt.tight_layout()
        plt.savefig('models/all_fits_combined.png', dpi=300, bbox_inches='tight')
        print("Combined fitting comparison saved: models/all_fits_combined.png")

def main():
    """Main execution function."""
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    print("="*60)
    print("Student Score Prediction System")
    print("="*60)
    
    # Initialize predictor
    data_path ="C:\\Users\\Ayush Reddy\\Desktop\\AI _Engineering\\Project_01_Student_Score_Prediction\\data\\StudentsPerformance.csv"
    predictor = StudentScorePredictor(data_path)
    
    # Load and preprocess data
    predictor.load_and_preprocess_data()
    
    # Compare all models
    predictor.compare_models()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nTo view MLflow UI, run:")
    print("  mlflow ui")
    print("\nThen open: http://localhost:5000")
    print("\nModel artifacts saved in: ./models")
    print("MLflow experiments saved in: ./mlruns")

if __name__ == "__main__":
    main()
