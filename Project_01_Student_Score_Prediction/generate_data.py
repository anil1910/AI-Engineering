"""
Generate synthetic student score data for regression modeling.
This script creates realistic fake data with multiple features affecting student scores.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_student_data(n_samples=500):
    """
    Generate synthetic student score data with realistic features.
    
    Features:
    - study_hours: Hours studied per week (0-40)
    - attendance: Attendance percentage (50-100)
    - previous_score: Previous exam score (40-100)
    - sleep_hours: Average sleep hours per day (4-10)
    - tutoring_sessions: Number of tutoring sessions (0-10)
    - family_income: Family income level (1-5 scale)
    - internet_access: Has internet at home (0 or 1)
    - extracurricular: Hours spent on extracurricular (0-15)
    """
    
    # Generate features with realistic distributions
    study_hours = np.random.gamma(shape=3, scale=3, size=n_samples)
    study_hours = np.clip(study_hours, 0, 40)
    
    attendance = np.random.beta(a=8, b=2, size=n_samples) * 50 + 50
    attendance = np.clip(attendance, 50, 100)
    
    previous_score = np.random.normal(loc=70, scale=15, size=n_samples)
    previous_score = np.clip(previous_score, 40, 100)
    
    sleep_hours = np.random.normal(loc=7, scale=1.5, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    
    tutoring_sessions = np.random.poisson(lam=3, size=n_samples)
    tutoring_sessions = np.clip(tutoring_sessions, 0, 10)
    
    family_income = np.random.choice([1, 2, 3, 4, 5], size=n_samples, 
                                     p=[0.15, 0.25, 0.30, 0.20, 0.10])
    
    internet_access = np.random.choice([0, 1], size=n_samples, p=[0.15, 0.85])
    
    extracurricular = np.random.exponential(scale=3, size=n_samples)
    extracurricular = np.clip(extracurricular, 0, 15)
    
    # Generate target variable (final_score) with realistic relationships
    # Base score influenced by multiple factors
    final_score = (
        40 +  # Base score
        study_hours * 0.8 +  # Study hours have strong positive effect
        (attendance - 50) * 0.3 +  # Attendance matters
        previous_score * 0.25 +  # Previous performance indicator
        sleep_hours * 1.5 +  # Sleep is important
        tutoring_sessions * 1.2 +  # Tutoring helps
        family_income * 1.5 +  # Socioeconomic factor
        internet_access * 3 +  # Access to resources
        extracurricular * 0.3 +  # Balanced activities
        np.random.normal(0, 8, n_samples)  # Random noise
    )
    
    # Add some non-linear effects (for polynomial regression to capture)
    # Too much or too little sleep has negative effects
    sleep_penalty = -0.5 * (sleep_hours - 7) ** 2
    final_score += sleep_penalty
    
    # Diminishing returns on study hours
    study_bonus = -0.02 * (study_hours ** 2) + 0.8 * study_hours
    final_score = final_score - (study_hours * 0.8) + study_bonus
    
    # Clip final score to realistic range
    final_score = np.clip(final_score, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'study_hours': study_hours.round(2),
        'attendance': attendance.round(2),
        'previous_score': previous_score.round(2),
        'sleep_hours': sleep_hours.round(2),
        'tutoring_sessions': tutoring_sessions,
        'family_income': family_income,
        'internet_access': internet_access,
        'extracurricular': extracurricular.round(2),
        'final_score': final_score.round(2)
    })
    
    return df

def main():
    """Generate and save student data."""
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate data
    print("Generating student score data...")
    df = generate_student_data(n_samples=500)
    
    # Split into train and test sets (80-20 split)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # Save to CSV files
    train_path = data_dir / 'student_scores_train.csv'
    test_path = data_dir / 'student_scores_test.csv'
    full_path = data_dir / 'student_scores_full.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)
    
    print(f"\nData generated successfully!")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Total samples: {len(df)}")
    print(f"\nFiles saved:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {full_path}")
    
    # Display basic statistics
    print("\n" + "="*60)
    print("Data Statistics:")
    print("="*60)
    print(df.describe().round(2))
    
    print("\n" + "="*60)
    print("Feature Correlations with Final Score:")
    print("="*60)
    correlations = df.corr()['final_score'].sort_values(ascending=False)
    print(correlations)

if __name__ == "__main__":
    main()
