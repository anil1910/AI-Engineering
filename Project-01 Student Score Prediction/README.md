# Student Score Prediction System

A comprehensive machine learning project demonstrating **regression modeling** with polynomial features, **MLflow experiment tracking**, and **DVC data versioning**.

## 📊 Project Overview

This project predicts student scores based on various academic and behavioral features using:
- **Exploratory Data Analysis (EDA)** - Understanding data distributions and correlations
- **Linear Regression (Degree 1)** - Simple linear fit baseline
- **Polynomial Regression (Degree 2)** - Captures quadratic non-linear relationships
- **Visual Analysis** - Smooth curve visualizations and model comparisons
- **MLflow Tracking** - Experiment tracking with metrics, parameters, and artifacts
- **DVC Data Versioning** - Version control for datasets

## 🎯 Target Variable

The project uses a **synthetic target variable** (`Target_Score`) with **quadratic characteristics**:

### Target Variable Design:
```
Target_Score = Base (30) 
             + Linear Component (max 48 points)
             + Quadratic Component (max 50 points)
             + Noise (±2.5 points)
```

**Linear Component:**
- Study Hours: 0-15 points
- Attendance: 0-10 points
- Reading: 0-8 points
- Listening in Class: 0-8 points
- Project Work: 0-7 points

**Quadratic Component** (Key for Degree 2!):
- Study Hours²: 0-20 points
- Attendance²: 0-12 points
- Study × Attendance: 0-10 points
- Reading × Listening: 0-8 points

### Why This Design?
✅ **Linear regression** captures basic trends → decent performance  
✅ **Polynomial (degree 2)** captures squared terms & interactions → **significantly better performance**

**Statistics:**
- Range: 36.0 - 100.0
- Mean: 61.51
- Std Dev: 16.34
- Study Hours Correlation: +0.85

## 📁 Project Structure

```
Project 1 – Student Score Prediction System/
├── data/
│   └── Students Performance .csv       # Dataset with Target_Score column
├── models/                             # Saved plots and visualizations
│   ├── linear_regression_predictions.png
│   ├── polynomial_degree_2_predictions.png
│   ├── model_comparison.png
│   ├── polynomial_fitting_comparison.png
│   └── all_fits_combined.png
├── mlruns/                             # MLflow experiment tracking data
├── notebooks/
│   └── Student_Score_Prediction_Regression.ipynb  # Jupyter notebook tutorial
├── venv/                               # Virtual environment
├── train_model.py                      # Main training script
├── create_target_variable.py           # Script to generate Target_Score
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Dependencies:**
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- mlflow
- dvc

### 3. Generate Target Variable (Optional - Already Done)

The CSV already includes `Target_Score`. If you need to regenerate:

```powershell
python create_target_variable.py
```

### 4. Train Models

```powershell
python train_model.py
```

This will:
- ✅ Perform **Exploratory Data Analysis (EDA)**
- ✅ Load and preprocess student performance data
- ✅ Train **Linear Regression** model (degree 1)
- ✅ Train **Polynomial Regression** model (degree 2)
- ✅ Generate visualization plots
- ✅ Track experiments with **MLflow**
- ✅ Save models and artifacts to `./models`

### 5. View MLflow UI

```powershell
mlflow ui
```

Then open: **http://localhost:5000**

## 📈 MLflow Experiment Tracking

MLflow automatically tracks all experiments in the `mlruns/` directory.

### What MLflow Tracks:

✅ **Parameters:**
- Model type (Linear/Polynomial)
- Polynomial degree
- Number of features (5)
- Train/test sample sizes

✅ **Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score (coefficient of determination)
- Training and test set performance

✅ **Artifacts:**
- Trained model files
- Prediction plots (actual vs predicted)
- Residual plots
- Feature importance charts
- Polynomial fitting visualizations

### View Results:
1. Run `mlflow ui`
2. Open http://localhost:5000
3. Compare experiments
4. View metrics, plots, and model parameters

## 🔄 DVC Data Versioning (Optional)

DVC enables version control for datasets.

### Initialize DVC:

```powershell
# Install DVC (already in requirements.txt)
pip install dvc

# Initialize DVC
dvc init

# Add data to DVC tracking
dvc add "data/Students Performance .csv"

# Track DVC metadata in Git
git add "data/Students Performance .csv.dvc" data/.gitignore
git commit -m "Add dataset to DVC tracking"
```

### DVC Workflow:

```powershell
# After modifying data
dvc add "data/Students Performance .csv"

# Commit changes
git add "data/Students Performance .csv.dvc"
git commit -m "Updated dataset with new target variable"

# Optional: Push to remote storage
dvc remote add -d storage s3://mybucket/dvcstore
dvc push

# Team members can pull exact version
dvc pull
```

## 📊 Model Comparison

The project trains and compares two regression models:

| Model | Description | Captures Non-linearity | Expected Performance |
|-------|-------------|------------------------|---------------------|
| **Linear (d=1)** | Simple linear fit | ❌ No | Decent baseline |
| **Polynomial (d=2)** | Quadratic relationships | ✅ Yes | **Significantly Better** |

### Why Polynomial Wins:
The target variable includes:
- Squared terms (study hours², attendance²)
- Interaction terms (study × attendance)
- These are captured perfectly by polynomial degree 2

## 📚 Features Used

The model uses **5 key features**:

| Feature | Description | Correlation |
|---------|-------------|-------------|
| `Weekly_Study_Hours` | Study hours per week (0-12) | +0.85 (strongest) |
| `Attendance` | Class attendance pattern | +0.37 |
| `Project_work` | Completes project work | +0.22 |
| `Reading` | Reading habits | +0.22 |
| `Listening_in_Class` | Listening in class | +0.11 |

## 🎯 Performance Metrics

Models are evaluated using:

- **R² Score** - Proportion of variance explained (0-1, **higher is better**)
  - 1.0 = perfect predictions
  - 0.0 = no better than mean baseline
  
- **RMSE** - Root Mean Squared Error (**lower is better**)
  - Average prediction error magnitude
  - Same units as target (score points)
  
- **MAE** - Mean Absolute Error (**lower is better**)
  - Average absolute prediction error
  - More robust to outliers than RMSE

## 📊 Visualizations Generated

The project automatically creates:

1. **EDA Visualizations:**
   - Grade distribution bar chart
   - Study hours distribution
   - Score histogram
   - Correlation heatmap

2. **Model Performance:**
   - Actual vs Predicted scatter plots
   - Residual plots (error analysis)
   - Feature importance (coefficient values)
   - Model comparison bar chart (R² scores)

3. **Polynomial Fitting:**
   - Side-by-side comparison (Linear vs Polynomial)
   - Combined overlay with smooth curves
   - Shows how degree 2 captures curved patterns

## 🔍 Key Insights

After training, you'll observe:

✅ **Linear Regression (Degree 1):**
- Simple straight-line fit
- Misses non-linear patterns
- Decent R² (~0.70-0.80 range)
- Underfits the data

✅ **Polynomial Regression (Degree 2):**
- Captures quadratic curved patterns
- Much better fit for this target
- High R² (~0.90-0.95 range)
- **Significantly outperforms linear**

## 📓 Jupyter Notebook

Interactive tutorial available: `notebooks/Student_Score_Prediction_Regression.ipynb`

**Includes:**
- Step-by-step explanations
- EDA with visualizations
- Model training code
- Performance comparison
- Residual analysis
- Educational summary

Run with:
```powershell
jupyter notebook
# or
jupyter lab
```

## 🛠️ Technical Details

### Data Preprocessing:
- Categorical encoding (LabelEncoder)
- Age range conversion to numerical
- Missing value handling (Scholarship column)
- Feature selection (5 most relevant features)
- Train/test split (80/20)

### Model Architecture:
- **Linear Regression:** `sklearn.linear_model.LinearRegression`
- **Polynomial Pipeline:**
  - PolynomialFeatures(degree=2)
  - StandardScaler() 
  - LinearRegression()

### Visualization Techniques:
- Smooth curve generation (300 interpolation points)
- Feature importance from coefficients
- Residual analysis for model validation
- Color-coded comparison charts

## 🤝 Contributing

To extend this project:

1. **Add More Models:**
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)
   - ElasticNet (combined regularization)
   - Random Forest Regressor
   - Gradient Boosting

2. **Feature Engineering:**
   - Create additional interaction terms
   - Try different polynomial degrees (3, 4)
   - Add feature scaling variations

3. **Hyperparameter Tuning:**
   - Grid Search CV
   - Random Search CV
   - Bayesian Optimization

4. **Deployment:**
   - Build REST API with Flask/FastAPI
   - Create Streamlit dashboard
   - Deploy to cloud (AWS/Azure/GCP)

## 📝 How to Use This Project

### For Learning:
1. Read through `train_model.py` to understand the workflow
2. Follow the Jupyter notebook step-by-step
3. Experiment with different polynomial degrees
4. Modify the target variable formula in `create_target_variable.py`
5. Compare results in MLflow UI

### For Demonstration:
1. Run `python train_model.py`
2. Launch `mlflow ui`
3. Show the visualizations in `./models`
4. Explain how polynomial regression captures non-linearity
5. Compare R² scores between models

## 🔧 Troubleshooting

**Issue: Import errors**
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Issue: MLflow UI not starting**
```powershell
# Check if port 5000 is available
mlflow ui --port 5001
```

**Issue: Target_Score column not found**
```powershell
# Regenerate target variable
python create_target_variable.py
```

## 📄 License

This is an educational project for demonstrating ML concepts.

## 👨‍💻 Author

Created as part of AI Engineering Projects portfolio.

---

**Happy Learning! 🎓**

For questions or improvements, feel free to open an issue or submit a pull request.
