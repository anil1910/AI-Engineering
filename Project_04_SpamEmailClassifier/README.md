# Spam Email Classifier - FastAPI Application

AI-powered spam email detection using NLP and Machine Learning (TF-IDF + Logistic Regression).

## 🚀 Features

- **High Precision**: Uses TF-IDF vectorization with Logistic Regression for accurate spam detection
- **REST API**: FastAPI-based API with automatic documentation
- **Batch Processing**: Predict multiple emails in a single request
- **Comprehensive Preprocessing**: Standard NLP pipeline including stopwords removal and lemmatization
- **Real-time Predictions**: Fast inference with confidence scores

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🔧 Installation

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Train the Model** (if not already trained)

Open and run the Jupyter notebook to train the model and save the artifacts:

```bash
jupyter notebook spam_classifier.ipynb
```

Make sure to run the notebook cells up to and including the "Save Model and Vectorizer for Deployment" section. This will create a `models/` directory with the trained model files.

## 📦 Model Files

After training, you should have the following files in the `models/` directory:

- `tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `lr_tfidf_model.pkl` - Logistic Regression model
- `preprocessing_components.pkl` - Stopwords and lemmatizer

## 🏃 Running the API

Start the FastAPI server:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:

- **Interactive API Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc

## 🔌 API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

### 2. Single Email Prediction

```bash
POST /predict
```

**Request Body:**
```json
{
  "text": "Congratulations! You've won $1,000,000!"
}
```

**Response:**
```json
{
  "prediction": "SPAM",
  "confidence": 98.45,
  "probabilities": {
    "ham": 0.0155,
    "spam": 0.9845
  },
  "cleaned_text": "congratulations won..."
}
```

### 3. Batch Prediction

```bash
POST /predict/batch
```

**Request Body:**
```json
{
  "emails": [
    "Congratulations! You've won $1,000,000!",
    "Hi John, meeting at 3 PM tomorrow."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "SPAM",
      "confidence": 98.45,
      "probabilities": {
        "ham": 0.0155,
        "spam": 0.9845
      },
      "cleaned_text": "congratulations won..."
    },
    {
      "prediction": "HAM",
      "confidence": 95.23,
      "probabilities": {
        "ham": 0.9523,
        "spam": 0.0477
      },
      "cleaned_text": "john meeting tomorrow..."
    }
  ],
  "total_processed": 2,
  "spam_count": 1,
  "ham_count": 1
}
```

### 4. Model Information

```bash
GET /model/info
```

**Response:**
```json
{
  "model_type": "Logistic Regression",
  "vectorizer_type": "TF-IDF",
  "max_features": 5000,
  "vocabulary_size": 5000,
  "ngram_range": [1, 2],
  "preprocessing": [
    "Lowercase conversion",
    "URL removal",
    "Email address removal",
    "HTML tag removal",
    "Special character removal",
    "Stopwords removal",
    "Lemmatization"
  ]
}
```

## 🧪 Testing the API

Run the test script to verify all endpoints:

```bash
python test_api.py
```

Or use curl:

```bash
# Test single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You won $1,000,000!"}'

# Test health check
curl "http://localhost:8000/health"
```

## 📊 Model Performance

The TF-IDF + Logistic Regression model achieves:

- **High Precision**: Minimizes false positives
- **High Recall**: Catches most spam emails
- **Fast Inference**: Predictions in milliseconds
- **Interpretable**: Clear feature importance

## 🛠️ Project Structure

```
Project 4 - Spam Email Classifier NLP/
├── DATA/
│   └── emails.csv              # Training data
├── models/                      # Saved model artifacts
│   ├── tfidf_vectorizer.pkl
│   ├── lr_tfidf_model.pkl
│   └── preprocessing_components.pkl
├── spam_classifier.ipynb        # Model training notebook
├── app.py                       # FastAPI application
├── test_api.py                  # API test script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔒 Security Considerations

- The API accepts any origin for CORS (development mode)
- For production, configure proper CORS origins
- Consider adding rate limiting
- Add authentication for production use
- Validate and sanitize all inputs

## 🚀 Deployment

For production deployment:

1. **Set proper CORS origins** in `app.py`
2. **Use a production ASGI server** like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

3. **Deploy to cloud platforms**:
   - AWS (EC2, ECS, Lambda)
   - Google Cloud (Cloud Run, GKE)
   - Azure (App Service, AKS)
   - Heroku
   - DigitalOcean

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📧 Contact

For questions or support, please open an issue in the repository.
