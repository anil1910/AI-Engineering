"""
FastAPI Application for Spam Email Classification
Uses TF-IDF Vectorizer + Logistic Regression model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import uvicorn
from typing import Dict, List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Spam Email Classifier API",
    description="AI-powered spam email detection using NLP and Machine Learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing
model = None
vectorizer = None
stop_words = None
lemmatizer = None

# Pydantic models for request/response
class EmailInput(BaseModel):
    text: str = Field(..., description="Email text to classify", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You've won $1,000,000! Click here to claim your prize now!"
            }
        }

class EmailBatchInput(BaseModel):
    emails: List[str] = Field(..., description="List of email texts to classify")
    
    class Config:
        json_schema_extra = {
            "example": {
                "emails": [
                    "Congratulations! You've won $1,000,000!",
                    "Hi John, meeting at 3 PM tomorrow."
                ]
            }
        }

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Prediction: SPAM or HAM")
    confidence: float = Field(..., description="Confidence score (0-100)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    cleaned_text: str = Field(..., description="Preprocessed email text")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    spam_count: int
    ham_count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool

# Text preprocessing function
def preprocess_text(text: str) -> str:
    """
    Comprehensive text preprocessing function
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stop_words and len(word) > 2]
    
    # Join tokens back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

@app.on_event("startup")
async def load_model():
    """
    Load model and vectorizer on startup
    """
    global model, vectorizer, stop_words, lemmatizer
    
    try:
        # Download NLTK resources if not already present
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        # Load preprocessing components
        preprocessing_components = joblib.load('models/preprocessing_components.pkl')
        stop_words = preprocessing_components['stop_words']
        lemmatizer = preprocessing_components['lemmatizer']
        
        # Load vectorizer
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Load model
        model = joblib.load('models/lr_tfidf_model.pkl')
        
        print("✓ Model and vectorizer loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Spam Email Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_email(email: EmailInput):
    """
    Predict if a single email is spam or ham
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess the email text
        cleaned_text = preprocess_text(email.text)
        
        if not cleaned_text:
            raise HTTPException(
                status_code=400, 
                detail="Email text is empty after preprocessing. Please provide meaningful content."
            )
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Format response
        result = "SPAM" if prediction == 1 else "HAM"
        confidence = float(probabilities[prediction] * 100)
        
        return {
            "prediction": result,
            "confidence": round(confidence, 2),
            "probabilities": {
                "ham": round(float(probabilities[0]), 4),
                "spam": round(float(probabilities[1]), 4)
            },
            "cleaned_text": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: EmailBatchInput):
    """
    Predict multiple emails at once
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch.emails) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 emails per batch")
    
    try:
        predictions = []
        spam_count = 0
        ham_count = 0
        
        for email_text in batch.emails:
            # Preprocess
            cleaned_text = preprocess_text(email_text)
            
            if not cleaned_text:
                predictions.append({
                    "prediction": "UNKNOWN",
                    "confidence": 0.0,
                    "probabilities": {"ham": 0.0, "spam": 0.0},
                    "cleaned_text": "Empty after preprocessing"
                })
                continue
            
            # Vectorize and predict
            text_vectorized = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]
            
            result = "SPAM" if prediction == 1 else "HAM"
            confidence = float(probabilities[prediction] * 100)
            
            if result == "SPAM":
                spam_count += 1
            else:
                ham_count += 1
            
            predictions.append({
                "prediction": result,
                "confidence": round(confidence, 2),
                "probabilities": {
                    "ham": round(float(probabilities[0]), 4),
                    "spam": round(float(probabilities[1]), 4)
                },
                "cleaned_text": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
            })
        
        return {
            "predictions": predictions,
            "total_processed": len(batch.emails),
            "spam_count": spam_count,
            "ham_count": ham_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info", tags=["Model Info"])
async def model_info():
    """
    Get information about the loaded model
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Logistic Regression",
        "vectorizer_type": "TF-IDF",
        "max_features": vectorizer.max_features,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "ngram_range": vectorizer.ngram_range,
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

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
