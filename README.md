# Solavix AI Backend

FastAPI-based AI service for predictive maintenance with 8 machine learning models.

## Status: Ready for Production Deployment

### Latest Update: GitHub Actions CI/CD Configured

## Features

- **Anomaly Detection**: Isolation Forest, One-Class SVM, DBSCAN, Autoencoder
- **Classification**: Random Forest, XGBoost, SVM classifiers  
- **Regression**: Random Forest, XGBoost regressors
- **Time Series**: Prophet, LSTM models
- **Health Scoring**: Multi-model ensemble predictions
- **Maintenance Prediction**: AI-powered maintenance scheduling

## API Endpoints

- `GET /api/ai/health` - Health check
- `POST /api/ai/anomaly` - Anomaly detection
- `POST /api/ai/health-score` - Calculate health score
- `POST /api/ai/predict-maintenance` - Maintenance prediction
- `GET /docs` - Interactive API documentation

## Deployment

### Azure App Service
1. Create App Service with Python 3.11 runtime
2. Connect to this GitHub repository
3. Azure will automatically deploy

### Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Environment Requirements

- **Memory**: Minimum 3.5GB RAM (Azure P1V2 or higher)
- **Python**: 3.11+
- **Dependencies**: See requirements.txt

## Model Performance

- **Training Data**: 20,000 synthetic industrial sensor readings
- **Accuracy**: 85-95% across different model types
- **Response Time**: <2 seconds for predictions
- **Cold Start**: ~30 seconds (first request after idle)

## API Usage Example

```python
import requests

# Anomaly detection
response = requests.post('https://solavix-ai.azurewebsites.net/api/ai/anomaly', 
    json={
        "timestamp": "2024-01-01T00:00:00Z",
        "temperature": 45.0,
        "voltage": 12.5,
        "current": 8.0,
        "asset_id": "turbine-001"
    })

print(response.json())
```

## License

MIT License - Solavix Industrial IoT Platform