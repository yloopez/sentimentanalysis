# Sentiment Analysis API - Production ML System

> End-to-end machine learning system for real-time sentiment analysis of product reviews

**Project Status**: Data preprocessing complete, ready for model training

## Overview

This project demonstrates a complete ML system pipeline from data exploration 
to production deployment. Currently trained on 194K+ cellphone and accessory reviews.

- Machine learning model development
- REST API with FastAPI
- Containerization with Docker
- Cloud deployment on AWS
- CI/CD pipeline
- Monitoring and logging

## Current Progress

- [x] Project setup
- [x] Data acquisition (194,439 reviews from Kaggle)
- [x] Initial data exploration
- [x] Data cleaning & preprocessing
- [x] Sentiment label creation
- [x] Class balancing
- [x] Feature engineering
- [ ] Model training (NEXT)
- [ ] Model evaluation
- [ ] API development
- [ ] Deployment

## Dataset

Using Cell_Phones_and_Accessories_5 with 194,439 reviews for sentiment analysis.


## Data Processing Pipeline

1. Remove missing/empty reviews
2. Filter by review length (3-500 words)
3. Remove duplicates
4. Create sentiment labels from ratings
5. Balance classes for training
6. Text preprocessing (cleaning, normalization)

## Tech Stack

**ML & Data Science:**
- Python 3.13.5
- pandas, numpy, scikit-learn
- NLTK for text processing
- matplotlib, seaborn for visualization

**Backend (Coming Soon):**
- FastAPI
- Redis
- PostgreSQL

**DevOps (Coming Soon):**
- Docker
- AWS ECS
- GitHub Actions

## Setup
```bash
# Clone repository
git clone https://github.com/yloopez/sentimentanalysis
cd sentimentanalysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## Project Structure
```
sentiment-analysis-production/
├── data/              # Raw and processed data
├── notebooks/         # python scripts for exploration
├── src/              # Source code (coming soon)
├── tests/            # Tests (coming soon)
├── models/           # Trained models
└── docs/             # Documentation
```

## Author

Yeison Bernal