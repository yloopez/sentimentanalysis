# Sentiment Analysis API - Production ML System

> End-to-end machine learning system for real-time sentiment analysis

**Project Status**: In Development

## Overview

This project demonstrates a complete ML system pipeline from data exploration 
to production deployment, including:

- Machine learning model development
- REST API with FastAPI
- Containerization with Docker
- Cloud deployment on AWS
- CI/CD pipeline
- Monitoring and logging

## Current Progress

- [x] Project setup
- [x] Data acquisition
- [x] Initial data exploration
- [ ] Data preprocessing
- [ ] Model training
- [ ] API development
- [ ] Deployment

## Dataset

Using [Dataset Name] with 194,439 reviews for sentiment analysis.

## Tech Stack

**ML & Data Science:**
- Python 3.13.5
- pandas, numpy, scikit-learn
- NLTK for text processing

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