# Backend Deployment Guide

## Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd resume-relevance-check

# Copy environment template  
cp .env.example .env

# Edit .env with your configuration
# Required: DATABASE_URL, REDIS_URL, OPENAI_API_KEY (optional)
```

### 2. Docker Deployment (Recommended)
```bash
# Start all services
docker-compose up --build -d

# Initialize database with sample data
docker-compose exec api python -m api.init_db

# View logs
docker-compose logs -f api worker
```

### 3. Access Services
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health  
- **Celery Monitor**: http://localhost:5555
- **Database**: localhost:5432 (postgres/postgres123)

### 4. Run Tests
```bash
docker-compose exec api pytest tests/ -v
```

### 5. Database Migrations
```bash
# Generate new migration
docker-compose exec api alembic revision --autogenerate -m "Description"

# Apply migrations
docker-compose exec api alembic upgrade head
```

## Production Deployment

### Environment Configuration
Update `.env` for production:

```bash
# Production settings
DEBUG=false
ENVIRONMENT=production
SECRET_KEY=secure-random-key-here

# Production database
DATABASE_URL=postgresql://user:password@prod-db:5432/resume_checker

# SSL and security
CORS_ORIGINS=https://yourdomain.com
```

### SSL Setup
1. Place SSL certificates in `nginx/ssl/`
2. Uncomment HTTPS server block in `nginx/nginx.conf`
3. Update domain name in configuration

### Monitoring Setup
```bash
# Start with monitoring profile
docker-compose --profile production up -d

# Access Flower monitoring
open http://localhost:5555
```

## Manual Installation (Development)

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set environment variables
export DATABASE_URL=postgresql://user:pass@localhost:5432/resume_checker
export REDIS_URL=redis://localhost:6379/0

# Initialize database
python -m api.init_db

# Run API server
uvicorn api.main:app --reload

# Run worker (separate terminal)
celery -A api.tasks worker --loglevel=info
```

## Troubleshooting

### Common Issues

**Database Connection Error:**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View database logs
docker-compose logs postgres
```

**Worker Not Processing:**
```bash
# Check Redis connection
docker-compose exec redis redis-cli ping

# View worker logs
docker-compose logs worker
```

**File Upload Errors:**
```bash
# Check upload directory permissions
docker-compose exec api ls -la uploads/
```

### Performance Tuning
```bash
# Scale workers based on CPU cores
docker-compose up --scale worker=4

# Monitor resource usage
docker stats
```

## Production Hardening

- **✅ Logging**: Implement structured logging with log aggregation (ELK stack, Datadog)
- **✅ Monitoring**: Set up application monitoring and alerting (Prometheus, Grafana, New Relic)  
- **✅ Secret Management**: Use proper secret management (AWS Secrets Manager, HashiCorp Vault)
- **✅ Rate Limiting**: Configure comprehensive rate limiting and DDoS protection (Cloudflare, AWS WAF)
