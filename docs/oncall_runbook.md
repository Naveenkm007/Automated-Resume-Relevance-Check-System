# On-Call Runbook - Resume Relevance Check System

## Emergency Contacts & Escalation

**Primary On-Call:** DevOps Team  
**Secondary:** Backend Team Lead  
**Escalation:** Engineering Manager  

**Service Level Objectives (SLO):**
- API Availability: 99.9% uptime
- Response Time: < 500ms (95th percentile)
- Background Processing: < 2 minutes per resume

## Quick Incident Response

### 🚨 **IMMEDIATE ACTIONS (First 5 minutes)**

1. **Check service health:**
   ```bash
   curl -f http://localhost:8000/health
   # Expected: {"status": "healthy"}
   ```

2. **Check critical dependencies:**
   ```bash
   # Database
   docker-compose exec postgres pg_isready
   
   # Redis  
   docker-compose exec redis redis-cli ping
   
   # Workers
   docker-compose exec api celery -A api.tasks inspect active
   ```

3. **Check recent logs:**
   ```bash
   docker-compose logs --tail=50 api worker
   ```

4. **Page stakeholders** if any service is down

---

## Service Architecture Overview

```
Users → Load Balancer → API Servers → Database
                     → Redis → Celery Workers
```

**Critical Components:**
- **API Servers**: Handle HTTP requests
- **Celery Workers**: Process resume evaluations  
- **PostgreSQL**: Store all application data
- **Redis**: Message broker + caching
- **File Storage**: Resume/JD uploads

---

## Common Issues & Solutions

### 🔥 **HIGH PRIORITY INCIDENTS**

#### API Server Down/Unresponsive

**Symptoms:**
- Health check returning 5xx errors
- High response times (>5 seconds)
- Users cannot upload resumes/JDs

**Immediate Actions:**
```bash
# 1. Check if containers are running
docker-compose ps

# 2. Restart API service
docker-compose restart api

# 3. Check logs for errors
docker-compose logs api --tail=100

# 4. If database connection issues:
docker-compose restart postgres
sleep 30
docker-compose restart api
```

**Root Cause Investigation:**
- Check memory/CPU usage: `docker stats`
- Check disk space: `df -h`
- Review error logs for patterns
- Check recent deployments

---

#### Database Connection Failures

**Symptoms:**
- `DatabaseError: connection refused`
- API returning 500 errors
- Workers failing to start

**Immediate Actions:**
```bash
# 1. Check PostgreSQL status
docker-compose exec postgres pg_isready -U postgres

# 2. Check connection limits
docker-compose exec postgres psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# 3. Restart database if needed
docker-compose restart postgres

# 4. Wait for health check, then restart dependent services
sleep 30
docker-compose restart api worker
```

**Prevention:**
- Monitor connection pool usage
- Set connection limits appropriately
- Implement connection retry logic

---

#### Worker Queue Backup

**Symptoms:**
- Resumes stuck in "processing" status
- Long evaluation times (>5 minutes)
- Celery queue growing indefinitely

**Immediate Actions:**
```bash
# 1. Check worker status
docker-compose exec api celery -A api.tasks inspect active

# 2. Check queue length
docker-compose exec api celery -A api.tasks inspect reserved

# 3. Scale up workers temporarily
docker-compose up --scale worker=5 -d

# 4. Purge failed tasks if necessary
docker-compose exec api celery -A api.tasks purge
```

**Investigation:**
```bash
# Check for stuck tasks
docker-compose exec api celery -A api.tasks inspect active

# Check worker logs
docker-compose logs worker --tail=100

# Monitor task processing rate
docker-compose exec api celery -A api.tasks events
```

---

### ⚠️ **MEDIUM PRIORITY INCIDENTS**

#### High Memory Usage

**Detection:**
```bash
# Check container memory usage
docker stats

# Check system memory
free -h
```

**Actions:**
```bash
# 1. Identify memory-heavy containers
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# 2. Restart heavy containers
docker-compose restart api worker

# 3. Clear Redis cache if needed
docker-compose exec redis redis-cli FLUSHDB

# 4. Clean up old files
docker-compose exec api python scripts/cleanup_old_files.py
```

---

#### Slow API Responses

**Detection:**
- Response times >2 seconds
- Users reporting slow uploads

**Investigation:**
```bash
# 1. Check active connections
docker-compose exec postgres psql -U postgres -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 2. Check slow queries
docker-compose exec postgres psql -U postgres -c "SELECT query, query_start, now() - query_start AS duration FROM pg_stat_activity WHERE now() - query_start > interval '5 seconds';"

# 3. Check Redis performance
docker-compose exec redis redis-cli info stats
```

**Actions:**
- Scale API servers: `docker-compose up --scale api=3 -d`
- Clear cache: `docker-compose exec redis redis-cli FLUSHALL`
- Restart services: `docker-compose restart api`

---

#### File Upload Issues

**Symptoms:**
- "File too large" errors
- Upload timeouts
- Missing files in storage

**Investigation:**
```bash
# 1. Check disk space
df -h

# 2. Check upload directory permissions
ls -la uploads/

# 3. Check file size limits
docker-compose exec api env | grep MAX_FILE_SIZE
```

**Actions:**
```bash
# 1. Clean old uploads
find uploads/ -type f -mtime +30 -delete

# 2. Fix permissions
chmod -R 755 uploads/

# 3. Increase limits if needed (update .env)
# MAX_FILE_SIZE=20971520  # 20MB
docker-compose restart api
```

---

## Monitoring & Health Checks

### Key Metrics to Monitor

1. **API Health:**
   ```bash
   curl -s http://localhost:8000/health | jq .
   ```

2. **Queue Length:**
   ```bash
   docker-compose exec api celery -A api.tasks inspect reserved | jq 'length'
   ```

3. **Database Connections:**
   ```bash
   docker-compose exec postgres psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
   ```

4. **Disk Usage:**
   ```bash
   du -sh uploads/ logs/
   ```

### Alerting Thresholds

- **API Response Time** > 2 seconds → Warning
- **API Response Time** > 5 seconds → Critical
- **Queue Length** > 100 tasks → Warning  
- **Queue Length** > 500 tasks → Critical
- **Disk Usage** > 80% → Warning
- **Disk Usage** > 90% → Critical
- **Worker Count** < 1 → Critical

---

## Operational Procedures

### 🔄 **Restart Worker Queue**

```bash
# 1. Gracefully stop workers
docker-compose exec worker celery -A api.tasks control shutdown

# 2. Wait for tasks to complete (max 2 minutes)
sleep 120

# 3. Restart worker service
docker-compose restart worker

# 4. Verify workers are active
docker-compose exec api celery -A api.tasks inspect active
```

### 🧹 **Flush Task Queue**

⚠️ **WARNING: This will delete all pending tasks**

```bash
# 1. Stop workers to prevent new task processing
docker-compose stop worker

# 2. Purge all pending tasks
docker-compose exec api celery -A api.tasks purge

# 3. Clear Redis queues
docker-compose exec redis redis-cli FLUSHALL

# 4. Restart workers
docker-compose start worker
```

### 🔁 **Re-run Failed Evaluations**

```bash
# 1. Find failed evaluations in database
docker-compose exec api python -c "
from api.database import SessionLocal
from api.models import Evaluation
db = SessionLocal()
failed = db.query(Evaluation).filter(Evaluation.status == 'failed').all()
print(f'Found {len(failed)} failed evaluations')
for eval in failed[:10]:  # Show first 10
    print(f'Resume: {eval.resume_id}, Error: {eval.error_message}')
db.close()
"

# 2. Re-queue specific evaluation
docker-compose exec api python -c "
from api.tasks import evaluate_resume_task
# Replace with actual IDs
evaluate_resume_task.delay('resume-id-here', 'jd-id-here')
print('Evaluation re-queued')
"

# 3. Bulk re-queue (use with caution)
docker-compose exec api python scripts/requeue_failed_evaluations.py
```

### 📊 **Database Maintenance**

```bash
# 1. Check database size
docker-compose exec postgres psql -U postgres -c "SELECT pg_size_pretty(pg_database_size('resume_checker'));"

# 2. Analyze table sizes
docker-compose exec postgres psql -U postgres -d resume_checker -c "
SELECT schemaname,tablename,attname,n_distinct,correlation
FROM pg_stats WHERE schemaname = 'public' ORDER BY n_distinct DESC;
"

# 3. Vacuum database (run during low usage)
docker-compose exec postgres psql -U postgres -d resume_checker -c "VACUUM ANALYZE;"

# 4. Check slow queries
docker-compose exec postgres psql -U postgres -d resume_checker -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;
"
```

---

## Backup & Recovery

### 🗃️ **Database Backup**

```bash
# 1. Create backup
docker-compose exec postgres pg_dump -U postgres resume_checker > backup_$(date +%Y%m%d_%H%M%S).sql

# 2. Verify backup
head -20 backup_*.sql

# 3. Compress backup
gzip backup_*.sql
```

### 📁 **File Backup**

```bash
# 1. Backup upload files
tar -czf uploads_backup_$(date +%Y%m%d_%H%M%S).tar.gz uploads/

# 2. Backup logs (if needed)
tar -czf logs_backup_$(date +%Y%m%d_%H%M%S).tar.gz logs/
```

### 🔄 **Recovery Process**

```bash
# 1. Stop all services
docker-compose down

# 2. Restore database
docker-compose up -d postgres
sleep 30
gunzip -c backup_20240101_120000.sql.gz | docker-compose exec -T postgres psql -U postgres -d resume_checker

# 3. Restore files
tar -xzf uploads_backup_20240101_120000.tar.gz

# 4. Start all services
docker-compose up -d

# 5. Verify restoration
curl http://localhost:8000/health
```

---

## Performance Tuning

### 🚀 **Scale Services**

```bash
# Scale API servers (for high traffic)
docker-compose up --scale api=5 -d

# Scale workers (for heavy processing)
docker-compose up --scale worker=8 -d

# Check resource usage
docker stats
```

### ⚡ **Cache Optimization**

```bash
# 1. Check Redis memory usage
docker-compose exec redis redis-cli info memory

# 2. Clear specific cache patterns
docker-compose exec redis redis-cli --scan --pattern "cache:*" | xargs docker-compose exec redis redis-cli del

# 3. Configure Redis memory policy
docker-compose exec redis redis-cli config set maxmemory-policy allkeys-lru
```

---

## Security Incidents

### 🔒 **Suspicious Activity**

1. **Check access logs:**
   ```bash
   docker-compose logs api | grep -E "(401|403|429)"
   ```

2. **Review recent uploads:**
   ```bash
   find uploads/ -type f -mmin -60 -ls  # Files uploaded in last hour
   ```

3. **Block suspicious IPs (if using nginx):**
   ```bash
   # Add to nginx config
   deny 192.168.1.100;
   docker-compose restart nginx
   ```

### 🔐 **API Key Compromise**

1. **Rotate compromised keys immediately**
2. **Check usage logs for the compromised key**
3. **Update affected users**
4. **Review and strengthen key management**

---

## Deployment Rollback

### 🔙 **Emergency Rollback**

```bash
# 1. Identify last known good version
docker images | grep resume-checker

# 2. Update docker-compose.yml with previous tag
# image: ghcr.io/your-org/resume-checker:v1.2.0

# 3. Deploy previous version
docker-compose pull
docker-compose up -d

# 4. Verify health
curl http://localhost:8000/health

# 5. Check functionality
python validate_backend.py
```

---

## Contact Information

**On-Call Rotation:**
- Primary: `+1-555-0101` 
- Secondary: `+1-555-0102`
- Manager: `+1-555-0103`

**Escalation Matrix:**
1. **0-15 minutes**: Self-resolve using runbook
2. **15-30 minutes**: Contact secondary on-call
3. **30-60 minutes**: Escalate to manager
4. **>60 minutes**: Engage vendor support if needed

**External Vendors:**
- Cloud Provider Support: `support@cloudprovider.com`
- Database Support: `enterprise@postgresql.com`
- Monitoring: `support@monitoring.com`

---

**Remember:** Always document incidents in the post-mortem template and update this runbook based on learnings!
