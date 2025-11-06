# 🚀 DEPLOYMENT CHECKLIST

**Status:** Production Ready ✅  
**Version:** 3.0 MVP  
**Last Updated:** November 7, 2025

---

## ✅ PRE-DEPLOYMENT VERIFICATION

### System Requirements
- [ ] Python 3.9+ installed
- [ ] CUDA toolkit installed (optional but recommended)
- [ ] PostgreSQL/Supabase configured
- [ ] Model file present: `models/pig_yolo11-seg.v4.pt` (~5.8 MB)

### Environment Setup
- [ ] `.env` file created with required variables
- [ ] `MODEL_PATH` points to valid YOLO model
- [ ] `DEVICE` configured (cuda:0 or cpu)
- [ ] Database credentials set in `SUPABASE_*` variables

### Dependencies
- [ ] Run: `pip install -r requirements.txt`
- [ ] Verify PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Test imports: `python scripts/test_pipeline_integration.py`

---

## 🧪 TESTING BEFORE DEPLOYMENT

### Unit Tests
```bash
# Pipeline integration tests
python scripts/test_pipeline_integration.py
# Expected: 4/4 tests passed

# API endpoint tests
python scripts/test_api_endpoints.py
# Expected: 5/5 endpoints working

# Complete system test
python scripts/test_complete_system.py
# Expected: All tests passed
```

### Manual Testing
```bash
# 1. Start API server
python main.py

# 2. Test console app
python console_app.py

# 3. Test with sample video
python console_app.py --video samples/test.mp4
```

### Mobile Dashboard
- [ ] Open http://localhost:8000/mobile
- [ ] Verify metrics display
- [ ] Test export button
- [ ] Test verify button

---

## 🎯 DEPLOYMENT STEPS

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Application
```bash
# Create .env file
cp .env.example .env

# Edit .env with your settings
# - Database credentials
# - Model path
# - API host/port
# - Camera URLs (if any)
```

### 3. Initialize Database
```bash
# Run migrations (automatic on first start)
docker-compose up -d  # Or use your DB setup

# Verify tables
python -c "from pig_tracking.database_manager import DatabaseManager; print('DB OK')"
```

### 4. Start Services

#### Option A: Development
```bash
python main.py
# Server runs on http://localhost:8000
```

#### Option B: Production with Gunicorn
```bash
pip install gunicorn uvicorn[standard]
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Option C: Docker
```bash
docker build -t pigweight .
docker run -p 8000:8000 -v $(pwd)/models:/app/models pigweight
```

### 5. Verify Deployment
```bash
# Check health
curl http://localhost:8000/api/health

# Check all standard endpoints
curl http://localhost:8000/api/stats/current
curl http://localhost:8000/api/events/list
curl http://localhost:8000/api/config/parameters

# Access web interface
# http://localhost:8000/
# http://localhost:8000/mobile
```

---

## 📊 MONITORING AFTER DEPLOYMENT

### Health Checks
```bash
# Every minute:
curl http://localhost:8000/api/health

# Log monitoring:
tail -f logs/api.log

# Process monitoring:
ps aux | grep python
```

### Performance Metrics
- API response time: < 500ms
- Video processing: 10+ fps (CPU), 30+ fps (GPU)
- Database connections: < 5
- Memory usage: < 2GB

### Common Issues & Solutions

**Issue: /api/events returns 404**
```bash
# Solution: Restart server
pkill -f "python main.py"
python main.py
```

**Issue: YOLO model not loading**
```bash
# Solution: Check model path in .env
# Verify file: ls -la models/pig_yolo11-seg.*
# Re-download if needed: python scripts/download_model.py
```

**Issue: Database connection error**
```bash
# Solution: Check Supabase credentials
# Test connection: python -c "from pig_tracking.database import Database; d = Database(); print('OK')"
```

---

## 🔄 ROLLBACK PROCEDURE

If deployment fails:

```bash
# 1. Stop current services
pkill -f "python main"

# 2. Check last good commit
git log --oneline -5

# 3. Rollback if needed
git checkout <last_stable_commit>

# 4. Restart
python main.py
```

---

## 📋 POST-DEPLOYMENT CHECKLIST

- [ ] All endpoints responding (200 OK)
- [ ] Mobile dashboard loads
- [ ] Console app works
- [ ] Database connected
- [ ] Logs being written
- [ ] Performance metrics acceptable
- [ ] Backup configured
- [ ] Monitoring configured
- [ ] Team trained
- [ ] Documentation updated

---

## 🎯 SUCCESS CRITERIA

✅ API responds to all 16 standard endpoints  
✅ Pipeline tests pass 4/4  
✅ Mobile dashboard functional  
✅ Video processing working  
✅ Database synchronized  
✅ No errors in logs  

**If all above OK → DEPLOYMENT SUCCESSFUL!** 🎉

---

## 📞 SUPPORT

- **Architecture:** See `.kiro/AGENT_CONTEXT.md`
- **API Spec:** See `.kiro/SYNC_PLAN.md`
- **Troubleshooting:** See `docs_archive/`

---

**Ready to Deploy!** 🚀

