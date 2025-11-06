# 🚀 QUICK START GUIDE

**Status:** Production Ready ✅  
**Version:** 3.0 MVP  
**Setup Time:** ~5 minutes

---

## ⚡ 30-SECOND STARTUP

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
python main.py

# 3. Open browser
# http://localhost:8000/mobile
```

**Done!** System is running.

---

## 📝 WHAT'S INCLUDED

✅ **REST API** - 16 spec-compliant endpoints  
✅ **Pipeline** - Unified video processing  
✅ **Database** - PostgreSQL with Supabase  
✅ **WebSocket** - Real-time updates  
✅ **Mobile UI** - Liquid Glass dashboard  
✅ **Console App** - CLI interface  
✅ **YOLO v11** - Pig detection model

---

## 🎯 3 WAYS TO USE

### Option 1: Web Dashboard (Easiest)
```bash
python main.py
# Visit: http://localhost:8000/mobile
```
- View real-time metrics
- Export data to Excel
- Verify recordings

### Option 2: Console Application (Interactive)
```bash
python console_app.py
# Choose from menu
```
- Process video files
- Monitor progress
- Export results

### Option 3: REST API (Programmatic)
```bash
# Get statistics
curl http://localhost:8000/api/stats/current

# List events
curl http://localhost:8000/api/events/list

# Check config
curl http://localhost:8000/api/config/parameters
```

---

## 📊 ARCHITECTURE OVERVIEW

```
Video Input (File or RTSP)
         ↓
YOLO v11 Detection (GPU accelerated)
         ↓
Pipeline Processing
  - Video Capture
  - Line Analysis
  - Act Detection
         ↓
Database (Supabase/PostgreSQL)
         ↓
API Endpoints + Dashboard
```

---

## 🔧 CONFIGURATION

Create `.env` file:

```env
# Model
MODEL_PATH=models/pig_yolo11-seg.v4.pt
DEVICE=cuda:0          # or cpu
CONF_THRESHOLD=0.30

# API
HOST=0.0.0.0
PORT=8000

# Database
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

# Processing
IMG_SIZE=960
BATCH_SIZE=4
```

---

## ✅ VERIFY INSTALLATION

```bash
# Test imports
python -c "from core import VideoPipeline; print('OK')"

# Run tests
python scripts/test_complete_system.py

# Health check
curl http://localhost:8000/api/health
```

**Expected:** All green ✅

---

## 📚 KEY COMMANDS

```bash
# Start everything
python main.py

# Interactive console
python console_app.py

# Process video file
python console_app.py --video path/to/video.mp4

# Run tests
python scripts/test_complete_system.py

# Daemon mode
python run_daemon.py --start --monitor

# Clean up
python scripts/clean_uploads.py
```

---

## 🌐 ENDPOINTS

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | System health |
| `/api/stats/current` | GET | Current metrics |
| `/api/events/list` | GET | All events |
| `/api/config/parameters` | GET | Configuration |
| `/api/export/excel` | POST | Export to Excel |
| `/api/verify/compare` | POST | Verify with manual |
| `/` | GET | Web dashboard |
| `/mobile` | GET | Mobile dashboard |

---

## 🐛 TROUBLESHOOTING

### API not responding
```bash
# Restart server
pkill -f "python main"
python main.py
```

### Model not loading
```bash
# Check model exists
ls models/pig_yolo11-seg.v4.pt

# Verify PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Database error
```bash
# Check connection
python -c "from pig_tracking.database_manager import DatabaseManager; print('OK')"
```

---

## 📖 DOCUMENTATION

- **Full Details:** `README.md`
- **Architecture:** `.kiro/AGENT_CONTEXT.md`
- **Deployment:** `DEPLOYMENT_CHECKLIST.md`
- **History:** `SESSION_SUMMARY.md`
- **Specs:** `.kiro/SYNC_PLAN.md`

---

## 🎯 NEXT STEPS

1. **Start:** `python main.py`
2. **Monitor:** Open `http://localhost:8000/mobile`
3. **Process:** Upload video or connect camera
4. **Export:** Download results as Excel
5. **Deploy:** Follow `DEPLOYMENT_CHECKLIST.md`

---

## ✨ FEATURES

✅ Real-time pig detection (YOLO v11)  
✅ Multi-stream support  
✅ Web & mobile dashboard  
✅ Database integration  
✅ Excel export/import  
✅ REST API  
✅ WebSocket streaming  
✅ CLI interface  
✅ Daemon mode  
✅ Production ready  

---

## 🚀 READY?

```bash
python main.py
# Then visit http://localhost:8000/mobile
```

**Enjoy!** 🎉

---

**Questions?** See `DEPLOYMENT_CHECKLIST.md` or `.kiro/AGENT_CONTEXT.md`

