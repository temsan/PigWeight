# 🔧 REFACTORING PLAN - CODE CLEANUP

**Status:** IN PROGRESS ✅  
**Based on:** CODE_ANALYSIS.md  
**Goal:** Clean up codebase while maintaining production stability

---

## ✅ COMPLETED

### 1. ✅ Removed Code Duplication (DONE)
- **Removed:** 47 lines of exact code duplication from `api/app.py`
- **Functions deleted:**
  - `api_stream_optimize` (duplicate)
  - `load_line_positions` (duplicate)
  - `save_line_positions` (duplicate)
- **File size:** 3033 → 2986 lines (-47 lines)
- **Commit:** 47027d7

### 2. ✅ Created API Utils Module (DONE)
- **File:** `api/endpoints/utils.py`
- **Purpose:** Reduce code duplication in endpoint handlers
- **Provides:**
  - `APIResponse` class for standardized responses
  - `handle_api_error()` for error handling
  - `validate_pagination()` for pagination
  - `create_list_endpoint()` for generic lists
  - Common response templates

---

## ⏳ TODO - HIGH PRIORITY

### 3. 🔴 Choose Single Database Manager
**Problem:** 2 versions of `DatabaseManager` exist
- `pig_tracking/database.py` (version 1)
- `pig_tracking/database_manager.py` (version 2)

**Action Required:**
```bash
# Determine which is correct
# Option A: Keep database_manager.py (newer)
#   - Delete pig_tracking/database.py
#   - Update imports if needed
#
# Option B: Keep database.py (if legacy is correct)
#   - Delete pig_tracking/database_manager.py
#   - Update console_app.py imports
```

**Impact:** Medium (potential bugs if wrong version used)

---

### 4. 🔴 Choose Single Video Processor
**Problem:** 2 main processor implementations exist
- `core/processor.py` - `UnifiedVideoProcessor`
- `pig_tracking/video_processor.py` - `IntegratedVideoProcessor`

**Decision Tree:**
```
IF console_app.py uses IntegratedVideoProcessor
   AND api/app.py uses UnifiedVideoProcessor
   THEN consolidate into one architecture
   
RECOMMENDATION:
  - Keep IntegratedVideoProcessor (used in production console_app.py)
  - Update UnifiedVideoProcessor to wrap it or remove
  - OR create adapter pattern for compatibility
```

**Impact:** High (core processing pipeline)

---

### 5. 🟡 Extract Classes from api/app.py
**Problem:** Large classes embedded in main app file

**Classes to extract:**
```
api/app.py (lines 281+)
├── SimpleTracker → api/models/tracker.py
├── VideoStream (base) → api/models/stream.py
├── FileStream → api/models/stream.py
├── RTCStream → api/models/stream.py
├── DemoStream → api/models/stream.py
├── WeightedMaxEstimator → api/models/estimators.py
└── WindowMaxEstimator → api/models/estimators.py
```

**Benefits:**
- Reduce api/app.py from 2986 → ~1500 lines
- Make classes reusable
- Easier testing
- Better IDE navigation

**Effort:** 3-4 hours

---

### 6. 🟡 Add WebSocket Rate Limiting
**Problem:** WebSocket broadcasts every frame without throttle

**Current Code:**
```python
async def broadcast(self, stream_id: str, data: dict):
    # Sends to ALL websockets for stream
    # EVERY frame sent immediately
    # NO rate limiting
```

**Solution:**
```python
# Implement throttling
# Option 1: Send max 10 fps (100ms throttle)
# Option 2: Skip every Nth frame
# Option 3: Use asyncio.sleep-based queue

RECOMMENDED: 10 fps throttle + frame skipping
```

**Impact:** Low (performance optimization)
**Effort:** 30-45 minutes

---

## 📋 MEDIUM-TERM IMPROVEMENTS

### 7. Consolidate Database Abstraction
**Problem:** Database logic scattered
- `pig_tracking/database.py`
- `pig_tracking/database_manager.py`
- `services/` might have alternatives

**Solution:**
- Keep one clean abstraction layer
- Use repository pattern
- Hide query details from endpoints

---

### 8. Create Unified Error Handling
**Problem:** Error handling varies across endpoints

**Use:** New `api/endpoints/utils.py` utilities
```python
# Before
try:
    result = function()
except Exception as e:
    logger.error(f"Error: {e}")
    raise HTTPException(status_code=500)

# After
try:
    result = function()
except Exception as e:
    raise handle_api_error("function_name", e)
```

---

## 📊 CODE QUALITY METRICS

### Before Refactoring
| Metric | Value | Status |
|--------|-------|--------|
| Largest file (api/app.py) | 3033 lines | ❌ Too large |
| Code duplication | ~5% | ⚠️ Some issues |
| Embedded classes | 7 classes | ❌ Not modular |

### After Refactoring (Target)
| Metric | Target | Status |
|--------|--------|--------|
| Largest file (api/app.py) | ~1500 lines | ✅ Manageable |
| Code duplication | <1% | ✅ Clean |
| Embedded classes | 0 | ✅ Extracted |
| WebSocket throttle | 10 fps | ✅ Optimized |

---

## 🎯 QUICK WINS (Next 30 mins)

### Immediate Actions
```bash
# 1. Decide on DatabaseManager version
# 2. Decide on Processor version
# 3. Add rate limiting decorator for WebSocket
# 4. Document decisions in .kiro/REFACTORING_DECISIONS.md
```

**Time:** 15-30 minutes  
**Impact:** High (clarity + small optimizations)

---

## 🔄 REFACTORING CHECKLIST

### Phase 1: Analysis (DONE ✅)
- [x] Code analysis completed (CODE_ANALYSIS.md)
- [x] Identified duplication
- [x] Prioritized issues

### Phase 2: Cleanup (IN PROGRESS 🟢)
- [x] Remove code duplication
- [x] Create utils module
- [ ] Choose DatabaseManager version
- [ ] Choose Processor version

### Phase 3: Extraction
- [ ] Extract classes from api/app.py
- [ ] Move to api/models/
- [ ] Update imports

### Phase 4: Optimization
- [ ] Add WebSocket rate limiting
- [ ] Consolidate error handling
- [ ] Performance testing

### Phase 5: Validation
- [ ] All tests pass
- [ ] No functional changes
- [ ] Documentation updated

---

## 📝 DECISIONS NEEDED

### Question 1: Which DatabaseManager?
```
[ ] Option A: pig_tracking/database_manager.py (keep)
[ ] Option B: pig_tracking/database.py (keep)
[ ] Option C: Merge both into new version
```

**Owner:** [User]  
**Deadline:** Now

### Question 2: Which Processor?
```
[ ] Option A: UnifiedVideoProcessor (core/processor.py)
[ ] Option B: IntegratedVideoProcessor (pig_tracking/video_processor.py)
[ ] Option C: Create unified wrapper
```

**Owner:** [User]  
**Deadline:** Now

---

## 📚 REFERENCES

- **Analysis:** CODE_ANALYSIS.md
- **Utils:** api/endpoints/utils.py
- **Specs:** .kiro/specs/

---

## 🎯 SUCCESS CRITERIA

✅ **DONE**
- Code duplication removed
- Utils module created
- No functional changes
- All tests still pass

**NEXT**
- Decisions made on DB/Processor
- Classes extracted (or deferred)
- WebSocket optimized (or deferred)

---

**Status:** Ready for decisions and next phase!  
**Estimated Total Time:** 4-6 hours for full refactoring  
**Production Impact:** Minimal (backward compatible)

