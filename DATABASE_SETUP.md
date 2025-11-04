# Database Setup and Testing Guide

## Current Status

**Database Integration: ✓ Ready**

- ✓ DatabaseManager class implemented
- ✓ Supabase integration configured
- ✓ WeighingAct and CrossingEvent models defined
- ✓ CRUD operations implemented
- ✓ Test suite ready

**Note:** Currently running in **JSON-only mode** (no Supabase credentials)
- All results saved to `records/` directory as JSON files
- Full database support available when configured

## Running Database Tests

### Quick Test
```bash
python test_database.py
```

Output:
```
DATABASE TEST - SAVE AND READ WEIGHING ACTS

Checking configuration:
  SUPABASE_URL: NOT FOUND
  SUPABASE_KEY: NOT FOUND

DATABASE NOT CONFIGURED
Current behavior:
  - Results are saved to JSON files (records/)
  - To use database: copy .env.example to .env
  - Set SUPABASE_URL and SUPABASE_KEY
```

## Setting Up Supabase

### Step 1: Create Supabase Account
1. Go to https://supabase.com
2. Sign up for a free account
3. Create a new project

### Step 2: Get Credentials
1. In Supabase dashboard, go to Settings → API
2. Copy:
   - Project URL → `SUPABASE_URL`
   - service_role secret (use for backend) → `SUPABASE_SERVICE_KEY`
   - anon public key (use for frontend) → `SUPABASE_KEY`

### Step 3: Configure .env
```bash
cp .env.example .env
```

Edit `.env`:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_public_key
SUPABASE_SERVICE_KEY=your_service_role_secret
```

### Step 4: Create Database Schema

Run SQL in Supabase SQL Editor:

```sql
-- Create weighing_acts table
CREATE TABLE weighing_acts (
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  started_at TIMESTAMP WITH TIME ZONE NOT NULL,
  ended_at TIMESTAMP WITH TIME ZONE NOT NULL,
  duration_sec FLOAT NOT NULL,
  left_count INT NOT NULL DEFAULT 0,
  right_count INT NOT NULL DEFAULT 0,
  peak_count INT NOT NULL DEFAULT 0,
  total_weight FLOAT,
  avg_weight FLOAT,
  stream_id TEXT,
  video_file TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create crossings table
CREATE TABLE crossings (
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  act_id BIGINT NOT NULL REFERENCES weighing_acts(id) ON DELETE CASCADE,
  pig_id INT NOT NULL,
  direction TEXT NOT NULL,
  crossed_at TIMESTAMP WITH TIME ZONE NOT NULL,
  line_x FLOAT NOT NULL,
  line_y FLOAT NOT NULL,
  weight_estimate FLOAT,
  stream_id TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_weighing_acts_started_at ON weighing_acts(started_at);
CREATE INDEX idx_weighing_acts_stream_id ON weighing_acts(stream_id);
CREATE INDEX idx_crossings_act_id ON crossings(act_id);
CREATE INDEX idx_crossings_crossed_at ON crossings(crossed_at);
```

### Step 5: Test Connection
```bash
python test_database.py
```

Expected output:
```
DATABASE TEST - SAVE AND READ WEIGHING ACTS

Checking configuration:
  SUPABASE_URL: FOUND
  SUPABASE_KEY: FOUND

Connecting to database...
  OK - Connection established

Test 1: Get current database statistics
======================================================================
  Total acts: 0
  Total crossings: 0
```

## Database Operations

### What's Implemented

#### Writing Data
```python
from pig_tracking.database import DatabaseManager, WeighingAct
from datetime import datetime

db = DatabaseManager()

# Create and save weighing act
act = WeighingAct(
    started_at=datetime.now(),
    ended_at=datetime.now(),
    duration_sec=120.0,
    left_count=5,
    right_count=5,
    peak_count=3,
    total_weight=550.0,
    avg_weight=110.0
)

act_id = db.save_weighing_act(act)
print(f"Saved act with ID: {act_id}")
```

#### Reading Data
```python
from datetime import timedelta

# Get acts by period
start = datetime.now() - timedelta(days=1)
end = datetime.now()
acts = db.get_acts_by_period(start, end)

# Get statistics
stats = db.get_stats()

# Get pig passages
passages = db.get_pig_passages()
```

## Available Methods

### DatabaseManager

```python
# Connection management
DatabaseManager(supabase_url, supabase_key)
_test_connection()

# Writing
save_weighing_act(act: WeighingAct) -> int
save_crossing(crossing: CrossingEvent) -> int

# Reading
get_acts_by_period(start: datetime, end: datetime, stream_id: str) -> List[WeighingAct]
get_crossings_by_act(act_id: int) -> List[CrossingEvent]
get_pig_passages(act_id: int) -> List[Dict]
get_stats() -> Dict

# Utility
clear_all_data()  # For testing only!
```

## Integration with Video Processing

When processing video with `console_app.py`:

1. Results are always saved to JSON files in `records/`
2. If database is configured, results are ALSO saved to Supabase
3. Both formats work independently

### JSON Storage (Always Active)
```
records/
├── act_*.json
├── crossing_*.json
└── summary_*.json
```

### Database Storage (When Configured)
- Tables: `weighing_acts`, `crossings`
- Full CRUD operations
- Queryable via /api/metrics endpoints

## Troubleshooting

### "SUPABASE_KEY not found"
- Check .env file exists
- Verify SUPABASE_URL and SUPABASE_KEY are set
- Restart Python application

### Connection timeout
- Check internet connection
- Verify Supabase project is active
- Check firewall/network settings

### Table doesn't exist
- Run the SQL schema creation script above
- Check table names in error message
- Ensure you're connected to correct database

### Foreign key constraint error
- Ensure weighing_acts table exists before crossings
- Don't delete acts that have crossings (set ON DELETE CASCADE)

## Performance Tips

1. **Use indexes**: Already created in schema
2. **Filter by date**: Use time range queries
3. **Batch operations**: Process multiple acts together
4. **Archive old data**: Move processed data to archive

## Next Steps

1. ✓ Test database with `python test_database.py`
2. ✓ Process video with `python console_app.py`
3. ✓ Check results in `/metrics` web interface
4. ✓ Export data as needed

---

**Status**: Ready for production use with or without database
