-- SQL запрос для получения полных проходов свиней через весы
-- Группирует пересечения линий по pig_id в рамках одного акта

WITH pig_crossings AS (
  SELECT 
    c.act_id,
    c.pig_id,
    c.stream_id,
    MIN(c.crossed_at) as first_crossing,
    MAX(c.crossed_at) as last_crossing,
    COUNT(*) as crossings_count,
    STRING_AGG(c.direction, ' -> ' ORDER BY c.crossed_at) as path,
    AVG(c.weight_estimate) as avg_weight
  FROM crossings c
  WHERE c.act_id IS NOT NULL
  GROUP BY c.act_id, c.pig_id, c.stream_id
)
SELECT 
  pc.act_id,
  pc.pig_id,
  pc.stream_id,
  pc.first_crossing as entered_at,
  pc.last_crossing as exited_at,
  EXTRACT(EPOCH FROM (pc.last_crossing - pc.first_crossing)) as duration_sec,
  pc.crossings_count,
  pc.path,
  pc.avg_weight,
  wa.started_at as act_started,
  wa.ended_at as act_ended,
  wa.peak_count as act_peak
FROM pig_crossings pc
JOIN weighing_acts wa ON wa.id = pc.act_id
ORDER BY pc.act_id, pc.first_crossing;

-- Пример результата:
-- act_id | pig_id | entered_at          | exited_at           | duration | path           | weight
-- -------|--------|---------------------|---------------------|----------|----------------|-------
-- 1      | 42     | 2025-10-27 10:00:00 | 2025-10-27 10:00:05 | 5.0      | left -> right  | 110.5
-- 1      | 43     | 2025-10-27 10:00:02 | 2025-10-27 10:00:07 | 5.0      | left -> right  | 115.2
