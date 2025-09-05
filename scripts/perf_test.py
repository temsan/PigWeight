import time
import requests

# Simple perf test: start a stream (file) and poll /api/stream/{id}/info for latency/fps

BASE = 'http://localhost:8000'

def start_stream(stream_id, source_uri):
    r = requests.post(f'{BASE}/api/stream/start', params={'stream_id': stream_id, 'source_uri': source_uri})
    return r.ok

def stop_stream(stream_id):
    r = requests.get(f'{BASE}/api/stream/{stream_id}/stop')
    return r.ok

def poll_info(stream_id, n=10, delay=0.5):
    times = []
    for i in range(n):
        t0 = time.time()
        r = requests.get(f'{BASE}/api/stream/{stream_id}/info')
        dt = time.time() - t0
        times.append(dt)
        time.sleep(delay)
    return times

if __name__ == '__main__':
    sid = 'perf_test_' + str(int(time.time()))
    src = 'uploads/2.mp4'  # adjust
    print('Starting stream...')
    ok = start_stream(sid, src)
    if not ok:
        print('Start failed')
        raise SystemExit(1)
    print('Polling info...')
    latencies = poll_info(sid, n=20, delay=0.2)
    print('Latencies (s):', latencies)
    print('Avg latency:', sum(latencies)/len(latencies))
    stop_stream(sid)
    print('Stopped')


