#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
from collections import defaultdict

# Убрать проблемы с кодировкой на Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('logs/localhost.har', 'r', encoding='utf-8') as f:
    har = json.load(f)

# Статистика
stats = defaultdict(int)
errors = []
slow_requests = []
urls_by_type = defaultdict(list)

for entry in har['log']['entries']:
    req = entry.get('request', {})
    resp = entry.get('response', {})
    url = req.get('url', 'Unknown')
    status = resp.get('status', 0)
    time_ms = entry.get('time', 0)
    
    stats[f'Status {status}'] += 1
    
    # Группировать по типу
    if '/api/' in url:
        urls_by_type['API'].append(url)
    elif '/ws' in url:
        urls_by_type['WebSocket'].append(url)
    elif url.endswith('.js'):
        urls_by_type['JavaScript'].append(url)
    elif url.endswith('.css'):
        urls_by_type['CSS'].append(url)
    else:
        urls_by_type['Other'].append(url)
    
    if status >= 400:
        errors.append({
            'url': url,
            'status': status,
            'time': time_ms
        })
    
    if time_ms > 1000:  # Более 1 секунды
        slow_requests.append({
            'url': url,
            'time': time_ms,
            'status': status
        })

print("=" * 80)
print("ANALIZ HAR FAILA - localhost.har")
print("=" * 80)

print("\nOSNOVNYE METRIKI:")
print(f"  Vsego zaprosov: {len(har['log']['entries'])}")
print(f"  Data: {har['log']['pages'][0]['startedDateTime']}")
print(f"  onContentLoad: {har['log']['pages'][0]['pageTimings']['onContentLoad']:.0f}ms")
print(f"  onLoad: {har['log']['pages'][0]['pageTimings']['onLoad']:.0f}ms")

print("\nRAZPREDELENIE PO TIPAM:")
for dtype, urls in sorted(urls_by_type.items()):
    print(f"  {dtype}: {len(urls)}")

print("\nHTTP STATUSY:")
for status, count in sorted(stats.items()):
    emoji = 'OK' if status.endswith('200') else 'WRN' if status.startswith('Status 3') else 'ERR'
    print(f"  {emoji} {status}: {count}")

if errors:
    print(f"\nOSHIBKI ({len(errors)}):")
    error_types = defaultdict(list)
    for err in errors:
        url_short = err['url'].replace('http://localhost:8000', '')
        error_types[err['status']].append(url_short)
    
    for status in sorted(error_types.keys(), reverse=True):
        urls = error_types[status]
        print(f"\n  [{status}] ({len(urls)} oshibok):")
        for url in urls[:5]:
            print(f"    * {url}")
        if len(urls) > 5:
            print(f"    ... i eshche {len(urls) - 5}")

if slow_requests:
    print(f"\nMEDLENNYE ZAPROSY > 1sec ({len(slow_requests)}):")
    for req in sorted(slow_requests, key=lambda x: x['time'], reverse=True)[:10]:
        url_short = req['url'].replace('http://localhost:8000', '')
        status_emoji = 'OK' if req['status'] == 200 else 'ERR'
        print(f"  {status_emoji} {req['time']:.0f}ms [{req['status']}] {url_short}")

# Найти особые события
print(f"\nSPECIAL FINDINGS:")
print(f"  WebSocket connections: {len([e for e in har['log']['entries'] if 'ws' in e.get('request', {}).get('url', '').lower()])}")
print(f"  Failed requests: {len(errors)}")
print(f"  Slow requests: {len(slow_requests)}")

print("\n" + "=" * 80)

# Сохранить результаты в файл
with open('HAR_ANALYSIS.txt', 'w', encoding='utf-8') as f:
    f.write("HAR ANALYSIS RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total requests: {len(har['log']['entries'])}\n")
    f.write(f"Total errors: {len(errors)}\n")
    f.write(f"Slow requests: {len(slow_requests)}\n")
    f.write(f"\nAPI URLs:\n")
    for url in urls_by_type.get('API', []):
        f.write(f"  {url}\n")
    if errors:
        f.write(f"\nERROR DETAILS:\n")
        for err in errors:
            f.write(f"  [{err['status']}] {err['url']}\n")

