#!/usr/bin/env python3
"""Arhivirovanie lishnih MD failov"""
import os
import shutil
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Failya kotorye ostaviayem v korne
KEEP = {
    'README.md',
    'BUSINESS_STATUS_BRIEF.md',
    'PROJECT_BUSINESS_REPORT.md'
}

# Vse ostalnye MD failya arhiviruem
docs_archive = 'docs_archive'
if not os.path.exists(docs_archive):
    os.makedirs(docs_archive)

moved = []
for file in os.listdir('.'):
    if file.endswith('.md') and file not in KEEP:
        src = file
        dst = os.path.join(docs_archive, file)
        try:
            shutil.move(src, dst)
            moved.append(file)
            print(f'OK {file}')
        except Exception as e:
            print(f'ERR {file}: {e}')

print(f'\nVsego arhivirovano: {len(moved)} failov')
print(f'\nOstalysy v korne:')
for f in KEEP:
    if os.path.exists(f):
        print(f'  OK {f}')

