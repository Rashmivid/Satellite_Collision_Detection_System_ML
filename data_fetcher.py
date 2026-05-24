"""
data_fetcher.py — Fixed version
- Updated CelesTrak URLs (old gp.php endpoint returns 403)
- Added all group aliases the frontend uses
"""
import os
import requests
from skyfield.api import load, EarthSatellite

# ── Working CelesTrak endpoints (as of 2026) ───────────────────────────────────
# New URL format: https://celestrak.org/SOCRATES/query.php  (for conjunctions)
# TLE format:     https://celestrak.org/NORAD/elements/gp.php?GROUP=x&FORMAT=tle
# Fallback:       https://celestrak.org/NORAD/elements/<name>.txt

CELESTRAK_GROUPS = {
    # Frontend value  →  CelesTrak GROUP param
    'starlink':       'starlink',
    'active':         'active',
    'iridium':        'iridium-NEXT',
    'stations':       'stations',
    'weather':        'weather',
    'debris':         '1982-092',      # Cosmos debris — valid group
    'oneweb':         'oneweb',
    'gps':            'gps-ops',
    'glonass':        'glo-ops',
}

# Fallback direct .txt URLs for groups that 403 on gp.php
DIRECT_URLS = {
    'stations': 'https://celestrak.org/NORAD/elements/stations.txt',
    'weather':  'https://celestrak.org/NORAD/elements/weather.txt',
    'active':   'https://celestrak.org/NORAD/elements/active.txt',
    'debris':   'https://celestrak.org/NORAD/elements/2012-044.txt',
}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_tle_data(group: str = 'starlink') -> str:
    """
    Fetch TLE data from CelesTrak for the given group.
    Tries the GP API first, falls back to direct .txt URL,
    then falls back to cached file if network fails.
    Returns path to saved TLE file.
    """
    group = group.lower().strip()

    # Normalise group name
    group_key   = group if group in CELESTRAK_GROUPS else 'starlink'
    celestrak_g = CELESTRAK_GROUPS[group_key]
    file_path   = os.path.join(DATA_DIR, f'{group_key}.tle')

    headers = {
        'User-Agent': 'OrbitGuard/1.0 (satellite collision detection research)',
        'Accept':     'text/plain',
    }

    # Try 1: GP API (new format)
    urls_to_try = [
        f'https://celestrak.org/NORAD/elements/gp.php?GROUP={celestrak_g}&FORMAT=tle',
        f'https://celestrak.org/NORAD/elements/gp.php?GROUP={celestrak_g}&FORMAT=TLE',
    ]
    # Try 2: Direct .txt fallback
    if group_key in DIRECT_URLS:
        urls_to_try.append(DIRECT_URLS[group_key])

    content = None
    for url in urls_to_try:
        try:
            print(f'[data_fetcher] Fetching {url}')
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200 and len(r.text.strip()) > 100:
                content = r.text
                print(f'[data_fetcher] ✓ Got {len(content)} chars from {url}')
                break
            else:
                print(f'[data_fetcher] ✗ {r.status_code} from {url}')
        except Exception as e:
            print(f'[data_fetcher] ✗ Error fetching {url}: {e}')

    if content:
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path

    # Try 3: Use cached file from previous successful fetch
    if os.path.exists(file_path):
        print(f'[data_fetcher] ⚠ Using cached TLE file: {file_path}')
        return file_path

    raise RuntimeError(
        f'Could not fetch TLE data for group "{group_key}". '
        f'Tried: {urls_to_try}. No cache available.'
    )


def load_satellites(file_path: str):
    """
    Load TLE file and return (list of EarthSatellite, Timescale).
    Skips malformed lines gracefully.
    """
    ts   = load.timescale()
    sats = []

    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # TLE format: name line, line1 (starts with 1), line2 (starts with 2)
    i = 0
    while i < len(lines) - 2:
        name  = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        if line1.startswith('1 ') and line2.startswith('2 '):
            try:
                sat = EarthSatellite(line1, line2, name, ts)
                sats.append(sat)
            except Exception as e:
                print(f'[data_fetcher] Skipping malformed TLE for {name}: {e}')
            i += 3
        else:
            i += 1

    print(f'[data_fetcher] Loaded {len(sats)} satellites from {file_path}')
    return sats, ts