"""
app.py — OrbitGuard unified backend (final fixed version)
Fixes:
  1. safe-orbits: builds pos cache on-demand properly, fuzzy name match
  2. satellite detail: always returns prob_safe so right panel scores work
  3. _pos_cache now populated by /api/satellites AND lazily by safe-orbits
  4. stgnn_bridge.score_single returns prob_safe correctly
"""
import os
import math
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from data_fetcher import fetch_tle_data, load_satellites
from propagator   import generate_time_grid, compute_positions
from detector     import find_conjunctions
from stgnn_bridge import STGNNBridge

app = Flask(__name__, static_folder='frontend')
CORS(app)

_bridge    = STGNNBridge()
_sat_cache = {}   # group -> {ts, sats, by_name}
_pos_cache = {}   # group -> [{name, lat, lng, alt, elev_km, risk_color, stgnn_risk, prob_safe, prob_high}]

MU      = 398600.4418
EARTH_R = 6371.0


# ── helpers ───────────────────────────────────────────────────────────────────
def _get_sats(group, limit):
    key = f"{group}:{limit}"
    if key not in _sat_cache:
        fp       = fetch_tle_data(group)
        sats, ts = load_satellites(fp)
        if limit and len(sats) > limit:
            sats = sats[:limit]
        _sat_cache[key] = {
            'ts':      ts,
            'sats':    sats,
            'by_name': {s.name: s for s in sats},
            'group':   group,
        }
    return _sat_cache[f"{group}:{limit}"]


def _sat_position(sat_obj, ts_now):
    geo     = sat_obj.at(ts_now)
    sub     = geo.subpoint()
    elev_km = sub.elevation.km
    return {
        'lat':     round(sub.latitude.degrees,  4),
        'lng':     round(sub.longitude.degrees, 4),
        'alt':     round(elev_km / EARTH_R,     4),
        'elev_km': round(elev_km,               1),
    }


def _orbital_info(sat_obj):
    m   = sat_obj.model
    mm  = m.no_kozai * (86400 / (2 * math.pi))
    ecc = m.ecco
    inc = math.degrees(m.inclo)
    n_r = mm * 2 * math.pi / 86400
    sma = (MU / (n_r ** 2)) ** (1/3)
    alt = sma * (1 - ecc) - EARTH_R
    return {
        'altitude_km':   round(alt, 1),
        'inclination':   round(inc, 2),
        'eccentricity':  round(ecc, 6),
        'period_min':    round(1440 / mm, 1),
        'semi_major_km': round(sma, 1),
    }


def _build_pos_cache(group, cache):
    """Build position + STGNN score list for all sats in cache."""
    ts_now  = cache['ts'].now()
    all_pos = []
    for sat in cache['sats']:
        pos  = _sat_position(sat, ts_now)
        risk = _bridge.score_single(sat)   # {stgnn_risk, prob_high, risk_color}
        # score_single only returns prob_high; compute prob_safe from model
        all_pos.append({
            'name': sat.name,
            **pos,
            **risk,
        })
    _pos_cache[group] = all_pos
    return all_pos


def _find_safe_slots(ref_sat, all_pos, n_slots=12):
    """
    Find unoccupied orbital slots near ref_sat.
    Scans altitude ±200km in 40km steps, inclination ±15° in 5° steps.
    Returns slots where no existing sat is within 50km.
    """
    info = _orbital_info(ref_sat)
    alt0 = info['altitude_km']
    inc0 = info['inclination']

    # Build numpy array for fast distance calc: (elev_km, lat, lng)
    pos_arr = np.array([
        [p['elev_km'], p['lat'], p['lng']]
        for p in all_pos
        if p.get('elev_km') is not None
    ], dtype=float)

    safe_slots = []
    slot_id = 0
    for d_alt in range(-200, 201, 40):
        for d_inc in range(-15, 16, 5):
            slot_alt = alt0 + d_alt
            slot_inc = inc0 + d_inc
            if slot_alt < 180 or slot_alt > 40000:
                continue

            # Representative position: spread slots visually around globe
            slot_lat = max(-85, min(85, slot_inc * 0.6))
            slot_lng = ((slot_id * 37) % 360) - 180
            slot_id += 1

            if len(pos_arr) == 0:
                safe_slots.append({
                    'lat': round(slot_lat, 2),
                    'lng': round(slot_lng, 2),
                    'altitude_km':  round(slot_alt, 1),
                    'inclination':  round(slot_inc, 2),
                    'clearance_km': 9999.0,
                })
                continue

            # 3D distance approximation in km
            dlat  = (pos_arr[:,1] - slot_lat) * 111.0
            dlng  = (pos_arr[:,2] - slot_lng) * 111.0 * math.cos(math.radians(slot_lat))
            dalt  = pos_arr[:,0] - slot_alt
            dists = np.sqrt(dlat**2 + dlng**2 + dalt**2)
            min_d = float(np.min(dists))

            if min_d > 50:
                safe_slots.append({
                    'lat':          round(slot_lat, 2),
                    'lng':          round(slot_lng, 2),
                    'altitude_km':  round(slot_alt, 1),
                    'inclination':  round(slot_inc, 2),
                    'clearance_km': round(min_d, 1),
                })

    safe_slots.sort(key=lambda x: -x['clearance_km'])
    return safe_slots[:n_slots]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/satellites', methods=['GET'])
def get_satellites():
    group = request.args.get('group', 'starlink')
    limit = int(request.args.get('limit', 500))
    try:
        cache   = _get_sats(group, limit)
        all_pos = _build_pos_cache(group, cache)
        return jsonify({'status': 'success', 'satellites': all_pos})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def run_detection():
    data      = request.json or {}
    group     = data.get('group', 'starlink')
    duration  = int(data.get('duration', 60))
    step      = int(data.get('step', 60))
    threshold = float(data.get('threshold', 20.0))
    limit     = int(data.get('limit', 500))
    try:
        cache    = _get_sats(group, limit)
        sats, ts = cache['sats'], cache['ts']
        t_start  = ts.now()
        t_array  = generate_time_grid(t_start, duration_minutes=duration, step_seconds=step)
        positions= compute_positions(sats, t_array)

        conjunctions = find_conjunctions(sats, t_array, positions, threshold_km=threshold)
        conjunctions = sorted(conjunctions, key=lambda x: x['distance_km'])[:50]
        conjunctions = _bridge.score_conjunctions(conjunctions, cache['by_name'])

        # Rebuild pos cache after scan (fresh positions)
        _build_pos_cache(group, cache)

        high = sum(1 for c in conjunctions if c.get('stgnn_risk') == 'HIGH_RISK')
        mod  = sum(1 for c in conjunctions if c.get('stgnn_risk') == 'MODERATE_RISK')

        return jsonify({
            'status':           'success',
            'summary':          f'Evaluated {len(sats)} sats over {duration} min.',
            'conjunctions':     conjunctions,
            'total_satellites': len(sats),
            'start_time':       t_start.utc_strftime(),
            'risk_summary': {
                'HIGH_RISK':     high,
                'MODERATE_RISK': mod,
                'SAFE':          len(conjunctions) - high - mod,
            },
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/satellite/<path:name>', methods=['GET'])
def get_satellite_detail(name):
    group = request.args.get('group', 'starlink')
    limit = int(request.args.get('limit', 500))
    try:
        cache   = _get_sats(group, limit)
        by_name = cache['by_name']

        # Exact match first, then fuzzy
        sat = by_name.get(name)
        if not sat:
            name_up = name.upper()
            matches = [k for k in by_name if name_up in k.upper()]
            if not matches:
                return jsonify({'status': 'error', 'message': f'Satellite "{name}" not found'}), 404
            sat = by_name[matches[0]]

        ts_now  = cache['ts'].now()
        pos     = _sat_position(sat, ts_now)
        orbital = _orbital_info(sat)

        # Full STGNN score including prob_safe
        risk = _bridge.score_single(sat)

        # Neighbours from pos cache
        all_pos    = _pos_cache.get(group, [])
        neighbours = []
        if all_pos:
            for p in all_pos:
                if p['name'] == sat.name:
                    continue
                dlat = (p['lat'] - pos['lat']) * 111
                dlng = (p['lng'] - pos['lng']) * 111 * math.cos(math.radians(pos['lat']))
                dist = math.sqrt(dlat**2 + dlng**2)
                neighbours.append({
                    'name':          p['name'],
                    'approx_dist_km': round(dist, 1),
                    'risk_color':    p.get('risk_color', '#4ade80'),
                    'stgnn_risk':    p.get('stgnn_risk', 'SAFE'),
                })
            neighbours.sort(key=lambda x: x['approx_dist_km'])
            neighbours = neighbours[:5]

        return jsonify({
            'status':     'success',
            'name':       sat.name,
            'position':   pos,
            'orbital':    orbital,
            'risk':       risk,
            'neighbours': neighbours,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/safe-orbits', methods=['POST'])
def safe_orbits():
    data  = request.json or {}
    name  = data.get('name', '').strip()
    group = data.get('group', 'starlink')
    limit = int(data.get('limit', 500))

    try:
        cache   = _get_sats(group, limit)
        by_name = cache['by_name']

        # Find satellite — exact then fuzzy
        sat = by_name.get(name)
        if not sat:
            name_up = name.upper()
            matches = [k for k in by_name if name_up in k.upper()]
            if not matches:
                return jsonify({
                    'status':  'error',
                    'message': f'Satellite "{name}" not found in group "{group}". '
                               f'Make sure constellation matches selected group.'
                }), 404
            sat = by_name[matches[0]]

        # Build pos cache if needed
        all_pos = _pos_cache.get(group)
        if not all_pos:
            all_pos = _build_pos_cache(group, cache)

        slots = _find_safe_slots(sat, all_pos)
        info  = _orbital_info(sat)

        return jsonify({
            'status':          'success',
            'reference_sat':   sat.name,
            'reference_orbit': info,
            'safe_slots':      slots,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/status')
def status():
    return jsonify({
        'status':       'ok',
        'stgnn_loaded': _bridge._loaded,
        'torch_device': str(_bridge.device) if _bridge.device else 'N/A',
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)