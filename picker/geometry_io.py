"""
geometry_io.py  (v17)

Geometry input and output for the first-arrival picker:

  * read_table()            csv / txt / dat / xls / xlsx tables (delimiter and header detected)
  * TopoSpec, load_topography()
                            external receiver or shot topography with user-chosen columns
                            (x, optional y, elevation, optional station / shot number)
  * receivers_from_topo(), shots_from_topo(), shots_at_geophones()
                            turn a loaded table into along-line distances, map coordinates
                            and elevations for import_survey()
  * segy_layout(), read_geometry_headers(), write_geometry_headers(), backup_files()
                            write the geometry into the trace headers of SEG-Y or SU files
                            by patching the header bytes in place (the samples are never
                            touched), after copying the original files to a backup folder

All functions are pure Python + numpy (+ pandas for tables) and never need Qt.
"""
import os
import re
import shutil
import struct
import datetime
from dataclasses import dataclass, asdict

import numpy as np

TABLE_EXTENSIONS = ('.csv', '.txt', '.dat', '.xls', '.xlsx', '.tsv', '.prn')


# =====================================================================
# Tables
# =====================================================================
def _is_number(s):
    try:
        float(str(s).strip().replace(',', '.') if str(s).count(',') == 1 and '.' not in str(s) else str(s).strip())
        return True
    except ValueError:
        return False


def sniff_delimiter(lines):
    """Delimiter of a text table: tab, comma, semicolon or whitespace (None)."""
    lines = [l for l in lines if l.strip() and not l.lstrip().startswith('#')][:30]
    if not lines:
        return None
    for d in ('\t', ';', ','):
        counts = [l.count(d) for l in lines]
        if min(counts) > 0 and len(set(counts[1:] or counts)) == 1:
            return d
    return None                                  # runs of spaces


def read_table(path, sheet=None, header='auto', delimiter='auto'):
    """
    Read a table of numbers into a pandas DataFrame with string column names.

    header: 'auto' (first row is a header if it has a non-numeric cell), True or False.
    Without a header the columns are named 'column 1', 'column 2', ...
    Rows starting with '#' are skipped in text files. Raises ValueError with a
    readable message when the file cannot be read.
    """
    import pandas as pd
    if not path or not os.path.isfile(path):
        raise ValueError(f'File not found: {path}')
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in ('.xls', '.xlsx', '.xlsm'):
            raw = pd.read_excel(path, sheet_name=sheet if sheet not in (None, '') else 0,
                                header=None, dtype=object)
        else:
            with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = f.read().splitlines()
            body = [l for l in lines if l.strip() and not l.lstrip().startswith('#')]
            if not body:
                raise ValueError('the file has no data rows')
            d = sniff_delimiter(body) if delimiter == 'auto' else delimiter
            rows = [(l.split(d) if d else l.split()) for l in body]
            rows = [[c.strip().strip('"').strip("'") for c in r] for r in rows]
            width = max(len(r) for r in rows)
            rows = [r + [''] * (width - len(r)) for r in rows]
            raw = pd.DataFrame(rows, dtype=object)
    except ImportError as e:
        raise ValueError(f'Reading {ext} files needs an extra package ({e}). '
                         f'Install it with: pip install openpyxl xlrd') from e
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f'Could not read {os.path.basename(path)}: {type(e).__name__}: {e}') from e
    raw = raw.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    if raw.empty:
        raise ValueError('the table is empty')
    first = [str(v) for v in raw.iloc[0].tolist()]
    if header == 'auto':
        has_header = any(v.strip() and not _is_number(v) for v in first)
    else:
        has_header = bool(header)
    if has_header:
        names, raw = first, raw.iloc[1:].reset_index(drop=True)
    else:
        names = [f'column {i + 1}' for i in range(raw.shape[1])]
    clean, seen = [], {}
    for i, n in enumerate(names):
        n = (n or '').strip() or f'column {i + 1}'
        if n in seen:
            seen[n] += 1
            n = f'{n} ({seen[n]})'
        else:
            seen[n] = 0
        clean.append(n)
    raw.columns = clean
    return raw


def excel_sheets(path):
    """Sheet names of an Excel file ([] for text files or on error)."""
    if os.path.splitext(path or '')[1].lower() not in ('.xls', '.xlsx', '.xlsm'):
        return []
    try:
        import pandas as pd
        return list(pd.ExcelFile(path).sheet_names)
    except Exception:
        return []


def numeric_column(df, col, what):
    """Column as float array; raises ValueError naming the bad rows."""
    import pandas as pd
    if col not in df.columns:
        raise ValueError(f'{what}: column "{col}" not in the table (columns: {", ".join(df.columns)})')
    vals = pd.to_numeric(df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
                         if df[col].dtype == object else df[col], errors='coerce')
    bad = np.where(~np.isfinite(vals.to_numpy(dtype=float)))[0]
    if len(bad):
        rows = ', '.join(str(b + 1) for b in bad[:8])
        raise ValueError(f'{what}: column "{col}" has {len(bad)} non-numeric value(s) '
                         f'(data rows {rows}{" ..." if len(bad) > 8 else ""})')
    return vals.to_numpy(dtype=float)


def guess_columns(columns):
    """Best guesses for (x, y, elevation, number) columns from header names."""
    low = [c.lower() for c in columns]

    def find(*keys, exclude=()):
        for k in keys:
            for c, l in zip(columns, low):
                if re.search(k, l) and not any(re.search(e, l) for e in exclude):
                    return c
        return None
    x = find(r'^x$', r'east', r'utm ?x', r'utmx', r'local_x', r'^x[_ ]', r'dist', r'offset', exclude=(r'scalar',))
    y = find(r'^y$', r'north', r'utm ?y', r'utmy', r'^y[_ ]', exclude=(r'local_y', r'scalar'))
    z = find(r'elev', r'^z$', r'height', r'alt', r'topo', r'^h$')
    n = find(r'^(station|stn|recv|receiver|geophone|shot|sp|trace|no|nr|id|number|index)',
             r'(no|nr|num|number|index|id)$')
    used = {x, y, z}
    if n in used:
        n = None
    nums = list(columns)
    if x is None and len(nums) >= 2:                   # plain numeric table: x [y] z
        if len(nums) == 2:
            x, z = nums[0], nums[1]
        elif len(nums) >= 3:
            x, y, z = nums[0], nums[1], nums[2]
    return x, y, z, n


# =====================================================================
# Topography specification
# =====================================================================
@dataclass
class TopoSpec:
    """Where to find x, (y), elevation and an optional station/shot number in a table.

    y_col empty  -> x_col is the DISTANCE ALONG THE LINE (m), used as it is.
    y_col given  -> (x_col, y_col) are map coordinates (easting, northing); the
                    along-line distance is computed from them (first receiver = 0).
    number_col   -> receivers: trace number (1 = first trace of the widest shot);
                    shots: shot number in import order (1 = first file).
                    Empty -> row i is receiver / shot i.
    """
    path: str = ''
    x_col: str = ''
    y_col: str = ''
    z_col: str = ''
    number_col: str = ''
    sheet: str = ''
    header: object = 'auto'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        if not d:
            return None
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def describe(self):
        cols = f"x='{self.x_col}'" + (f", y='{self.y_col}'" if self.y_col else ' (distance along line)') + \
               f", elevation='{self.z_col}'" + (f", number='{self.number_col}'" if self.number_col else '')
        return f'{os.path.basename(self.path)} [{cols}]'


@dataclass
class Topo:
    """A loaded topography table."""
    x: object
    y: object                     # None = x is along-line distance
    z: object
    number: object                # None = row order
    spec: TopoSpec = None

    @property
    def is_map(self):
        return self.y is not None

    def __len__(self):
        return len(self.x)


def load_topography(spec):
    """Read the table of a TopoSpec. Raises ValueError with a readable message."""
    if spec is None or not spec.path:
        raise ValueError('No topography file chosen.')
    if not spec.x_col or not spec.z_col:
        raise ValueError(f'{os.path.basename(spec.path)}: choose the x and the elevation columns.')
    df = read_table(spec.path, sheet=spec.sheet or None, header=spec.header)
    what = os.path.basename(spec.path)
    x = numeric_column(df, spec.x_col, what)
    z = numeric_column(df, spec.z_col, what)
    y = numeric_column(df, spec.y_col, what) if spec.y_col else None
    n = numeric_column(df, spec.number_col, what) if spec.number_col else None
    if n is not None:
        if np.any(np.abs(n - np.round(n)) > 1e-6):
            raise ValueError(f'{what}: the number column "{spec.number_col}" must hold whole numbers.')
        n = np.round(n).astype(int)
        dup = sorted({int(v) for v in n if list(n).count(v) > 1})
        if dup:
            raise ValueError(f'{what}: number(s) {dup[:10]} appear more than once in "{spec.number_col}".')
    return Topo(x=x, y=y, z=z, number=n, spec=spec)


def _rows_for(topo, count, label):
    """Index of the table row for items 1..count (by number column or row order)."""
    what = os.path.basename(topo.spec.path) if topo.spec else 'topography'
    if topo.number is None:
        if len(topo) != count:
            raise ValueError(f'{what}: {len(topo)} rows but the data have {count} {label}s. '
                             f'Add a {label}-number column, or give one row per {label}.')
        return np.arange(count)
    rows = {int(v): i for i, v in enumerate(topo.number)}
    missing = [k for k in range(1, count + 1) if k not in rows]
    if missing:
        raise ValueError(f'{what}: no row for {label} number(s) '
                         f'{", ".join(str(m) for m in missing[:12])}{" ..." if len(missing) > 12 else ""} '
                         f'(the {label}s are numbered 1-{count}).')
    return np.array([rows[k] for k in range(1, count + 1)])


def along_line(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.zeros(len(x))
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])


def receivers_from_topo(topo, n_receivers):
    """(rx, ry, rdist, relev, is_map, notes) for the master receiver list from a Topo."""
    r = _rows_for(topo, n_receivers, 'receiver')
    notes = []
    z = topo.z[r]
    if topo.is_map:
        rx, ry = topo.x[r], topo.y[r]
        rdist = along_line(rx, ry)
        d = np.diff(rdist)
        if np.any(d <= 0):
            k = int(np.argmin(d))
            notes.append(f'Receivers {k + 1} and {k + 2} have the same map coordinates.')
    else:
        rdist = topo.x[r].astype(float)
        rx, ry = rdist.copy(), np.zeros(n_receivers)
        d = np.diff(rdist)
        if len(d) and not (np.all(d > 0) or np.all(d < 0)):
            bad = int(np.where((d <= 0) if np.median(d) > 0 else (d >= 0))[0][0])
            notes.append(f'Distances along the line are not in order at receivers {bad + 1}-{bad + 2} '
                         f'({rdist[bad]:g} m, {rdist[bad + 1]:g} m): the offset view needs them sorted.')
    return rx, ry, rdist, z, topo.is_map, notes


def shots_from_topo(topo, n_shots, recv_is_map):
    """(sx, sy, selev, sdist_or_None) from a shot Topo. sdist is None for map
    coordinates (they are projected on the receiver line by the caller)."""
    r = _rows_for(topo, n_shots, 'shot')
    if topo.is_map and not recv_is_map:
        raise ValueError('The shot file has map coordinates (x and y) but the receiver positions are '
                         'distances along the line: use the same kind of coordinates for both.')
    if not topo.is_map and recv_is_map:
        raise ValueError('The shot file has only distances along the line (no y column) but the '
                         'receivers have map coordinates: use the same kind of coordinates for both.')
    if topo.is_map:
        return topo.x[r], topo.y[r], topo.z[r], None
    sd = topo.x[r].astype(float)
    return sd.copy(), np.zeros(n_shots), topo.z[r], sd


def shots_at_geophones(shot_pos, recv_pos, rx, ry, rdist, relev):
    """Shot coordinates taken from the receiver at each shot's picker position.
    A shot beyond the spread (no receiver at its position) is extrapolated
    linearly from the two end receivers. Returns (sx, sy, selev, sdist, notes)."""
    recv_pos = np.asarray(recv_pos, float)
    shot_pos = np.asarray(shot_pos, float)
    n = len(recv_pos)
    sp = float(np.median(np.diff(recv_pos))) if n > 1 else 1.0
    out = np.zeros((len(shot_pos), 4))
    notes = []
    for i, p in enumerate(shot_pos):
        k = (p - recv_pos[0]) / sp
        kr = int(round(k))
        if abs(k - kr) < 1e-6 and 0 <= kr < n:
            out[i] = (rx[kr], ry[kr], relev[kr], rdist[kr])
            continue
        if n < 2:
            out[i] = (rx[0], ry[0], relev[0], rdist[0])
            continue
        if k < 0:
            a, b, f = 0, 1, k
        elif k > n - 1:
            a, b, f = n - 2, n - 1, k - (n - 2)
        else:                                    # between receivers (a gap or off-grid shot)
            a = int(np.floor(k)); b = a + 1; f = k - a
        lerp = lambda v: v[a] + f * (v[b] - v[a])
        out[i] = (lerp(rx), lerp(ry), lerp(relev), lerp(rdist))
        notes.append(f'Shot {i + 1} (position {p:g}) has no receiver at its position: '
                     f'{"extrapolated" if (k < 0 or k > n - 1) else "interpolated"} from receivers '
                     f'{a + 1} and {b + 1} (x = {out[i, 3]:.2f} m, elevation {out[i, 2]:.2f}).')
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3], notes


# =====================================================================
# SEG-Y / SU trace headers (byte patching)
# =====================================================================
# 1-based byte positions of the SEG-Y rev1 trace header fields we touch
TH = dict(offset=37, gelev=41, selev=45, sdepth=49, datum_r=53, datum_s=57, wdep_s=61, wdep_g=65,
          scalel=69, scalco=71, sx=73, sy=77, gx=81, gy=85, coord_units=89, ns=115, dt=117,
          cdpx=181, cdpy=185)
SAMPLE_BYTES = {1: 4, 2: 4, 3: 2, 4: 4, 5: 4, 6: 8, 7: 3, 8: 1, 9: 8, 10: 4, 11: 2, 12: 8, 15: 3, 16: 1}


def _i(buf, pos, fmt, endian):
    size = struct.calcsize(fmt)
    return struct.unpack(endian + fmt, buf[pos - 1:pos - 1 + size])[0]


def segy_layout(path, kind='auto'):
    """
    Byte layout of a SEG-Y (.segy/.sgy) or Seismic Unix (.su) file:
    dict(kind, endian, offsets (trace header start of each trace), ns, sample_bytes).
    Raises ValueError for files that do not look like SEG-Y / SU.
    """
    size = os.path.getsize(path)
    if kind == 'auto':
        kind = 'su' if path.lower().endswith('.su') else 'segy'
    with open(path, 'rb') as f:
        head = f.read(3600)
    if kind == 'segy':
        if size < 3600 + 240:
            raise ValueError(f'{os.path.basename(path)}: too small for SEG-Y')
        endian = None
        for e in ('>', '<'):
            code = _i(head, 3225, 'h', e)
            if code in SAMPLE_BYTES:
                endian = e
                break
        if endian is None:
            raise ValueError(f'{os.path.basename(path)}: unknown data sample format in the binary header')
        sb = SAMPLE_BYTES[_i(head, 3225, 'h', endian)]
        n_ext = _i(head, 3505, 'h', endian)
        if n_ext < 0:
            raise ValueError(f'{os.path.basename(path)}: variable number of extended textual headers '
                             f'is not supported')
        pos = 3600 + 3200 * max(0, n_ext)
        ns_bin = _i(head, 3221, 'H', endian)
    else:
        endian, sb, pos, ns_bin = None, 4, 0, None
        with open(path, 'rb') as f:
            th = f.read(240)
        for e in ('>', '<'):
            ns = _i(th, 115, 'H', e)
            if ns > 0 and size % (240 + 4 * ns) == 0:
                endian = e
                break
        if endian is None:
            raise ValueError(f'{os.path.basename(path)}: does not look like a Seismic Unix (.su) file')
    offsets = []
    nss = []
    with open(path, 'rb') as f:
        while pos + 240 <= size:
            f.seek(pos)
            th = f.read(240)
            ns = _i(th, 115, 'H', endian) or (ns_bin or 0)
            if ns <= 0:
                raise ValueError(f'{os.path.basename(path)}: trace {len(offsets) + 1} has no samples')
            offsets.append(pos)
            nss.append(ns)
            pos += 240 + ns * sb
    if pos != size:
        raise ValueError(f'{os.path.basename(path)}: trace sizes do not add up to the file size '
                         f'({size - pos:+d} bytes): not a standard {kind.upper()} file')
    return dict(kind=kind, endian=endian, offsets=offsets, ns=nss, sample_bytes=sb)


def _apply(v, s):
    s = int(s)
    return v * (abs(s) if s > 0 else 1.0 / abs(s)) if s else float(v)


def read_geometry_headers(path, kind='auto'):
    """Per-trace geometry from the header bytes (scalars applied): dict of arrays."""
    lay = segy_layout(path, kind)
    e = lay['endian']
    out = {k: [] for k in ('sx', 'sy', 'selev', 'gx', 'gy', 'gelev', 'offset', 'scalco', 'scalel')}
    with open(path, 'rb') as f:
        for p in lay['offsets']:
            f.seek(p)
            th = f.read(240)
            sc, se = _i(th, TH['scalco'], 'h', e), _i(th, TH['scalel'], 'h', e)
            out['scalco'].append(sc)
            out['scalel'].append(se)
            for k in ('sx', 'sy', 'gx', 'gy'):
                out[k].append(_apply(_i(th, TH[k], 'i', e), sc))
            out['selev'].append(_apply(_i(th, TH['selev'], 'i', e), se))
            out['gelev'].append(_apply(_i(th, TH['gelev'], 'i', e), se))
            out['offset'].append(_i(th, TH['offset'], 'i', e))
    return {k: np.array(v) for k, v in out.items()}


def choose_scalar(values, digits=2):
    """SEG-Y scalar (-100, -10 or 1) keeping `digits` decimals inside int32."""
    m = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    for s, f in ((-1000, 1000.0), (-100, 100.0), (-10, 10.0), (1, 1.0)):
        if f > 10 ** digits:
            continue
        if m * f < 2 ** 31 - 1:
            return s
    raise ValueError(f'Coordinate {m:g} is too large for a SEG-Y header')


def _to_int(v, scalar):
    f = abs(scalar) if scalar < 0 else (1.0 / scalar if scalar > 0 else 1.0)
    return int(round(float(v) * f))


def backup_files(files, folder=None, tag='original_headers_backup'):
    """Copy files (with their timestamps) into a new backup folder next to the first
    file: <folder>/<tag>_<YYYYmmdd_HHMMSS>/. Returns the folder. Refuses to overwrite."""
    if not files:
        raise ValueError('No files to back up.')
    base = folder or os.path.dirname(os.path.abspath(files[0]))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = os.path.join(base, f'{tag}_{stamp}')
    k = 1
    while os.path.exists(dest):
        k += 1
        dest = os.path.join(base, f'{tag}_{stamp}_{k}')
    os.makedirs(dest)
    for f in files:
        target = os.path.join(dest, os.path.basename(f))
        if os.path.exists(target):
            raise ValueError(f'Two files called {os.path.basename(f)}: back them up separately.')
        shutil.copy2(f, target)
        if os.path.getsize(target) != os.path.getsize(f):
            raise IOError(f'Backup of {os.path.basename(f)} is incomplete.')
    return dest


def write_geometry_headers(path, sx, sy, selev, gx, gy, gelev, offset=None, kind='auto',
                           coord_digits=2, elev_digits=2):
    """
    Write one shot's geometry into the trace headers of a SEG-Y / SU file in place.
    sx, sy, selev: scalars (the source); gx, gy, gelev, offset: one value per trace
    (NaN = leave that trace's field unchanged). Coordinate / elevation scalars are
    chosen to keep 2 decimals; other fields that use the same scalar (CDP x/y,
    datum, depths) are rescaled so they keep their values. Only header bytes are
    written; the samples are never touched. Returns the number of traces written.
    """
    lay = segy_layout(path, kind)
    e = lay['endian']
    n = len(lay['offsets'])
    for a in (gx, gy, gelev):
        if np.ndim(a) and len(a) != n:
            raise ValueError(f'{os.path.basename(path)}: {n} traces but {len(a)} receiver values')
    gx, gy, gelev = (np.broadcast_to(np.asarray(a, float), (n,)) for a in (gx, gy, gelev))
    offset = None if offset is None else np.broadcast_to(np.asarray(offset, float), (n,))
    allc = np.concatenate([[sx, sy], gx[np.isfinite(gx)], gy[np.isfinite(gy)]])
    alle = np.concatenate([[selev], gelev[np.isfinite(gelev)]])
    allc, alle = allc[np.isfinite(allc)], alle[np.isfinite(alle)]
    sc = choose_scalar(allc, coord_digits) if len(allc) else 1
    se = choose_scalar(alle, elev_digits) if len(alle) else 1
    with open(path, 'r+b') as f:
        for j, p in enumerate(lay['offsets']):
            f.seek(p)
            th = bytearray(f.read(240))
            old_sc, old_se = _i(th, TH['scalco'], 'h', e), _i(th, TH['scalel'], 'h', e)

            def put(field, fmt, val):
                struct.pack_into(e + fmt, th, TH[field] - 1, val)

            def get(field, fmt='i'):
                return _i(th, TH[field], fmt, e)
            # keep fields that share the scalars (rescale with the new scalar)
            keep_c = {k: _apply(get(k), old_sc) for k in ('cdpx', 'cdpy', 'sx', 'sy', 'gx', 'gy')}
            keep_e = {k: _apply(get(k), old_se) for k in ('gelev', 'selev', 'sdepth', 'datum_r', 'datum_s',
                                                          'wdep_s', 'wdep_g')}
            new_c = dict(keep_c)
            new_e = dict(keep_e)
            for k, v in (('sx', sx), ('sy', sy), ('gx', gx[j]), ('gy', gy[j])):
                if np.isfinite(v):
                    new_c[k] = v
            for k, v in (('selev', selev), ('gelev', gelev[j])):
                if np.isfinite(v):
                    new_e[k] = v
            put('scalco', 'h', sc)
            put('scalel', 'h', se)
            for k, v in new_c.items():
                put(k, 'i', _to_int(v, sc))
            for k, v in new_e.items():
                put(k, 'i', _to_int(v, se))
            if offset is not None and np.isfinite(offset[j]):
                put('offset', 'i', int(round(offset[j])))
            if get('coord_units', 'h') not in (1, 2, 3, 4):
                put('coord_units', 'h', 1)          # 1 = length (metres or feet)
            f.seek(p)
            f.write(bytes(th))
    return n


def survey_header_geometry(survey, shot_index):
    """Values write_geometry_headers() needs for one shot of an imported Survey:
    (sx, sy, selev, gx, gy, gelev, offset) using survey.placement for shots with
    fewer traces. Offsets are signed (receiver minus shot along the line)."""
    i = shot_index
    cols = np.asarray(survey.placement.get(i, np.arange(survey.recv_xy.shape[0])), int)
    ntr = len(cols)
    gx = np.full(ntr, np.nan)
    gy = np.full(ntr, np.nan)
    ge = np.full(ntr, np.nan)
    off = np.full(ntr, np.nan)
    ok = cols >= 0
    gx[ok] = survey.recv_xy[cols[ok], 0]
    gy[ok] = survey.recv_xy[cols[ok], 1]
    ge[ok] = np.asarray(survey.recv_elev, float)[cols[ok]]
    off[ok] = np.asarray(survey.recv_dist, float)[cols[ok]] - float(survey.shot_dist[i])
    return (float(survey.shot_xy[i, 0]), float(survey.shot_xy[i, 1]), float(survey.shot_elev[i]),
            gx, gy, ge, off)


def write_survey_headers(survey, backup=True, backup_folder=None, progress=None):
    """Back up the survey's files, then write its geometry into their trace headers.
    Returns (backup_folder or None, report lines). Raises on the first failure
    (the backup is complete before any file is changed)."""
    files = list(survey.files)
    dest = backup_files(files, backup_folder) if backup else None
    report = []
    for i, f in enumerate(files):
        sx, sy, se, gx, gy, ge, off = survey_header_geometry(survey, i)
        n = write_geometry_headers(f, sx, sy, se, gx, gy, ge, off)
        back = read_geometry_headers(f)
        err = max(np.nanmax(np.abs(back['gx'] - np.where(np.isfinite(gx), gx, back['gx']))),
                  np.nanmax(np.abs(back['gelev'] - np.where(np.isfinite(ge), ge, back['gelev']))),
                  abs(back['sx'][0] - sx) if np.isfinite(sx) else 0.0)
        if err > 0.006:
            raise IOError(f'{os.path.basename(f)}: headers read back differ by {err:g} m')
        report.append(f'{os.path.basename(f)}: {n} trace headers written')
        if progress is not None:
            progress(i, len(files), f)
    return dest, report
