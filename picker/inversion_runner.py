# -*- coding: utf-8 -*-
"""
inversion_runner.py  (v20)
==========================
Runs one pyGIMLi inversion in a process of its own, so the picker and the inversion window
stay responsive and the run keeps going if the window is closed (as FWI Studio's
stage_runner). The window starts it and follows status.json, history.csv and iterations/.

    python inversion_runner.py --run <run folder>          (settings.json in the folder)
    python inversion_runner.py --data file.txt --out <folder> [--lam 30 ...]   (without the window)

A file named STOP in the run folder stops it after the current iteration; the last model is
still written.
"""
import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None):
    ap = argparse.ArgumentParser(description='pyGIMLi travel-time inversion (first-arrival picker v20)')
    ap.add_argument('--run', default='', help='run folder holding settings.json')
    ap.add_argument('--data', default='', help='pyGIMLi data file (without --run)')
    ap.add_argument('--out', default='', help='folder for a new run folder (without --run)')
    for k, v in (('lam', float), ('z_weight', float), ('v_top', float), ('v_bottom', float),
                 ('v_min', float), ('v_max', float), ('max_iter', int), ('para_depth', float),
                 ('para_max_cell', float), ('para_dx', float)):
        ap.add_argument('--' + k.replace('_', '-'), type=v, default=None)
    a = ap.parse_args(argv)
    import inversion_core as ic
    if a.run:
        run_dir = a.run
        s = ic.InvSettings.load(os.path.join(run_dir, 'settings.json'))
    else:
        if not a.data:
            ap.error('give --run, or --data (and --out)')
        s = ic.InvSettings(data_file=os.path.abspath(a.data),
                           line=os.path.splitext(os.path.basename(a.data))[0])
        for k in ('lam', 'z_weight', 'v_top', 'v_bottom', 'v_min', 'v_max', 'max_iter', 'para_depth',
                  'para_max_cell', 'para_dx'):
            if getattr(a, k) is not None:
                setattr(s, k, getattr(a, k))
        run_dir = ic.new_run_dir(a.out or os.path.join(os.path.dirname(os.path.abspath(a.data)),
                                                       f'{s.line}_inversion'))
    log_path = os.path.join(run_dir, 'run.log')
    logf = open(log_path, 'a', buffering=1)

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')

    log(f'Run folder: {run_dir}')
    try:
        st = ic.run(s, run_dir, log=log)
        log(st.get('message', ''))
        return 0 if st.get('state') in ('done', 'stopped') else 1
    except Exception as e:                                    # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        ic.write_status(run_dir, state='failed', message=f'{type(e).__name__}: {e}', traceback=tb)
        return 1
    finally:
        logf.close()


if __name__ == '__main__':
    sys.exit(main())
