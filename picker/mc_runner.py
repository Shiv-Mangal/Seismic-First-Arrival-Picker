# -*- coding: utf-8 -*-
"""
mc_runner.py  (v21)
===================
Runs a Monte Carlo set in a process of its own (started by the inversion window, or from a
terminal / a cluster job). It spreads the inversions over worker processes and writes
mc_status.json, runs_summary.csv and mc.log in the set folder.

    python mc_runner.py --mc <set folder>          (mc_settings.json in the folder)
    python mc_runner.py --mc <set folder> --workers 16

A file named STOP in the set folder cancels the waiting runs; running ones stop after their
current iteration. Starting the same folder again carries on with the runs not done.
"""
import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Monte Carlo travel-time inversion (first-arrival picker v21)')
    ap.add_argument('--mc', required=True, help='Monte Carlo set folder (with mc_settings.json)')
    ap.add_argument('--workers', type=int, default=None, help='worker processes (default: the set\'s setting)')
    a = ap.parse_args(argv)
    import mc_core as mc
    folder = os.path.abspath(a.mc)
    if a.workers:
        m = mc.MCSettings.load(os.path.join(folder, 'mc_settings.json'))
        m.workers = a.workers
        m.save(os.path.join(folder, 'mc_settings.json'))
    try:
        os.remove(os.path.join(folder, mc.STOP_FILE))
    except OSError:
        pass
    logf = open(os.path.join(folder, 'mc.log'), 'a', buffering=1)

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')

    try:
        st = mc.run_set(folder, log=log)
        log(st.get('message', ''))
        return 0
    except Exception as e:                                 # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        mc.write_mc_status(folder, state='failed', message=f'{type(e).__name__}: {e}')
        return 1
    finally:
        logf.close()


if __name__ == '__main__':
    sys.exit(main())
