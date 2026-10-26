# -*- coding: utf-8 -*-
"""
Poll the Kaggle kernel until v5 finishes, then download the result bundle.
Runs in the background; writes progress to scratch/kaggle_v5_watch.log.
"""
import subprocess, time, os, sys, json

KERNEL = "maihuongcai/notebook0e7b0400ab"
OUT_DIR = r"D:\machine_learning\scratch\kaggle_output_v5"
LOG = r"D:\machine_learning\scratch\kaggle_v5_watch.log"
DEADLINE_MIN = 300
POLL_SEC = 90


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status():
    # Try Python API first for stability
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        res = api.kernels_status(KERNEL)
        if isinstance(res, dict) and "status" in res:
            return res["status"].upper()
    except Exception as e:
        pass

    # Fallback to CLI
    try:
        r = subprocess.run(["kaggle", "kernels", "status", KERNEL],
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout or "") + (r.stderr or "")
        for st in ["RUNNING", "COMPLETE", "ERROR", "CANCELLED", "INCOMPLETE"]:
            if st in out:
                return st
    except Exception:
        pass
    return "UNKNOWN"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log(f"resuming watch on {KERNEL} v5 (extended deadline {DEADLINE_MIN} min)")
    t0 = time.time()
    st = None
    while (time.time() - t0) / 60 < DEADLINE_MIN:
        try:
            st = status()
        except Exception as ex:
            log(f"status check failed: {ex}")
            time.sleep(POLL_SEC)
            continue
        log(f"status = {st}  (elapsed {(time.time()-t0)/60:.1f} min)")
        if st in ("COMPLETE", "ERROR", "CANCELLED", "INCOMPLETE"):
            break
        time.sleep(POLL_SEC)

    if st == "COMPLETE":
        log("kernel COMPLETE -> downloading outputs")
        for attempt in range(5):
            try:
                r = subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", OUT_DIR],
                                   capture_output=True, text=True, timeout=3600)
                log(f"download rc={r.returncode} {(r.stdout or '')[-300:]} {(r.stderr or '')[-300:]}")
                if r.returncode == 0:
                    break
            except Exception as ex:
                log(f"download attempt {attempt+1} failed: {ex}")
            time.sleep(10)
            
        files = []
        for root, _, fs in os.walk(OUT_DIR):
            for fn in fs:
                p = os.path.join(root, fn)
                files.append((os.path.relpath(p, OUT_DIR), os.path.getsize(p)))
        log(f"downloaded {len(files)} files into {OUT_DIR}")
        for rel, sz in sorted(files, key=lambda x: -x[1])[:20]:
            log(f"  {rel}  ({sz/1e6:.1f} MB)")
    else:
        log(f"final status = {st}; nothing downloaded")


if __name__ == "__main__":
    main()
