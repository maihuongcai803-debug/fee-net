"""
Agent Auto Bridge: Local Multi-Agent Event Bus & Notification Bridge
Monitors AGENT_COMMUNICATION.md and project events to enable seamless,
human-in-the-loop-free coordination between Antigravity (Agent A) and WorkBuddy (Agent B).
"""

import os
import sys
import time
import json
from datetime import datetime
import winsound

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMM_FILE = os.path.join(REPO_ROOT, "AGENT_COMMUNICATION.md")
EVENT_LOG = os.path.join(REPO_ROOT, "assets", "agent_sync_event.json")
PROBE_DIR = os.path.join(REPO_ROOT, "assets", "probe_results")

def log_event(source, message, details=None):
    os.makedirs(os.path.dirname(EVENT_LOG), exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "message": message,
        "details": details or {}
    }
    with open(EVENT_LOG, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{source}] {message}")

def check_communication_file(last_mtime):
    if not os.path.exists(COMM_FILE):
        return last_mtime
    mtime = os.path.getmtime(COMM_FILE)
    if last_mtime is not None and mtime > last_mtime:
        try:
            with open(COMM_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            # Find the last message section
            sections = content.split("### [")
            if len(sections) > 1:
                last_sec = "### [" + sections[-1]
                header = last_sec.splitlines()[0] if last_sec.splitlines() else "Unknown"
                log_event("CommunicationBus", f"New dialogue entry detected: {header}")
                # Play notification sound
                try:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error reading comm file: {e}")
    return mtime

def main():
    print(f"[*] Agent Auto Bridge started. Monitoring: {COMM_FILE}")
    last_mtime = os.path.getmtime(COMM_FILE) if os.path.exists(COMM_FILE) else None
    
    try:
        while True:
            last_mtime = check_communication_file(last_mtime)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[*] Agent Auto Bridge stopped.")

if __name__ == "__main__":
    main()
