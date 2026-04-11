"""
Generates and installs a macOS LaunchAgent plist so the voice assistant
auto-starts on login.

Detects the current Python interpreter and absolute path to main.py at runtime,
so this works correctly inside any virtualenv or conda environment.

Usage:
    python install_launchagent.py          # install
    python install_launchagent.py --remove # uninstall
"""

import os
import subprocess
import sys
from pathlib import Path

from config import LAUNCHAGENT_LABEL, LAUNCHAGENT_PLIST_NAME, LOG_DIR

LAUNCHAGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = LAUNCHAGENTS_DIR / LAUNCHAGENT_PLIST_NAME


def generate_plist(python_path: str, main_py_path: str, log_dir: str) -> str:
    """Return a LaunchAgent plist XML string."""
    stdout_log = os.path.join(log_dir, "launchagent_stdout.log")
    stderr_log = os.path.join(log_dir, "launchagent_stderr.log")

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHAGENT_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{main_py_path}</string>
    </array>

    <!-- Start at login -->
    <key>RunAtLoad</key>
    <true/>

    <!-- Restart automatically if it crashes -->
    <key>KeepAlive</key>
    <true/>

    <!-- Working directory = project root -->
    <key>WorkingDirectory</key>
    <string>{os.path.dirname(main_py_path)}</string>

    <!-- Log stdout/stderr to files in ~/.voice-assistant/ -->
    <key>StandardOutPath</key>
    <string>{stdout_log}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>

    <!-- Give GUI access so rumps menu bar works -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{os.path.dirname(python_path)}</string>
    </dict>
</dict>
</plist>
"""


def install() -> None:
    python_path = sys.executable
    main_py_path = str(Path(__file__).parent.resolve() / "main.py")
    log_dir = os.path.expanduser(LOG_DIR)

    # Ensure directories exist
    LAUNCHAGENTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if not Path(main_py_path).exists():
        print(f"ERROR: main.py not found at {main_py_path}")
        sys.exit(1)

    plist_content = generate_plist(python_path, main_py_path, log_dir)

    # Write plist
    PLIST_PATH.write_text(plist_content, encoding="utf-8")
    print(f"Plist written to: {PLIST_PATH}")

    # Unload if already loaded (ignore errors)
    subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)],
        capture_output=True,
    )

    # Load the new agent
    result = subprocess.run(
        ["launchctl", "load", str(PLIST_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: launchctl load returned non-zero: {result.stderr}")
    else:
        print("LaunchAgent loaded successfully.")

    print(f"\nVoice Assistant will auto-start on next login.")
    print(f"  Python:  {python_path}")
    print(f"  Script:  {main_py_path}")
    print(f"  Logs:    {log_dir}/")
    print(f"\nTo uninstall:  python install_launchagent.py --remove")


def remove() -> None:
    if not PLIST_PATH.exists():
        print(f"No plist found at {PLIST_PATH} — nothing to remove.")
        return

    # Unload
    result = subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: launchctl unload: {result.stderr}")
    else:
        print("LaunchAgent unloaded.")

    PLIST_PATH.unlink()
    print(f"Plist removed: {PLIST_PATH}")
    print("Auto-start disabled.")


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    else:
        install()
