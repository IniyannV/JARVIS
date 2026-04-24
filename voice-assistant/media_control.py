"""
Media playback control for JARVIS wake/response lifecycle.

Pauses Spotify and supported browsers (Firefox Developer Edition, Chrome,
Safari) when JARVIS activates, then resumes them after JARVIS finishes
speaking.

Pause strategy:
  - Spotify: AppleScript native pause command
  - Browsers: AppleScript + JavaScript injection into every tab

Resume strategy:
  - Only resumes apps that were actually paused by this module
  - Waits until speaker finishes, then adds a 0.5s grace delay
"""

import logging
import subprocess
import threading
import time
from typing import Optional

import state

logger = logging.getLogger("voice-assistant.media_control")

# ---------------------------------------------------------------------------
# AppleScript helpers
# ---------------------------------------------------------------------------

def _run_applescript(script: str) -> tuple[bool, str]:
    """Run an osascript snippet. Returns (success, stdout)."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.warning("osascript not found — media control unavailable.")
        return False, ""

    if result.returncode != 0:
        logger.debug("AppleScript error: %s", result.stderr.strip())
        return False, result.stderr.strip()

    return True, result.stdout.strip()


def _app_is_running(app_name: str) -> bool:
    """Return True if the named application process is currently running."""
    script = (
        f'tell application "System Events" to '
        f'(name of processes) contains "{app_name}"'
    )
    ok, output = _run_applescript(script)
    return ok and output.lower() == "true"


# ---------------------------------------------------------------------------
# Spotify
# ---------------------------------------------------------------------------

def _spotify_is_playing() -> bool:
    script = 'tell application "Spotify" to player state as string'
    ok, output = _run_applescript(script)
    return ok and output.lower() == "playing"


def _pause_spotify() -> bool:
    """Pause Spotify. Returns True if it was playing and has been paused."""
    if not _app_is_running("Spotify"):
        return False
    if not _spotify_is_playing():
        return False
    ok, _ = _run_applescript('tell application "Spotify" to pause')
    if ok:
        logger.info("Paused Spotify.")
    return ok


def _resume_spotify() -> None:
    """Resume Spotify playback."""
    if not _app_is_running("Spotify"):
        return
    ok, _ = _run_applescript('tell application "Spotify" to play')
    if ok:
        logger.info("Resumed Spotify.")


# ---------------------------------------------------------------------------
# Browser helpers (JS injection via AppleScript)
# ---------------------------------------------------------------------------

# JavaScript that pauses all HTML5 media elements on a page.
_JS_PAUSE_ALL = (
    "document.querySelectorAll('video, audio')"
    ".forEach(function(el) { if (!el.paused) { el.pause(); } });"
)

# JavaScript that resumes all HTML5 media elements that are paused.
_JS_RESUME_ALL = (
    "document.querySelectorAll('video, audio')"
    ".forEach(function(el) { if (el.paused && el.readyState >= 2) { el.play(); } });"
)

# Maps human-readable name → exact macOS process name used in AppleScript.
_BROWSER_PROCESS_NAMES: dict[str, str] = {
    "Firefox Developer Edition": "Firefox Developer Edition",
    "Google Chrome": "Google Chrome",
    "Safari": "Safari",
}


def _inject_js_chrome_family(app_name: str, js: str) -> bool:
    """
    Inject JavaScript into every tab of every window of a Chrome-family browser.
    Chrome and Firefox Developer Edition both expose a Chrome-compatible
    AppleScript dictionary.
    """
    safe_js = js.replace("\\", "\\\\").replace('"', '\\"')
    script = f"""
tell application "{app_name}"
    set windowList to every window
    repeat with w in windowList
        set tabList to every tab of w
        repeat with t in tabList
            try
                execute t javascript "{safe_js}"
            end try
        end repeat
    end repeat
end tell
"""
    ok, err = _run_applescript(script.strip())
    if not ok:
        logger.debug("%s JS injection error: %s", app_name, err)
    return ok


def _inject_js_safari(js: str) -> bool:
    """
    Inject JavaScript into every tab of every window of Safari.
    Safari uses a different AppleScript dictionary (do JavaScript).
    """
    safe_js = js.replace("\\", "\\\\").replace('"', '\\"')
    script = f"""
tell application "Safari"
    set windowList to every window
    repeat with w in windowList
        set tabList to every tab of w
        repeat with t in tabList
            try
                do JavaScript "{safe_js}" in t
            end try
        end repeat
    end repeat
end tell
"""
    ok, err = _run_applescript(script.strip())
    if not ok:
        logger.debug("Safari JS injection error: %s", err)
    return ok


def _pause_browser(app_name: str) -> bool:
    """Pause all media in the given browser. Returns True if the app was running."""
    if not _app_is_running(app_name):
        return False

    if app_name == "Safari":
        _inject_js_safari(_JS_PAUSE_ALL)
    else:
        _inject_js_chrome_family(app_name, _JS_PAUSE_ALL)

    logger.info("Sent pause to %s.", app_name)
    return True


def _resume_browser(app_name: str) -> None:
    """Resume all paused media in the given browser."""
    if not _app_is_running(app_name):
        return

    if app_name == "Safari":
        _inject_js_safari(_JS_RESUME_ALL)
    else:
        _inject_js_chrome_family(app_name, _JS_RESUME_ALL)

    logger.info("Sent resume to %s.", app_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Tracks which sources were paused by JARVIS so we only resume those.
_paused_sources: set[str] = set()
_sources_lock = threading.Lock()

# Guard against overlapping resume timers.
_resume_timer: Optional[threading.Timer] = None
_resume_timer_lock = threading.Lock()


def pause_all() -> None:
    """
    Pause Spotify and all supported browsers.

    Records which sources were actually paused so resume_after_speech()
    only restores those.
    """
    # Cancel any pending resume that hasn't fired yet (e.g. rapid re-activation).
    _cancel_pending_resume()

    paused: set[str] = set()

    if _pause_spotify():
        paused.add("Spotify")

    for browser in _BROWSER_PROCESS_NAMES:
        if _pause_browser(browser):
            paused.add(browser)

    with _sources_lock:
        _paused_sources.clear()
        _paused_sources.update(paused)

    if paused:
        logger.info("Paused media sources: %s", paused)
    else:
        logger.debug("No media sources were playing; nothing paused.")


def resume_after_speech(resume_delay: float = 0.5) -> None:
    """
    Spawn a daemon thread that waits for the speaker to finish, then resumes
    all sources that were paused by pause_all().

    Safe to call immediately after speaker.say() — the thread will block
    until is_speaking is False before starting the delay countdown.
    """
    with _sources_lock:
        if not _paused_sources:
            return  # Nothing was paused; nothing to do.

    t = threading.Thread(
        target=_resume_worker,
        args=(resume_delay,),
        daemon=True,
        name="media-resume",
    )
    t.start()

    with _resume_timer_lock:
        global _resume_timer
        _resume_timer = t  # store reference so we can guard against duplicates


def _resume_worker(delay: float) -> None:
    """Wait for speaker to finish, sleep for delay, then resume sources."""
    # Poll until the speaker is no longer active.
    poll_interval = 0.05  # 50 ms
    max_wait = 60.0        # safety cap — never block forever
    waited = 0.0

    while waited < max_wait:
        speaker = state.speaker
        if speaker is None or not speaker.is_speaking:
            break
        time.sleep(poll_interval)
        waited += poll_interval

    # Grace delay after speech ends.
    time.sleep(delay)

    with _sources_lock:
        sources = set(_paused_sources)
        _paused_sources.clear()

    if not sources:
        return

    logger.info("Resuming media sources: %s", sources)

    if "Spotify" in sources:
        _resume_spotify()

    for browser in _BROWSER_PROCESS_NAMES:
        if browser in sources:
            _resume_browser(browser)


def _cancel_pending_resume() -> None:
    """
    If a resume worker thread is running, invalidate its source list so it
    does nothing when it wakes up. (Threads are daemons and can't be killed,
    but clearing _paused_sources neutralises them.)
    """
    with _sources_lock:
        _paused_sources.clear()
