"""
Native macOS dashboard window for JARVIS using PyObjC / AppKit.

Must be instantiated and shown on the main thread.
All public update_* methods are thread-safe — they dispatch to the main queue.
"""

import logging
import threading
from collections import deque
from datetime import datetime

import objc
from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSMakeRect,
    NSMakeSize,
    NSObject,
    NSScrollView,
    NSTextView,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSMutableAttributedString, NSAttributedString
import AppKit

from config import (
    DASHBOARD_HEIGHT,
    DASHBOARD_TITLE,
    DASHBOARD_WIDTH,
    MAX_HISTORY_ENTRIES,
    MIC_METER_SATURATION_RMS,
)

logger = logging.getLogger("voice-assistant.dashboard")

# ---------------------------------------------------------------------------
# Helper: dispatch callables to the main thread via a thread-safe queue
# drained by a rumps.Timer running on the main runloop.
# ---------------------------------------------------------------------------

import collections
import rumps as _rumps

_ui_queue: collections.deque = collections.deque()


def _drain_queue(_timer=None):
    """Drain all pending UI callables. Must run on the main thread."""
    while _ui_queue:
        try:
            fn = _ui_queue.popleft()
            fn()
        except Exception as exc:
            logger.exception("Error draining UI queue: %s", exc)


def _on_main(fn):
    """Enqueue fn() for execution on the main thread. Safe from any thread."""
    _ui_queue.append(fn)


def start_drain_timer():
    """Start the polling timer that drains _ui_queue on the main thread.
    Call once after the NSApplication run loop has started."""
    timer = _rumps.Timer(_drain_queue, 1.0 / 30)  # ~30fps
    timer.start()


# ---------------------------------------------------------------------------
# Custom NSView subclass for the mic level bar
# ---------------------------------------------------------------------------

class MicLevelView(NSView):
    """A simple filled-rect progress bar showing RMS mic level."""

    def initWithFrame_(self, frame):
        self = objc.super(MicLevelView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._level = 0.0  # 0.0 – 1.0
        return self

    def setLevel_(self, level: float):
        self._level = max(0.0, min(1.0, level))
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        # Background
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.15, 0.15, 1.0).set()
        NSBezierPath.fillRect_(rect)

        # Filled portion
        bounds = self.bounds()
        fill_width = bounds.size.width * self._level
        fill_rect = NSMakeRect(0, 0, fill_width, bounds.size.height)
        if self._level > 0.6:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.3, 0.3, 1.0).set()
        elif self._level > 0.3:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.8, 0.0, 1.0).set()
        else:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.8, 0.2, 1.0).set()
        NSBezierPath.fillRect_(fill_rect)


# ---------------------------------------------------------------------------
# Window delegate — hide instead of close
# ---------------------------------------------------------------------------

class _WindowDelegate(NSObject):
    def windowShouldClose_(self, sender):
        sender.orderOut_(None)
        return False  # do not actually close/destroy the window


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    """
    Native macOS status window for JARVIS.

    Call show() once from the main thread after NSApplication is running.
    All update_* methods are safe to call from any thread.
    """

    def __init__(self) -> None:
        self._window: NSWindow = None
        self._status_circle: NSView = None
        self._status_label: NSTextView = None
        self._llm_label: NSTextView = None
        self._transcript_field: NSTextView = None
        self._action_field: NSTextView = None
        self._history_view: NSTextView = None
        self._mic_bar: MicLevelView = None
        self._history: deque = deque(maxlen=MAX_HISTORY_ENTRIES)
        self._lock = threading.Lock()
        self._built = False

    # ------------------------------------------------------------------
    # Build and show
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Build the window (if not already built) and bring it to front."""
        if not self._built:
            self._build()
            self._built = True
        self._window.makeKeyAndOrderFront_(None)

    def bring_to_front(self) -> None:
        """Bring the window to front. Safe to call from main thread only."""
        if self._built:
            self._window.makeKeyAndOrderFront_(None)
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def _build(self) -> None:
        """Construct the entire window hierarchy. Called once on the main thread."""
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, DASHBOARD_WIDTH, DASHBOARD_HEIGHT),
            style,
            NSBackingStoreBuffered,
            False,
        )
        self._window.setTitle_(DASHBOARD_TITLE)
        self._window.center()
        self._window.setReleasedWhenClosed_(False)

        delegate = _WindowDelegate.alloc().init()
        self._window.setDelegate_(delegate)
        # Keep strong reference
        self._delegate = delegate

        content = self._window.contentView()
        content_h = DASHBOARD_HEIGHT
        content_w = DASHBOARD_WIDTH
        pad = 16
        y = content_h - pad

        # ---- Status row ------------------------------------------------
        circle_size = 18
        y -= circle_size + 4

        self._status_circle = NSView.alloc().initWithFrame_(
            NSMakeRect(pad, y, circle_size, circle_size)
        )
        self._status_circle.setWantsLayer_(True)
        self._status_circle.layer().setCornerRadius_(circle_size / 2)
        self._status_circle.layer().setBackgroundColor_(
            AppKit.NSColor.grayColor().CGColor()
        )
        content.addSubview_(self._status_circle)

        self._status_label = self._make_label(
            "Not active",
            NSMakeRect(pad + circle_size + 8, y - 2, 160, circle_size + 4),
            font=NSFont.boldSystemFontOfSize_(14),
        )
        content.addSubview_(self._status_label)

        # LLM status (right-aligned)
        self._llm_label = self._make_label(
            "⚫ LLM Unknown",
            NSMakeRect(content_w - 160 - pad, y - 2, 160, circle_size + 4),
            font=NSFont.systemFontOfSize_(12),
            alignment=2,  # right
        )
        content.addSubview_(self._llm_label)

        y -= 8

        # ---- Divider ---------------------------------------------------
        y -= 1
        divider = NSView.alloc().initWithFrame_(
            NSMakeRect(pad, y, content_w - pad * 2, 1)
        )
        divider.setWantsLayer_(True)
        divider.layer().setBackgroundColor_(
            AppKit.NSColor.separatorColor().CGColor()
        )
        content.addSubview_(divider)
        y -= 8

        # ---- Transcript ------------------------------------------------
        y -= 14
        hdr = self._make_label(
            "Last heard:",
            NSMakeRect(pad, y, content_w - pad * 2, 14),
            font=NSFont.systemFontOfSize_(11),
            color=NSColor.secondaryLabelColor(),
        )
        content.addSubview_(hdr)
        y -= 4

        transcript_h = 44
        y -= transcript_h
        self._transcript_field = self._make_text_view(
            NSMakeRect(pad, y, content_w - pad * 2, transcript_h),
            font=NSFont.boldSystemFontOfSize_(15),
            placeholder="—",
        )
        content.addSubview_(self._transcript_field)
        y -= 12

        # ---- Action ----------------------------------------------------
        y -= 14
        hdr2 = self._make_label(
            "Action taken:",
            NSMakeRect(pad, y, content_w - pad * 2, 14),
            font=NSFont.systemFontOfSize_(11),
            color=NSColor.secondaryLabelColor(),
        )
        content.addSubview_(hdr2)
        y -= 4

        action_h = 32
        y -= action_h
        self._action_field = self._make_text_view(
            NSMakeRect(pad, y, content_w - pad * 2, action_h),
            font=NSFont.systemFontOfSize_(13),
            placeholder="—",
        )
        content.addSubview_(self._action_field)
        y -= 12

        # ---- Mic level meter ------------------------------------------
        y -= 14
        hdr3 = self._make_label(
            "Mic level:",
            NSMakeRect(pad, y, content_w - pad * 2, 14),
            font=NSFont.systemFontOfSize_(11),
            color=NSColor.secondaryLabelColor(),
        )
        content.addSubview_(hdr3)
        y -= 4

        meter_h = 12
        y -= meter_h
        self._mic_bar = MicLevelView.alloc().initWithFrame_(
            NSMakeRect(pad, y, content_w - pad * 2, meter_h)
        )
        self._mic_bar.setWantsLayer_(True)
        self._mic_bar.layer().setCornerRadius_(meter_h / 2)
        content.addSubview_(self._mic_bar)
        y -= 12

        # ---- History ---------------------------------------------------
        y -= 14
        hdr4 = self._make_label(
            "Command history:",
            NSMakeRect(pad, y, content_w - pad * 2, 14),
            font=NSFont.systemFontOfSize_(11),
            color=NSColor.secondaryLabelColor(),
        )
        content.addSubview_(hdr4)
        y -= 4

        history_h = max(y - pad, 80)
        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(pad, pad, content_w - pad * 2, history_h)
        )
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(AppKit.NSBezelBorder)

        self._history_view = NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, content_w - pad * 2, history_h)
        )
        self._history_view.setEditable_(False)
        self._history_view.setSelectable_(True)
        self._history_view.setFont_(NSFont.monospacedSystemFontOfSize_weight_(11, 0))
        self._history_view.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.08, 0.08, 1.0)
        )
        self._history_view.setTextColor_(NSColor.lightGrayColor())
        scroll.setDocumentView_(self._history_view)
        content.addSubview_(scroll)

        self._window.makeKeyAndOrderFront_(None)
        logger.info("Dashboard window built and shown.")

    # ------------------------------------------------------------------
    # Thread-safe public API
    # ------------------------------------------------------------------

    def update_transcript(self, text: str) -> None:
        """Update the 'Last heard' field. Safe to call from any thread."""
        def _do():
            if self._transcript_field:
                self._transcript_field.setString_(text)
        _on_main(_do)

    def update_action(self, text: str, success: bool) -> None:
        """Update the 'Action taken' field with color coding."""
        def _do():
            if self._action_field:
                color = NSColor.systemGreenColor() if success else NSColor.systemRedColor()
                self._action_field.setString_(text)
                self._action_field.setTextColor_(color)
        _on_main(_do)

    def update_processing(self, text: str = "Processing…") -> None:
        """Show a neutral 'processing' status in the Action field."""
        def _do():
            if self._action_field:
                self._action_field.setString_(text)
                self._action_field.setTextColor_(NSColor.systemYellowColor())
        _on_main(_do)

    def update_listening_state(self, is_listening: bool) -> None:
        """Update the status circle and label."""
        def _do():
            if self._status_circle and self._status_label:
                if is_listening:
                    self._status_circle.layer().setBackgroundColor_(
                        AppKit.NSColor.systemGreenColor().CGColor()
                    )
                    self._status_label.setString_("Listening...")
                else:
                    self._status_circle.layer().setBackgroundColor_(
                        AppKit.NSColor.grayColor().CGColor()
                    )
                    self._status_label.setString_("Not active")
        _on_main(_do)

    def update_llm_status(self, online: bool) -> None:
        """Update the LLM connectivity indicator."""
        def _do():
            if self._llm_label:
                if online:
                    self._llm_label.setString_("🟢 LLM Ready")
                    self._llm_label.setTextColor_(NSColor.systemGreenColor())
                else:
                    self._llm_label.setString_("🔴 LLM Offline")
                    self._llm_label.setTextColor_(NSColor.systemRedColor())
        _on_main(_do)

    def update_mic_level(self, rms: float) -> None:
        """Update the mic level meter bar (0.0–1.0 scale). High-frequency safe."""
        normalized = min(rms / max(MIC_METER_SATURATION_RMS, 1e-6), 1.0)
        def _do():
            if self._mic_bar:
                self._mic_bar.setLevel_(normalized)
        _on_main(_do)

    def add_history_entry(self, timestamp: str, transcript: str, action: str) -> None:
        """Prepend a new entry to the command history log."""
        short_ts = timestamp[11:19] if len(timestamp) >= 19 else timestamp
        line = f"[{short_ts}] {transcript} → {action}\n"
        with self._lock:
            self._history.appendleft(line)
            history_snapshot = list(self._history)

        def _do():
            if self._history_view:
                combined = "".join(history_snapshot)
                self._history_view.setString_(combined)
        _on_main(_do)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_label(self, text, frame, font=None, color=None, alignment=0):
        """Create a non-editable NSTextView used as a label."""
        tv = NSTextView.alloc().initWithFrame_(frame)
        tv.setEditable_(False)
        tv.setSelectable_(False)
        tv.setDrawsBackground_(False)
        tv.setString_(text)
        if font:
            tv.setFont_(font)
        if color:
            tv.setTextColor_(color)
        if alignment:
            tv.setAlignment_(alignment)
        return tv

    def _make_text_view(self, frame, font=None, placeholder=""):
        """Create a styled display NSTextView (non-editable, dark bg)."""
        tv = NSTextView.alloc().initWithFrame_(frame)
        tv.setEditable_(False)
        tv.setSelectable_(True)
        tv.setString_(placeholder)
        tv.setDrawsBackground_(True)
        tv.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 1.0)
        )
        tv.setTextColor_(NSColor.whiteColor())
        if font:
            tv.setFont_(font)
        return tv
