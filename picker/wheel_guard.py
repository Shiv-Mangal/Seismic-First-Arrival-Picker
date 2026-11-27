# -*- coding: utf-8 -*-
"""
wheel_guard.py  (v22)
=====================
The scroll wheel / trackpad no longer changes a value just by passing over it (as in FWI
Studio). Spin boxes (cut-offs, transition widths, gains, ...), drop-downs and sliders take
wheel steps only after you click them (or reach them with Tab); otherwise the wheel scrolls
the panel underneath as if the widget were not there. When a panel starts scrolling, a
clicked widget inside it lets go, so momentum scrolling cannot change it either.

The matplotlib canvases (zoom with the wheel, Ctrl+wheel gain) are not affected.

Installed once on the QApplication by the picker; File > "Scroll wheel changes a value only
after clicking it" switches it off and on (remembered).
"""
from PyQt5.QtCore import Qt, QEvent, QObject, QSettings
from PyQt5.QtWidgets import (QApplication, QWidget, QAbstractSpinBox, QComboBox, QAbstractSlider, QScrollBar,
                             QLineEdit, QAbstractScrollArea)

SETTINGS = ('First-arrival picker', 'picker')


class WheelGuard(QObject):
    GUARDED = (QAbstractSpinBox, QComboBox, QAbstractSlider)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = True

    def _guarded(self, obj):
        return isinstance(obj, self.GUARDED) and not isinstance(obj, QScrollBar)

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.Polish and isinstance(obj, QWidget) and self._guarded(obj):
            if obj.focusPolicy() & Qt.WheelFocus == Qt.WheelFocus:
                obj.setFocusPolicy(Qt.StrongFocus)          # the wheel must not give focus
            return False
        if t in (QEvent.FocusIn, QEvent.FocusOut) and isinstance(obj, QWidget):
            w = obj if self._guarded(obj) else obj.parentWidget()
            if w is not None and self._guarded(w):
                # armed only by the user (click, Tab, shortcut), not by the focus a window
                # gives its first widget when it opens
                armed = t == QEvent.FocusIn and ev.reason() in (Qt.MouseFocusReason, Qt.TabFocusReason,
                                                               Qt.BacktabFocusReason, Qt.ShortcutFocusReason)
                w.setProperty('_wheel_armed', armed)
            return False
        if t != QEvent.Wheel or not self.enabled or not isinstance(obj, QWidget):
            return False
        w = obj
        if not self._guarded(w):
            par = w.parentWidget()                          # e.g. the line edit inside a spin box
            if par is not None and self._guarded(par):
                w = par
            else:
                return False
        focused = w.hasFocus() or (w.focusProxy() is not None and w.focusProxy().hasFocus()) \
            or any(c.hasFocus() for c in w.findChildren(QLineEdit))
        if (focused and w.property('_wheel_armed')) or (isinstance(w, QComboBox) and w.view().isVisible()):
            return False                                    # clicked: the wheel changes the value
        area = _scroll_area_of(w)
        if area is not None:
            fw = QApplication.focusWidget()
            if fw is not None and area.isAncestorOf(fw):
                g = fw if self._guarded(fw) else fw.parentWidget()
                if g is not None and self._guarded(g) and g.property('_wheel_armed'):
                    g.setProperty('_wheel_armed', False)    # scrolling the panel lets go of it
            QApplication.sendEvent(area.viewport(), ev)
        return True                                         # never reaches the value widget


def _scroll_area_of(w):
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QAbstractScrollArea) and not isinstance(p, QComboBox):
            return p
        p = p.parentWidget()
    return None


def install(app=None):
    """Install once on the application; returns the guard."""
    import sys
    app = app or QApplication.instance() or QApplication(sys.argv[:1])
    g = getattr(app, '_wheel_guard', None)
    if g is None:
        g = WheelGuard(app)
        app.installEventFilter(g)
        app._wheel_guard = g
        try:
            g.enabled = QSettings(*SETTINGS).value('wheel_guard', True, type=bool)
        except Exception:
            pass
    return g


def set_enabled(on):
    g = install()
    g.enabled = bool(on)
    try:
        QSettings(*SETTINGS).setValue('wheel_guard', bool(on))
    except Exception:
        pass
    return g
