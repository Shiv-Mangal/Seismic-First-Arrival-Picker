"""v22: the scroll wheel changes a spin box only after it was clicked."""
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtTest import QTest


def test_wheel_needs_a_click(qapp):
    from PyQt5.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QDoubleSpinBox
    import wheel_guard as wg
    g = wg.install(qapp)
    g.enabled = True
    sa = QScrollArea()
    inner = QWidget()
    lay = QVBoxLayout(inner)
    sps = [QDoubleSpinBox() for _ in range(40)]
    for sp in sps:
        sp.setRange(0, 1000)
        sp.setValue(80)
        lay.addWidget(sp)
    sa.setWidget(inner)
    sa.resize(300, 300)
    sa.show()
    qapp.processEvents()

    def wheel(w):
        c = w.rect().center()
        ev = QWheelEvent(QPointF(c), QPointF(w.mapToGlobal(c)), QPoint(0, 0), QPoint(0, -120), Qt.NoButton,
                         Qt.NoModifier, Qt.NoScrollPhase, False)
        qapp.sendEvent(w, ev)
        qapp.processEvents()

    bar = sa.verticalScrollBar()
    v0 = bar.value()
    wheel(sps[2])
    assert sps[2].value() == 80 and bar.value() > v0          # passing over: the panel scrolls
    QTest.mouseClick(sps[2], Qt.LeftButton)
    qapp.processEvents()
    wheel(sps[2])
    assert sps[2].value() != 80                               # clicked: the wheel changes it
    sa.close()


def test_progress_bar_turns_red_when_stopped(qapp):
    from PyQt5.QtWidgets import QProgressBar
    try:
        from inversion_window import progress_state
    except ImportError:                     # the inversion window needs pyGIMLi's environment
        import pytest
        pytest.skip('inversion window not importable')
    bar = QProgressBar()
    progress_state(bar, 'running')
    assert bar.property('run_state') == 'running' and 'running' in bar.format()
    progress_state(bar, 'stopped')
    assert bar.property('run_state') == 'stopped' and '#e0483c' in bar.styleSheet()
