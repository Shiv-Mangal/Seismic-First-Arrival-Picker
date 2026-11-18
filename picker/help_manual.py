"""
help_manual.py - the picker's built-in user manual (v13)

The manual is one HTML file (manual/manual.html, with images in manual/images/)
and the same content as a PDF (manual/First_Arrival_Picker_User_Manual.pdf).

  HelpPanel        contents tree + manual + search + back/forward + Open PDF
  make_help_dock() the panel as a dockable side tab of the picker window
                   (Help > User manual, toolbar Help, F1); it can be floated
  open_help()      a separate Help window (used from dialogs, F1)
  install_f1()     F1 in a dialog opens the manual at that dialog's section
  open_pdf()       opens the PDF in the system viewer

Section anchors (manual.html <h2 id=...>):
  workflow, starting, import, canvas, menus, picking, traces-axes-gain,
  preprocessing, filter, delete-picks, autopick, qc, geometry, spectra, files,
  shortcuts, troubleshooting, history

@author: shiv3210
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MANUAL_DIR = os.path.join(HERE, 'manual')
MANUAL_HTML = os.path.join(MANUAL_DIR, 'manual.html')
MANUAL_PDF = os.path.join(MANUAL_DIR, 'First_Arrival_Picker_User_Manual.pdf')

SECTION_TITLES = {
    'workflow': 'Overview and workflow', 'starting': 'Installing and starting',
    'import': 'Importing SEG-Y data', 'canvas': 'The picking canvas', 'menus': 'Menus and toolbar',
    'picking': 'Picking by hand', 'traces-axes-gain': 'Traces, axes and gain',
    'preprocessing': 'Preprocessing', 'filter': 'Filter designer', 'delete-picks': 'Deleting picks and undo',
    'autopick': 'Automatic picking', 'qc': 'Reciprocity and pick QC', 'geometry': 'Survey geometry',
    'spectra': 'Spectra and the sensor response', 'files': 'Files, saving and the SGT export',
    'shortcuts': 'Keyboard and mouse', 'troubleshooting': 'Troubleshooting', 'history': 'Version history',
}


def manual_available():
    return os.path.isfile(MANUAL_HTML)


def read_contents(path=MANUAL_HTML):
    """[(level, anchor, title)] from the <h2 id>/<h3 id> headings of the manual."""
    try:
        html = open(path, encoding='utf-8').read()
    except OSError:
        return []
    out = []
    for m in re.finditer(r'<h([23])[^>]*\bid="([^"]+)"[^>]*>(.*?)</h\1>', html, re.S | re.I):
        title = re.sub(r'<[^>]+>', '', m.group(3))
        title = (title.replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
                 .replace('&nbsp;', ' ').strip())
        out.append((int(m.group(1)), m.group(2), title))
    return out


def fit_images(html, width_px):
    """Give every <img> a pixel width that fits the reading pane (QTextBrowser ignores % widths)."""
    def repl(m):
        tag = m.group(0)
        w = re.search(r'data-w="(\d+)"', tag)
        natural = int(w.group(1)) if w else width_px
        new_w = max(120, min(natural, int(width_px)))
        tag = re.sub(r'\swidth="\d+"', '', tag)
        return tag[:-1].rstrip('/') + f' width="{new_w}">'
    return re.sub(r'<img\b[^>]*>', repl, html)


def open_pdf():
    """Open the PDF manual in the system viewer. Returns False if it is missing."""
    if not os.path.isfile(MANUAL_PDF):
        return False
    try:
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(MANUAL_PDF)))
    except Exception:
        import webbrowser
        return webbrowser.open('file://' + MANUAL_PDF)


try:
    from PyQt5.QtCore import Qt, QUrl, QTimer
    from PyQt5.QtGui import QKeySequence, QTextDocument
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget,
                                 QTreeWidgetItem, QTextBrowser, QLineEdit, QToolButton, QLabel,
                                 QDialog, QDockWidget, QShortcut, QMessageBox, QApplication, QComboBox)
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


if _HAVE_QT:

    class HelpPanel(QWidget):
        """Contents on the left, the manual on the right; search, back/forward, Open PDF."""

        def __init__(self, parent=None, section=None):
            super().__init__(parent)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(4, 4, 4, 4)
            bar = QHBoxLayout()

            def tb(text, tip, slot):
                b = QToolButton()
                b.setText(text)
                b.setToolTip(tip)
                b.clicked.connect(slot)
                bar.addWidget(b)
                return b
            self.b_back = tb('◀', 'Back', lambda: self.browser.backward())
            self.b_fwd = tb('▶', 'Forward', lambda: self.browser.forward())
            tb('⌂', 'Start of the manual', lambda: self.show_section(None))
            self.b_tree = tb('☰', 'Show / hide the contents tree', self._toggle_tree)
            self.b_tree.setCheckable(True)
            self.cb_sec = QComboBox()                        # contents as a list (narrow panel)
            self.cb_sec.setToolTip('Go to a section')
            self.cb_sec.setMinimumContentsLength(14)
            self.cb_sec.activated.connect(lambda i: self.show_section(self.cb_sec.itemData(i)))
            bar.addWidget(self.cb_sec, 1)
            self.ed_find = QLineEdit()
            self.ed_find.setMinimumWidth(120)
            self.ed_find.setPlaceholderText('Search the manual (Enter = next match)')
            self.ed_find.returnPressed.connect(self.find_next)
            bar.addWidget(self.ed_find, 1)
            self._tree_user = None                            # None = automatic by width
            self.lab_find = QLabel('')
            self.lab_find.setStyleSheet('color: #888;')
            bar.addWidget(self.lab_find)
            self.b_pdf = tb('\U0001F4C4 PDF', 'Open the PDF manual (same content, for printing)', self._pdf)
            lay.addLayout(bar)

            split = QSplitter(Qt.Horizontal)
            self.tree = QTreeWidget()
            self.tree.setHeaderHidden(True)
            self.tree.itemClicked.connect(lambda it, _c: self.show_section(it.data(0, Qt.UserRole)))
            self.tree.itemActivated.connect(lambda it, _c: self.show_section(it.data(0, Qt.UserRole)))
            split.addWidget(self.tree)
            self.browser = QTextBrowser()
            self.browser.setOpenExternalLinks(True)
            self.browser.setOpenLinks(False)                 # internal links handled below
            self.browser.anchorClicked.connect(self._link)
            self.browser.setSearchPaths([MANUAL_DIR])
            self.browser.backwardAvailable.connect(self.b_back.setEnabled)
            self.browser.forwardAvailable.connect(self.b_fwd.setEnabled)
            split.addWidget(self.browser)
            split.setStretchFactor(0, 0)
            split.setStretchFactor(1, 1)
            split.setSizes([230, 800])
            lay.addWidget(split, 1)
            self.b_back.setEnabled(False)
            self.b_fwd.setEnabled(False)
            self._html = ''
            self._width = 0
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.setInterval(250)
            self._resize_timer.timeout.connect(self._refit)
            self.load()
            if section:
                QTimer.singleShot(0, lambda: self.show_section(section))

        # ---------------- content ----------------
        def load(self):
            self.tree.clear()
            self.cb_sec.clear()
            if not manual_available():
                self.browser.setHtml(f'<h2>Manual not found</h2><p>Expected <code>{MANUAL_HTML}</code>. '
                                     f'Keep the <code>manual</code> folder next to the picker files.</p>')
                return False
            self._html = open(MANUAL_HTML, encoding='utf-8').read()
            items = {}
            parent = None
            for level, anchor, title in read_contents():
                if level == 2:
                    self.cb_sec.addItem(title, anchor)
                    parent = QTreeWidgetItem([title])
                    parent.setData(0, Qt.UserRole, anchor)
                    self.tree.addTopLevelItem(parent)
                    items[anchor] = parent
                elif parent is not None:
                    it = QTreeWidgetItem([title])
                    it.setData(0, Qt.UserRole, anchor)
                    parent.addChild(it)
                    items[anchor] = it
            self._items = items
            self._render()
            return True

        def _pane_width(self):
            return max(300, self.browser.viewport().width() - 30)

        def _render(self):
            self._width = self._pane_width()
            doc = self.browser.document()
            doc.setBaseUrl(QUrl.fromLocalFile(MANUAL_DIR + os.sep))
            self.browser.setHtml(fit_images(self._html, self._width))

        def _refit(self):
            if abs(self._pane_width() - self._width) < 40 or not self._html:
                return
            pos = self.browser.verticalScrollBar().value()
            frac = pos / max(1, self.browser.verticalScrollBar().maximum())
            self._render()
            sb = self.browser.verticalScrollBar()
            sb.setValue(int(frac * sb.maximum()))

        def resizeEvent(self, ev):
            super().resizeEvent(ev)
            self._auto_tree()
            self._resize_timer.start()

        def _auto_tree(self):
            show = self.width() >= 820 if self._tree_user is None else self._tree_user
            self.tree.setVisible(show)
            self.b_tree.setChecked(show)

        def _toggle_tree(self):
            self._tree_user = not self.tree.isVisible()
            self._auto_tree()
            self._resize_timer.start()

        def show_section(self, anchor):
            """Scroll to a section (None = top) and select it in the contents."""
            if not anchor:
                self.browser.verticalScrollBar().setValue(0)
                return
            self.browser.scrollToAnchor(anchor)
            it = getattr(self, '_items', {}).get(anchor)
            top = anchor
            if it is not None and it.parent() is not None:
                top = it.parent().data(0, Qt.UserRole)
            k = self.cb_sec.findData(top)
            if k >= 0:
                self.cb_sec.setCurrentIndex(k)
            if it is not None:
                self.tree.setCurrentItem(it)
                it.setExpanded(True)

        def _link(self, url):
            if url.scheme() in ('http', 'https', 'mailto'):
                from PyQt5.QtGui import QDesktopServices
                QDesktopServices.openUrl(url)
            elif url.fragment():
                self.show_section(url.fragment())
            elif url.isLocalFile() and url.toLocalFile().lower().endswith('.pdf'):
                self._pdf()

        def find_next(self):
            text = self.ed_find.text().strip()
            if not text:
                return False
            ok = self.browser.find(text)
            if not ok:                                   # wrap around
                cur = self.browser.textCursor()
                cur.movePosition(cur.Start)
                self.browser.setTextCursor(cur)
                ok = self.browser.find(text)
            self.lab_find.setText('' if ok else 'not found')
            return ok

        def _pdf(self):
            if not open_pdf():
                QMessageBox.information(self, 'PDF manual',
                                        f'The PDF manual was not found or could not be opened:\n{MANUAL_PDF}')

    class HelpWindow(QDialog):
        """The manual in its own window (non-modal)."""

        def __init__(self, parent=None, section=None):
            super().__init__(parent)
            self.setWindowTitle('First-arrival picker - User manual')
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint |
                                Qt.WindowMinimizeButtonHint)
            self.setModal(False)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            self.panel = HelpPanel(self, section)
            lay.addWidget(self.panel)
            self.resize(1150, 850)

    _windows = {}

    def open_help(parent=None, section=None):
        """Show the manual in a window (one per parent), at `section`. Returns the window."""
        key = id(parent)
        w = _windows.get(key)
        try:
            if w is not None:
                w.isVisible()
        except RuntimeError:                              # deleted with its parent
            w = None
        if w is None:
            w = HelpWindow(parent, section)
            _windows[key] = w
        elif section:
            w.panel.show_section(section)
        w.show()
        w.raise_()
        w.activateWindow()
        return w

    def install_f1(widget, section=None):
        """F1 in `widget` (a dialog or window) opens the manual at `section`."""
        sc = QShortcut(QKeySequence(Qt.Key_F1), widget)
        sc.setContext(Qt.WindowShortcut)
        sc.activated.connect(lambda: open_help(widget, section))
        widget._help_shortcut = sc                        # keep it alive
        tip = widget.toolTip() if hasattr(widget, 'toolTip') else ''
        if hasattr(widget, 'setToolTip') and not tip:
            widget.setToolTip('F1: help for this window')
        return sc

    def make_help_dock(main_window, section=None):
        """The manual as a dockable side tab of `main_window` (hidden until shown)."""
        dock = QDockWidget('User manual  (F1)', main_window)
        dock.setObjectName('help_dock')
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable)
        dock.panel = HelpPanel(dock, section)
        dock.setWidget(dock.panel)
        dock.setMinimumWidth(460)
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.hide()
        return dock

    def show_in_dock(dock, section=None, width_fraction=0.42):
        """Show the dock (at `section`), about width_fraction of the window wide."""
        dock.show()
        dock.raise_()

        def size_it():
            try:
                mw = dock.parent()
                if not dock.isFloating() and hasattr(mw, 'resizeDocks'):
                    mw.resizeDocks([dock], [int(mw.width() * width_fraction)], Qt.Horizontal)
            except Exception:
                pass
        size_it()
        QTimer.singleShot(0, size_it)
        if section:
            QTimer.singleShot(0, lambda: dock.panel.show_section(section))
        return dock
