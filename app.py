"""
Weight·Expression Lab — PyQt6 GUI for the "same recipe, different weights" experiment.

    python app.py

Shows the saved results (results.json) on start; "실험 실행" re-runs the whole
experiment on this PC (CPU, ~15 s) and adds the weight-level visualizations.
"""
import json, sys, traceback
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib import font_manager
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Rectangle
from matplotlib.ticker import FuncFormatter
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QFileDialog, QFrame, QGraphicsDropShadowEffect, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import core

ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "results.json"

# ---- palette ---------------------------------------------------------------
INK = "#0b1220"
INK_2 = "#52606d"
MUTED = "#8a96a3"
LINE = "#e3e8ef"
SURFACE = "#ffffff"
APP_BG = "#f4f6fa"
BLUE = "#2a78d6"
BLUE_DARK = "#1f62b3"
ORANGE = "#eb6834"
GOOD = "#1a7f37"
SIDEBAR = "#0f1b2d"

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc3ef", "#2a78d6", "#0d3b78"])
DIV = LinearSegmentedColormap.from_list("div", ["#b8452a", "#f3b8a3", "#f2f2f0", "#9cc3ef", "#1f5fae"])

KOREAN_FONTS = ["Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic",
                "Noto Sans KR", "Noto Sans CJK KR", "NanumGothic"]


def setup_matplotlib():
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in KOREAN_FONTS if f in available][:1] + ["DejaVu Sans"]
    matplotlib.rcParams.update({
        "font.family": family, "axes.unicode_minus": False, "font.size": 9.5,
        "axes.edgecolor": LINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": False, "grid.color": "#edf0f4", "grid.linewidth": 0.8,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def pick_qt_font():
    families = set(QFontDatabase.families())
    for f in KOREAN_FONTS:
        if f in families:
            return f
    return QApplication.font().family()


STYLE = f"""
QWidget {{ color: {INK}; }}
QMainWindow, #content {{ background: {APP_BG}; }}
QScrollArea, #page {{ background: {APP_BG}; border: none; }}
#sidebar {{ background: {SIDEBAR}; }}
#brand {{ color: white; font-size: 17px; font-weight: 700; }}
#brandSub {{ color: #8fa3bf; font-size: 11px; }}
#navButton {{
    color: #c8d3e3; background: transparent; border: none; border-radius: 8px;
    text-align: left; padding: 10px 14px; font-size: 13px;
}}
#navButton:hover {{ background: #18294a; color: white; }}
#navButton:checked {{ background: #1e3a66; color: white; font-weight: 600; border-left: 3px solid #5ea2f0; }}
#sideInfo {{ color: #8fa3bf; font-size: 11px; }}
#header {{ background: {SURFACE}; border-bottom: 1px solid {LINE}; }}
#pageTitle {{ font-size: 20px; font-weight: 700; }}
#pageSub {{ color: {INK_2}; font-size: 12.5px; }}
#badge {{ border-radius: 10px; padding: 3px 10px; font-size: 11.5px; font-weight: 600; }}
#badge[source="saved"] {{ background: #fff4e5; color: #9a5b00; }}
#badge[source="live"] {{ background: #e7f6ec; color: {GOOD}; }}
QPushButton#primary {{
    background: {BLUE}; color: white; border: none; border-radius: 8px;
    padding: 9px 18px; font-weight: 600; font-size: 13px;
}}
QPushButton#primary:hover {{ background: {BLUE_DARK}; }}
QPushButton#primary:disabled {{ background: #a9c4e6; }}
QPushButton#secondary {{
    background: {SURFACE}; color: {INK}; border: 1px solid #cfd7e2; border-radius: 8px;
    padding: 8px 14px; font-size: 12.5px;
}}
QPushButton#secondary:hover {{ border-color: {BLUE}; color: {BLUE}; }}
QPushButton#secondary:disabled {{ color: {MUTED}; border-color: {LINE}; }}
QProgressBar {{ background: #e8edf3; border: none; border-radius: 4px; height: 8px; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 4px; }}
#status {{ color: {INK_2}; font-size: 11.5px; }}
#card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px; }}
#cardTitle {{ font-size: 14px; font-weight: 700; }}
#cardText {{ color: {INK_2}; font-size: 12.5px; }}
#kpiValue {{ font-size: 26px; font-weight: 700; }}
#kpiLabel {{ color: {INK_2}; font-size: 12px; }}
#kpiNote {{ color: {MUTED}; font-size: 11px; }}
#legal {{ background: #eef5ff; border: none; border-left: 4px solid {BLUE}; border-radius: 10px; }}
#legalTitle {{ color: {BLUE_DARK}; font-size: 12px; font-weight: 700; }}
#legalText {{ color: #1d2b3a; font-size: 13px; }}
#heroTitle {{ font-size: 22px; font-weight: 700; }}
#heroText {{ color: {INK_2}; font-size: 13.5px; }}
#step {{ background: #f6f9fd; border: 1px solid {LINE}; border-radius: 10px; }}
#stepNum {{ color: {BLUE}; font-size: 12px; font-weight: 700; }}
#stepText {{ font-size: 13px; }}
#expCard {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px; }}
#expCard:hover {{ border: 1px solid {BLUE}; }}
#expTag {{ color: {BLUE}; font-weight: 700; font-size: 12px; }}
#placeholder {{ color: {INK_2}; font-size: 13px; }}
#fieldLabel {{ color: {INK_2}; font-size: 12px; }}
QTableWidget {{
    background: {SURFACE}; border: none; gridline-color: {LINE}; font-size: 12.5px;
    selection-background-color: #e5effb; selection-color: {INK};
}}
QTableWidget::item {{ padding: 8px 6px; }}
QHeaderView::section {{
    background: #f6f8fb; color: {INK_2}; border: none; border-bottom: 1px solid {LINE};
    padding: 8px; font-weight: 600; font-size: 12px;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c9d2de; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{ background: {INK}; color: white; border: none; padding: 6px; }}
"""


# ---- small widgets ---------------------------------------------------------
def label(text, name=None, wrap=True):
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    lb.setTextFormat(Qt.TextFormat.RichText)
    return lb


def shadow(widget):
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(18)
    eff.setOffset(0, 2)
    eff.setColor(QColor(15, 27, 45, 18))
    widget.setGraphicsEffect(eff)


class Card(QFrame):
    def __init__(self, title=None, subtitle=None, name="card"):
        super().__init__()
        self.setObjectName(name)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(20, 18, 20, 18)
        self.lay.setSpacing(10)
        if title:
            self.lay.addWidget(label(title, "cardTitle"))
        if subtitle:
            self.lay.addWidget(label(subtitle, "cardText"))
        shadow(self)


class Kpi(Card):
    def __init__(self, caption, note=""):
        super().__init__()
        self.lay.setSpacing(2)
        self.cap = label(caption, "kpiLabel")
        self.value = label("—", "kpiValue")
        self.note = label(note, "kpiNote")
        for w in (self.cap, self.value, self.note):
            self.lay.addWidget(w)

    def set(self, value, note=None, color=INK):
        self.value.setText(value)
        self.value.setStyleSheet(f"color: {color};")
        if note is not None:
            self.note.setText(note)


class Legal(QFrame):
    def __init__(self, text, title="결과 정리"):
        super().__init__()
        self.setObjectName("legal")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.addWidget(label(title, "legalTitle"))
        lay.addWidget(label(text, "legalText"))


def new_table(headers, nrows, stretch):
    """Read-only table whose rows grow with their text; stretch[i] says whether column i takes spare width."""
    t = QTableWidget(nrows, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setWordWrap(True)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    for i, s in enumerate(stretch):
        t.horizontalHeader().setSectionResizeMode(
            i, QHeaderView.ResizeMode.Stretch if s else QHeaderView.ResizeMode.ResizeToContents)
    t.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
    return t


def fill_table(t, rows):
    for r, row in enumerate(rows):
        for col, text in enumerate(row):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            if col == 0:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            t.setItem(r, col, item)


# (citation, what the work reports · which experiment it relates to)
REFERENCES = [
    ("Hecht-Nielsen, R. (1990). On the algebraic structure of feedforward network weight spaces. "
     "In R. Eckmiller (Ed.), <i>Advanced Neural Computers</i> (pp. 129–135). North-Holland.",
     "피드포워드 신경망 가중치 공간의 대칭성(은닉 유닛 순서 교환 등 기능을 바꾸지 않는 변환)을 다룸 · 관련: 실험 C"),
    ("Chen, A. M., Lu, H., &amp; Hecht-Nielsen, R. (1993). On the geometry of feedforward neural network error "
     "surfaces. <i>Neural Computation</i>, 5(6), 910–927.",
     "기능을 바꾸지 않는 가중치 변환이 군(group)을 이루며, 하나의 최적 가중치에 대해 기능이 같은 사본이 많이 존재함을 보임 · 관련: 실험 C"),
    ("Sussmann, H. J. (1992). Uniqueness of the weights for minimal feedforward nets with a given input-output map. "
     "<i>Neural Networks</i>, 5(4), 589–593.",
     "tanh 신경망에서, 입출력 기능이 같은 최소 신경망의 가중치는 순서 교환·부호 반전을 제외하면 유일함을 증명 · "
     "관련: 실험 B"),
    ("Dinh, L., Pascanu, R., Bengio, S., &amp; Bengio, Y. (2017). Sharp minima can generalize for deep nets. "
     "<i>Proceedings of ICML 2017</i>, PMLR 70, 1019–1028.",
     "ReLU 신경망에서 양수 배율 조정(한쪽에 α, 다른 쪽에 1/α)으로 기능이 같은 가중치를 만들 수 있음을 보임 · 관련: 실험 B·C"),
    ("Li, Y., Yosinski, J., Clune, J., Lipson, H., &amp; Hopcroft, J. (2016). Convergent learning: Do different "
     "neural networks learn the same representations? <i>ICLR 2016</i>.",
     "초기값만 다른 신경망들의 유닛을 이분 매칭 등으로 짝지어 비교, 일부 특징은 여러 신경망에서 공통으로 학습되고 일부는 그렇지 않음을 보고 · 관련: 실험 B"),
    ("Fort, S., Hu, H., &amp; Lakshminarayanan, B. (2019). Deep ensembles: A loss landscape perspective. "
     "arXiv:1912.02757.",
     "무작위 초기값이 다르면 서로 다른 해(mode)에 도달함을 보고, 가중치 비교에 코사인 유사도를 사용 · 관련: 실험 B"),
    ("Entezari, R., Sedghi, H., Saukh, O., &amp; Neyshabur, B. (2022). The role of permutation invariance in linear "
     "mode connectivity of neural networks. <i>ICLR 2022</i>.",
     "순서 대칭성을 고려하면 독립적으로 학습된 해들 사이의 선형 보간 장벽이 사라질 가능성이 높다는 추측을 제시하고 실험으로 검토 · 관련: 실험 B·C"),
    ("Ainsworth, S. K., Hayase, J., &amp; Srinivasa, S. (2023). Git Re-Basin: Merging models modulo permutation "
     "symmetries. <i>ICLR 2023</i>.",
     "한 모델의 은닉 유닛 순서를 기준 모델에 맞추는 정렬 알고리즘을 제시, 독립적으로 학습된 ResNet(CIFAR-10)에 적용 · 관련: 실험 B·C"),
]


def references_html(indices=None):
    items = [REFERENCES[i] for i in (indices if indices is not None else range(len(REFERENCES)))]
    return "<br><br>".join(f"{cite}<br><span style='color:{BLUE_DARK}'>· {use}</span>" for cite, use in items)


class Chart(FigureCanvasQTAgg):
    def __init__(self, w, h, name):
        self.fig = Figure(figsize=(w, h), dpi=100, layout="constrained")
        super().__init__(self.fig)
        self.name = name
        self.setMinimumHeight(int(h * 100))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class ChartSlot(QStackedWidget):
    """A chart that shows a 'run the experiment first' message until data exists."""

    def __init__(self, chart, message, run_cb):
        super().__init__()
        self.chart = chart
        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg = label(message, "placeholder", wrap=False)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn = QPushButton("▶  실험 실행 (약 15초)")
        btn.setObjectName("primary")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(run_cb)
        v.addWidget(msg)
        v.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        holder.setMinimumHeight(chart.minimumHeight())
        self.addWidget(holder)
        self.addWidget(chart)

    def show_chart(self):
        self.setCurrentIndex(1)
        self.chart.draw_idle()


def pct(v, digits=1):
    """Percent with half-up rounding, so 0.1185 reads 11.9% as in the paper (f"{v:.1%}" gives 11.8%)."""
    q = (Decimal(str(v)) * 100).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{q}%"


def lr_str(lr):
    return np.format_float_positional(lr, trim="-")


def annotate_heatmap(ax, M, fmt, cmap_norm_vals):
    for (i, j), v in np.ndenumerate(M):
        dark = cmap_norm_vals[i, j] > 0.55
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9.5,
                color="white" if dark else INK, fontweight="bold" if dark else "normal")


def page_scroll(widget):
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.Shape.NoFrame)
    sc.setWidget(widget)
    return sc


def new_page():
    w = QWidget()
    w.setObjectName("page")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(28, 24, 28, 28)
    lay.setSpacing(16)
    return w, lay


SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


# ---- explanatory illustrations (static, no experiment run needed) ------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _arrow(ax, start, end, color, lw=2.2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14, shrinkA=0, shrinkB=0))


def draw_cosine_algorithm(fig):
    """Three-step picture of how the cosine similarity of two weight files is measured."""
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3, width_ratios=[1.3, 1, 1.05])
    for ax in (ax1, ax2, ax3):
        ax.set_axis_off()

    # 1) flatten each weight file into one long row of numbers
    ax1.set_title("① 가중치 파일을 숫자 한 줄로 펼친다")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    rng = np.random.default_rng(7)
    for y0, name, vec in [(6.0, "모델 A", "a"), (1.4, "모델 B", "b")]:
        vals = rng.normal(size=(3, 4))
        ax1.text(0.2, y0 + 2.05, name, fontsize=9.5, fontweight="bold", color=INK)
        for r in range(3):
            for c in range(4):
                ax1.add_patch(Rectangle((0.2 + c * 0.5, y0 + 1.1 - r * 0.55), 0.48, 0.52,
                                        color=DIV((vals[r, c] + 2.5) / 5), lw=0))
        _arrow(ax1, (2.5, y0 + 0.82), (3.4, y0 + 0.82), INK_2, lw=1.5)
        for k, v in enumerate(vals.ravel()):
            ax1.add_patch(Rectangle((3.6 + k * 0.45, y0 + 0.55), 0.43, 0.55, color=DIV((v + 2.5) / 5), lw=0))
        ax1.text(9.1, y0 + 0.82, "…", fontsize=12, va="center", color=INK_2)
        ax1.text(3.6, y0 + 0.05, f"{vec} = ({vec}₁, {vec}₂, …, {vec}₉₆₁₀)", fontsize=8.5, color=INK_2)
    ax1.text(0.2, 0.15, "행렬 W₁(64×128), W₂(128×10)과 편향을 정해진 순서로 이어 붙임",
             fontsize=8, color=MUTED)

    # 2) treat each row as an arrow and measure the angle between them
    ax2.set_title("② 두 줄을 화살표로 보고 사이 각도를 잰다")
    ax2.set_xlim(-0.15, 1.25)
    ax2.set_ylim(-0.75, 1.05)
    ax2.set_aspect("equal")
    ta, tb = np.deg2rad(18), np.deg2rad(62)
    _arrow(ax2, (0, 0), (np.cos(ta), np.sin(ta)), BLUE)
    _arrow(ax2, (0, 0), (0.85 * np.cos(tb), 0.85 * np.sin(tb)), ORANGE)
    ax2.text(1.02 * np.cos(ta), 1.02 * np.sin(ta) + 0.05, "a", color=BLUE, fontsize=11, fontweight="bold")
    ax2.text(0.88 * np.cos(tb) + 0.03, 0.88 * np.sin(tb), "b", color=ORANGE, fontsize=11, fontweight="bold")
    ax2.add_patch(Arc((0, 0), 0.6, 0.6, theta1=18, theta2=62, color=INK_2, lw=1.2))
    ax2.text(0.36 * np.cos(np.deg2rad(40)), 0.36 * np.sin(np.deg2rad(40)), "θ", fontsize=11, color=INK)
    ax2.text(0.55, -0.3, "cos θ = (a₁b₁ + a₂b₂ + … + a₉₆₁₀b₉₆₁₀) ÷ (|a| × |b|)",
             ha="center", fontsize=8.6, color=INK)
    ax2.text(0.55, -0.55, "같은 자리 숫자끼리 곱해 모두 더하고, 두 화살표의 길이로 나눈다",
             ha="center", fontsize=7.8, color=INK_2)

    # 3) how to read the value
    ax3.set_title("③ 값 읽는 법")
    ax3.set_xlim(0, 3.3)
    ax3.set_ylim(-1.0, 1.05)
    cases = [(0.55, 30, 30, "1", "같은 패턴"), (1.65, 0, 90, "0", "서로 무관"), (2.75, 0, 180, "−1", "정반대")]
    for cx, a_deg, b_deg, val, desc in cases:
        base = (cx - 0.3, 0.05) if b_deg != 180 else (cx, 0.05)
        la, lb = (0.75, 0.5) if a_deg == b_deg else (0.5, 0.5)
        _arrow(ax3, base, (base[0] + la * np.cos(np.deg2rad(a_deg)), base[1] + la * np.sin(np.deg2rad(a_deg))), BLUE)
        _arrow(ax3, base, (base[0] + lb * np.cos(np.deg2rad(b_deg)), base[1] + lb * np.sin(np.deg2rad(b_deg))), ORANGE)
        ax3.text(cx, -0.42, val, ha="center", fontsize=17, fontweight="bold", color=INK)
        ax3.text(cx, -0.72, desc, ha="center", fontsize=9, color=INK_2)
    ax3.text(1.65, -0.98, "원래 순서의 0.03은 0에 가깝지만, 순서 차이만으로도 생기는 값(실험 C)", ha="center", fontsize=7.8, color=MUTED)


def draw_cosine_toy(fig):
    """Six numbers: scaling keeps cosine at 1, reordering drops it to 0, undoing the order restores 1."""
    fig.clear()
    a = np.array([3.0, 1.0, -2.0, 0.0, 2.0, -1.0])
    perm = np.array([1, 5, 4, 0, 3, 2])          # c = a[perm] = [1, -1, 2, 3, 0, -2]
    panels = [("원본 a", a, np.arange(6), None),
              ("b = a × 2 (크기만 다름)", 2 * a, np.arange(6), 2 * a),
              ("c = a의 순서만 섞음", a[perm], perm, a[perm]),
              ("c의 순서를 a에 맞춤", a[perm][np.argsort(perm)], np.arange(6), a[perm][np.argsort(perm)])]
    axes = fig.subplots(1, 4, sharey=True)
    for ax, (title, v, src, other) in zip(axes, panels):
        x = np.arange(1, 7)
        ax.bar(x, v, width=0.68, color=[SERIES[i] for i in src], zorder=2)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + (0.35 if vi >= 0 else -0.35), f"{vi:g}", ha="center",
                    va="bottom" if vi >= 0 else "top", fontsize=8, color=INK_2)
        ax.axhline(0, color=LINE, lw=1, zorder=1)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks(x, [str(i) for i in x])
        ax.tick_params(length=0, labelsize=8)
        ax.set_ylim(-5.6, 7.8)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        verdict = "기준" if other is None else f"a와의 코사인 = {core.cos(a, other):.2f}"
        ax.set_xlabel(verdict, fontsize=9.5, color=INK, fontweight="bold")


# ---- worker thread ---------------------------------------------------------
class Worker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.finished.emit(self.fn(*self.args, progress=self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---- pages -----------------------------------------------------------------
class OverviewPage(QWidget):
    def __init__(self, go):
        super().__init__()
        page, lay = new_page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page_scroll(page))

        hero = Card()
        hero.lay.setSpacing(8)
        hero.lay.addWidget(label("같은 설정으로 학습한 신경망은 같은 가중치를 갖는가?", "heroTitle"))
        hero.lay.addWidget(label(
            "작은 신경망을 처음부터 학습하여 학습 설정, 예측(기능), 가중치 수치 사이의 관계를 측정합니다. "
            "모든 데이터와 코드가 공개되어 있어 같은 결과를 재현할 수 있습니다.", "heroText"))
        steps = QHBoxLayout()
        steps.setSpacing(12)
        for num, text in [("실험 A", "학습 설정(은닉 유닛 수·학습률)을 바꾸면 평가 정확도는 얼마나 달라지는가?"),
                          ("실험 B", "학습 설정을 같게 두고 seed만 바꾸면 예측과 가중치는 얼마나 같은가?"),
                          ("실험 C", "학습된 모델의 은닉 유닛 순서를 바꾸면 예측과 가중치는 어떻게 되는가?")]:
            f = QFrame()
            f.setObjectName("step")
            v = QVBoxLayout(f)
            v.setContentsMargins(14, 10, 14, 10)
            v.addWidget(label(num, "stepNum"))
            v.addWidget(label(text, "stepText"))
            steps.addWidget(f)
        hero.lay.addLayout(steps)
        lay.addWidget(hero)

        grid = QHBoxLayout()
        grid.setSpacing(16)
        self.exp_values = {}
        specs = [("A", "학습 설정을 바꾼 9개 모델", "seed 고정, 은닉 유닛 × 학습률 9가지 조합", 1),
                 ("B", "같은 설정, 다른 seed", "학습 설정을 같게 두고 seed만 0~4", 2),
                 ("C", "은닉 유닛 순서 교환 (시연)", "학습된 모델의 은닉 유닛 순서만 바꿈", 3)]
        for tag, title, desc, idx in specs:
            card = QFrame()
            card.setObjectName("expCard")
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.mousePressEvent = lambda e, i=idx: go(i)
            shadow(card)
            v = QVBoxLayout(card)
            v.setContentsMargins(20, 16, 20, 16)
            v.setSpacing(4)
            v.addWidget(label(f"실험 {tag}", "expTag"))
            v.addWidget(label(title, "cardTitle"))
            v.addWidget(label(desc, "cardText"))
            v.addSpacing(8)
            val = label("—", "kpiValue")
            note = label("", "kpiNote")
            v.addWidget(val)
            v.addWidget(note)
            v.addStretch(1)
            v.addWidget(label("자세히 보기 →", "expTag"))
            self.exp_values[tag] = (val, note)
            grid.addWidget(card)
        lay.addLayout(grid)

        data_card = Card("데이터와 모델",
                         "scikit-learn <b>digits</b> — 8×8 손글씨 숫자 1,797장, 10개 클래스(패키지 내장, 다운로드 불필요). "
                         "학습 70% / 평가 30%로 고정 분할. 모델은 은닉층 하나짜리 신경망 <b>MLP 64-128-10</b>을 처음부터 학습합니다. "
                         "GPU 없이 CPU에서 약 10~15초면 전체 실험이 끝납니다.")
        self.digits = Chart(10, 1.15, "digits_samples")
        data_card.lay.addWidget(self.digits)
        lay.addWidget(data_card)
        lay.addStretch(1)
        self.draw_digits()

    def draw_digits(self):
        from sklearn.datasets import load_digits
        d = load_digits()
        fig = self.digits.fig
        fig.clear()
        axes = fig.subplots(1, 14)
        for k, ax in enumerate(axes):
            idx = int(np.where(d.target == k % 10)[0][k // 10])
            ax.imshow(d.images[idx], cmap="Greys", interpolation="nearest")
            ax.set_title(str(d.target[idx]), fontsize=9, loc="center", fontweight="normal", color=INK_2)
            ax.set_axis_off()
        self.digits.draw_idle()

    def update_results(self, out, details):
        accs = [r["test_acc"] for r in out["A_human_choices"]]
        b, c = out["B_seed"], out["C_permutation"]
        self.exp_values["A"][0].setText(f"{pct(min(accs))} ~ {pct(max(accs))}")
        self.exp_values["A"][1].setText("조합에 따른 평가 정확도 범위")
        self.exp_values["B"][0].setText(
            f"{pct(b['prediction_agreement_mean'])} vs {b['weight_cosine_after_alignment_and_rescaling_mean']:.2f}")
        self.exp_values["B"][1].setText("예측 일치율 vs 가중치 코사인 유사도(순서·배율을 맞춘 뒤)")
        exp = c["number_of_equivalent_arrays"].split("10^")[-1]
        self.exp_values["C"][0].setText(f"10<sup>{exp}</sup>개")
        self.exp_values["C"][1].setText(
            f"출력이 같은 배열 수의 하한(128!) · 예측 {'100% 동일' if c['prediction_identical'] else '불일치'} · 코사인 "
            f"{c['weight_cosine_original_vs_permuted']:.2f}")


class PageA(QWidget):
    def __init__(self, train_cb):
        super().__init__()
        page, lay = new_page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page_scroll(page))

        kpis = QHBoxLayout()
        kpis.setSpacing(16)
        self.k_min, self.k_max, self.k_span = Kpi("최저 정확도"), Kpi("최고 정확도"), Kpi("차이")
        for k in (self.k_min, self.k_max, self.k_span):
            kpis.addWidget(k)
        lay.addLayout(kpis)

        row = QHBoxLayout()
        row.setSpacing(16)
        heat_card = Card("은닉 유닛 × 학습률에 따른 평가 정확도",
                         "seed 0 · 50 epochs 고정. 사람이 고르는 두 가지 값만 바꿨습니다.")
        self.heat = Chart(5.2, 3.6, "A_accuracy_heatmap")
        heat_card.lay.addWidget(self.heat)
        heat_card.lay.addStretch(1)
        row.addWidget(heat_card, 3)

        try_card = Card("직접 해보기", "원하는 값을 골라 모델 하나를 학습해 보세요. 결과는 몇 초 안에 나옵니다.")
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self.sp_hidden = QComboBox()
        self.sp_hidden.addItems(["2", "4", "8", "16", "32", "64", "128", "256"])
        self.sp_hidden.setCurrentText("16")
        self.cb_lr = QComboBox()
        self.cb_lr.addItems(["0.1", "0.01", "0.001", "0.0001", "0.00001"])
        self.cb_lr.setCurrentText("0.001")
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 9999)
        self.sp_epochs = QSpinBox()
        self.sp_epochs.setRange(5, 300)
        self.sp_epochs.setSingleStep(5)
        self.sp_epochs.setValue(50)
        for r, (name, w) in enumerate([("은닉 유닛 수", self.sp_hidden), ("학습률", self.cb_lr),
                                       ("seed", self.sp_seed), ("epochs", self.sp_epochs)]):
            w.setMinimumHeight(30)
            form.addWidget(label(name, "fieldLabel"), r, 0)
            form.addWidget(w, r, 1)
        try_card.lay.addLayout(form)
        self.btn_train = QPushButton("학습하기")
        self.btn_train.setObjectName("primary")
        self.btn_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_train.clicked.connect(lambda: train_cb(int(self.sp_hidden.currentText()),
                                                        float(self.cb_lr.currentText()),
                                                        self.sp_seed.value(), self.sp_epochs.value()))
        try_card.lay.addWidget(self.btn_train)
        self.try_result = label("<span style='color:#8a96a3'>아직 학습하지 않았습니다.</span>", "cardText")
        try_card.lay.addWidget(self.try_result)
        self.loss = Chart(3.2, 1.7, "A_custom_loss_curve")
        try_card.lay.addWidget(self.loss)
        self.loss.hide()
        try_card.lay.addStretch(1)
        row.addWidget(try_card, 2)
        lay.addLayout(row)

        lay.addWidget(Legal(
            "데이터·seed·학습량(50 epochs)을 같게 두고 <b>은닉 유닛 수와 학습률</b>만 바꿨을 때, 평가 정확도는 "
            "11.9%에서 97.8%까지 달라졌습니다. 10개 클래스 문제이므로 무작위로 답할 때의 기대 정확도는 10%입니다."))
        lay.addStretch(1)

    def update_results(self, out, details):
        A = out["A_human_choices"]
        M = np.array([[next(r["test_acc"] for r in A if r["hidden_units"] == h and r["lr"] == lr)
                       for lr in core.A_LR] for h in core.A_HIDDEN])
        lo, hi = np.unravel_index(M.argmin(), M.shape), np.unravel_index(M.argmax(), M.shape)
        self.k_min.set(f"{pct(M.min())}", f"은닉 {core.A_HIDDEN[lo[0]]} · 학습률 {lr_str(core.A_LR[lo[1]])}")
        self.k_max.set(f"{pct(M.max())}", f"은닉 {core.A_HIDDEN[hi[0]]} · 학습률 {lr_str(core.A_LR[hi[1]])}", BLUE)
        self.k_span.set(f"{(M.max() - M.min()) * 100:.1f}%p", "최고와 최저의 차이")
        fig = self.heat.fig
        fig.clear()
        ax = fig.add_subplot()
        im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        annotate_heatmap(ax, M, lambda v: f"{pct(v)}", M)
        ax.set_xticks(range(3), [lr_str(lr) for lr in core.A_LR])
        ax.set_yticks(range(3), [str(h) for h in core.A_HIDDEN])
        ax.set_xlabel("학습률 (learning rate)")
        ax.set_ylabel("은닉 유닛 수")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, format=FuncFormatter(lambda v, _: f"{v:.0%}"))
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8.5)
        self.heat.draw_idle()

    def set_training(self, busy):
        self.btn_train.setEnabled(not busy)
        self.btn_train.setText("학습 중…" if busy else "학습하기")

    def show_custom(self, r):
        self.try_result.setText(
            f"은닉 {r['hidden']} · 학습률 {lr_str(r['lr'])} · seed {r['seed']} · {r['epochs']} epochs<br>"
            f"<span style='font-size:22px; font-weight:700; color:{INK}'>{pct(r['test_acc'])}</span>"
            f" <span style='color:{MUTED}'>평가 정확도</span>")
        fig = self.loss.fig
        fig.clear()
        ax = fig.add_subplot()
        ax.plot(range(1, len(r["loss_curve"]) + 1), r["loss_curve"], color=BLUE, lw=2)
        ax.set_title("학습 손실", fontsize=9.5)
        ax.set_xlabel("epoch", fontsize=8.5)
        ax.tick_params(labelsize=8, length=0)
        ax.grid(axis="y")
        self.loss.show()
        self.loss.draw_idle()


class PageB(QWidget):
    def __init__(self, run_cb):
        super().__init__()
        page, lay = new_page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page_scroll(page))

        kpis = QHBoxLayout()
        kpis.setSpacing(12)
        self.k_acc = Kpi("평가 정확도 (5개 seed)")
        self.k_agree = Kpi("예측 일치율", "모델 쌍 10개 평균")
        self.k_rescaled = Kpi("코사인 · 순서·배율을 맞춘 뒤")
        self.k_align = Kpi("코사인 · 순서를 맞춘 뒤")
        self.k_cos = Kpi("코사인 · 원래 순서")
        for k in (self.k_acc, self.k_agree, self.k_rescaled, self.k_align, self.k_cos):
            kpis.addWidget(k)
        lay.addLayout(kpis)

        why = Card("코사인 유사도로 재는 이유",
                   "학습된 모델의 가중치는 <b>숫자 9,610개</b>입니다(64×128 + 128×10 + 편향 128 + 10). 두 모델의 가중치가 "
                   "같은지를 하나의 값으로 비교하기 위해 코사인 유사도를 씁니다. 코사인 유사도는 두 숫자 목록을 각각 벡터로 보고 "
                   "두 벡터 사이 각도의 코사인을 계산합니다. 값은 −1에서 1 사이이며, 한쪽 목록 전체에 같은 양수를 곱해도 값이 "
                   "변하지 않습니다. 신경망 가중치 비교에 코사인 유사도를 쓴 선행 연구로 Fort et al.(2019)이 있습니다.")
        self.cos_algo = Chart(10.5, 3.5, "B_cosine_algorithm")
        why.lay.addWidget(self.cos_algo)
        lay.addWidget(why)

        toy = Card("숫자 6개 예시: 순서만 바꾼 경우",
                   "막대 색은 ‘어떤 숫자인가’를, 가로 위치는 ‘몇 번째 자리인가’를 나타냅니다. 같은 숫자들을 자리만 바꾸면 "
                   "이 예에서 코사인은 1에서 0으로 떨어지고, 순서를 되돌리면 다시 1이 됩니다.")
        self.cos_toy = Chart(10.5, 2.9, "B_cosine_toy_example")
        toy.lay.addWidget(self.cos_toy)
        lay.addWidget(toy)
        draw_cosine_algorithm(self.cos_algo.fig)
        draw_cosine_toy(self.cos_toy.fig)

        three = Card("실험 B가 코사인을 세 가지로 보고하는 이유",
                     "ReLU 신경망에는 기능(출력)을 바꾸지 않고 가중치 수치만 바꾸는 변환이 두 가지 있습니다. 원래 순서로 잰 "
                     "코사인에는 이 두 가지 차이가 함께 들어 있으므로, 이를 하나씩 제거한 값을 함께 보고합니다.")
        three.lay.addWidget(label(
            "<b>• 은닉 유닛의 순서 교환</b> — 유닛의 순서를 바꾸고 입력·출력 가중치를 함께 옮기면 출력이 같습니다(실험 C; "
            "Hecht-Nielsen, 1990).<br>"
            "<b>• 유닛별 배율 조정</b> — 한 유닛의 입력 가중치와 편향에 양수 a를 곱하고 출력 가중치를 a로 나누면, "
            "ReLU(a·z) = a·ReLU(z)이므로 출력이 같습니다(Dinh et al., 2017).<br><br>"
            "<b>원래 순서</b>: 두 가중치 목록을 그대로 비교합니다.<br>"
            "<b>순서를 맞춘 뒤</b>: 헝가리안 알고리즘으로 두 모델의 은닉 유닛 128개를 가장 비슷한 것끼리 1:1로 짝지은 뒤 "
            "비교합니다. 유닛 짝짓기는 Li et al.(2016), Ainsworth et al.(2023)에서 쓰인 방식과 같은 계열입니다.<br>"
            "<b>순서·배율을 맞춘 뒤</b>: 위 짝짓기에 더해, 각 유닛의 입력 가중치와 편향의 크기를 1로 표준화하고 출력 가중치를 "
            "반대로 조정한 뒤 비교합니다. 이 표준화로 출력은 바뀌지 않습니다(출력값 차이 10<sup>−14</sup> 수준).<br><br>"
            "참고로 tanh 신경망에서는, 기능이 같은 최소 신경망의 가중치가 이러한 대칭 변환을 제외하면 유일하다는 것이 증명되어 "
            "있습니다(Sussmann, 1992). 이 실험의 모델은 ReLU 신경망이므로 순서와 배율 대칭을 모두 제거한 뒤 비교했습니다.",
            "cardText"))
        lay.addWidget(three)

        mat_card = Card("모델 쌍별 비교",
                        "seed 0~4로 학습한 다섯 모델을 짝지어 비교했습니다. 왼쪽은 평가 데이터에 대한 예측 일치율, "
                        "나머지 셋은 가중치 코사인 유사도입니다.")
        self.mats = Chart(10.5, 3.0, "B_pairwise_matrices")
        self.mat_slot = ChartSlot(self.mats, "모델 쌍별 행렬은 이 PC에서 실험을 실행하면 표시됩니다.", run_cb)
        mat_card.lay.addWidget(self.mat_slot)
        lay.addWidget(mat_card)

        rf_card = Card("은닉 유닛의 입력 가중치 (8×8로 표시)",
                       "seed 0과 seed 1 모델의 은닉 유닛 12개. 가운데 줄은 seed 1의 같은 순번 유닛, 아래 줄은 순서를 맞춘 뒤 "
                       "seed 0의 각 유닛에 대응하는 seed 1의 유닛입니다. 빨강 = 음수, 파랑 = 양수.")
        self.rf = Chart(10.5, 3.4, "B_hidden_unit_patterns")
        self.rf_slot = ChartSlot(self.rf, "가중치 패턴은 이 PC에서 실험을 실행하면 표시됩니다.", run_cb)
        rf_card.lay.addWidget(self.rf_slot)
        lay.addWidget(rf_card)

        lay.addWidget(Legal(
            "학습 설정(은닉 유닛 128개, 학습률 0.001, 같은 데이터, 200 epochs)을 같게 두고 seed(가중치 초기값과 미니배치 "
            "순서를 정하는 난수)만 바꾼 다섯 모델의 평가 정확도는 98.0% ± 0.2%, 모델 쌍 사이 예측 일치율은 평균 99.4%였습니다. "
            "같은 모델 쌍의 가중치 코사인 유사도는 원래 순서로 0.03, 순서를 맞춘 뒤 0.50, 순서와 배율을 모두 맞춘 뒤 "
            "0.52였습니다(첫 번째 층만 비교하면 0.44)."))
        lay.addStretch(1)

    def update_results(self, out, details):
        b = out["B_seed"]
        self.k_acc.set(f"{pct(b['test_acc_mean'])}", f"± {pct(b['test_acc_sd'])} (표준편차)")
        self.k_agree.set(f"{pct(b['prediction_agreement_mean'])}", None, BLUE)
        for kpi, key, color in [(self.k_rescaled, "weight_cosine_after_alignment_and_rescaling", BLUE),
                                (self.k_align, "weight_cosine_after_alignment", INK),
                                (self.k_cos, "weight_cosine_raw", ORANGE)]:
            if f"{key}_mean" in b:
                rng = b.get(f"{key}_range")
                kpi.set(f"{b[f'{key}_mean']:.2f}", f"범위 {rng[0]:.2f} ~ {rng[1]:.2f}" if rng else "", color)
        if details:
            self.draw_matrices(details["B"])
            self.draw_patterns(details["B"])

    def draw_matrices(self, d):
        fig = self.mats.fig
        fig.clear()
        axes = fig.subplots(1, 4)
        labels = [f"s{i}" for i in range(core.B_SEEDS)]
        for ax, key, title in zip(axes, ["agree", "raw_cos", "aligned_cos", "rescaled_cos"],
                                  ["예측 일치율", "코사인 · 원래 순서", "코사인 · 순서를 맞춘 뒤",
                                   "코사인 · 순서·배율을 맞춘 뒤"]):
            M = d[key]
            im = ax.imshow(np.clip(M, 0, 1), cmap=SEQ, vmin=0, vmax=1)
            annotate_heatmap(ax, M, lambda v: f"{v:.2f}", np.clip(M, 0, 1))
            ax.set_title(title)
            ax.set_xticks(range(len(labels)), labels)
            ax.set_yticks(range(len(labels)), labels)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
        cb = fig.colorbar(im, ax=axes, shrink=0.85)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8.5)
        self.mat_slot.show_chart()

    def draw_patterns(self, d):
        m0, m1 = d["models"][0], d["models"][1]
        perm = d["perm_1_to_0"]
        n = 12
        rows = [("seed 0", m0.coefs_[0][:, :n]),
                ("seed 1 (같은 순번)", m1.coefs_[0][:, :n]),
                ("seed 1 (순서 맞춤)", m1.coefs_[0][:, perm[:n]])]
        fig = self.rf.fig
        fig.clear()
        axes = fig.subplots(3, n)
        for r, (name, W) in enumerate(rows):
            for k in range(n):
                ax = axes[r, k]
                w = W[:, k].reshape(8, 8)
                lim = np.abs(w).max()
                ax.imshow(w, cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if r == 0:
                    ax.set_title(f"#{k}", fontsize=8, loc="center", fontweight="normal", color=MUTED)
                if k == 0:
                    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9, color=INK_2)
        self.rf_slot.show_chart()


class PageC(QWidget):
    def __init__(self, run_cb):
        super().__init__()
        self.details = None
        page, lay = new_page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page_scroll(page))

        kpis = QHBoxLayout()
        kpis.setSpacing(16)
        self.k_same = Kpi("예측 결과", "평가 데이터 540장 전부")
        self.k_cos = Kpi("가중치 코사인 유사도", "원본 vs 순서를 바꾼 배열")
        self.k_count = Kpi("기능이 같은 배열 수 (하한)", "은닉 유닛 128개의 순서 = 128!")
        for k in (self.k_same, self.k_cos, self.k_count):
            kpis.addWidget(k)
        lay.addLayout(kpis)

        ctrl_card = Card("직접 섞어 보기",
                         "실험 B의 seed 0 모델에서 은닉 유닛 128개의 순서를 무작위로 바꿉니다. 들어오는 가중치와 "
                         "나가는 가중치를 함께 옮기므로 계산 결과는 수학적으로 같습니다.")
        row = QHBoxLayout()
        row.addWidget(label("섞기 seed", "fieldLabel", wrap=False))
        self.sp_perm = QSpinBox()
        self.sp_perm.setRange(0, 999999)
        self.sp_perm.setValue(core.C_PERM_SEED)
        self.sp_perm.setFixedWidth(110)
        self.sp_perm.setMinimumHeight(32)
        row.addWidget(self.sp_perm)
        self.btn_perm = QPushButton("이 순서로 섞기")
        self.btn_perm.setObjectName("secondary")
        self.btn_rand = QPushButton("무작위로 다시 섞기")
        self.btn_rand.setObjectName("primary")
        for b in (self.btn_perm, self.btn_rand):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setEnabled(False)
            row.addWidget(b)
        row.addStretch(1)
        self.btn_perm.clicked.connect(lambda: self.reshuffle(self.sp_perm.value()))
        self.btn_rand.clicked.connect(self.random_shuffle)
        ctrl_card.lay.addLayout(row)
        lay.addWidget(ctrl_card)

        w_card = Card("가중치 배열: 원본과 순서를 바꾼 배열",
                      "입력(64) → 은닉(128) 가중치 행렬의 앞쪽 24칸. 열 하나가 은닉 유닛 하나이며, 아래 숫자는 그 칸에 "
                      "들어 있는 유닛의 원래 번호입니다. 각 열의 값은 그대로이고 자리만 바뀌었습니다. "
                      "빨강 = 음수, 파랑 = 양수.")
        self.wmap = Chart(10.5, 3.6, "C_weight_arrays")
        self.w_slot = ChartSlot(self.wmap, "가중치 배열은 이 PC에서 실험을 실행하면 표시됩니다.", run_cb)
        w_card.lay.addWidget(self.w_slot)
        lay.addWidget(w_card)

        o_card = Card("출력 비교: 두 배열이 내놓는 값",
                      "평가 데이터 540장 × 10개 클래스의 출력값(logit). 모든 점이 대각선 위에 있으면 두 배열의 출력이 같다는 뜻입니다.")
        self.out = Chart(10.5, 3.2, "C_output_comparison")
        self.o_slot = ChartSlot(self.out, "출력 비교는 이 PC에서 실험을 실행하면 표시됩니다.", run_cb)
        o_card.lay.addWidget(self.o_slot)
        lay.addWidget(o_card)

        means = Card("측정 내용")
        means.lay.addWidget(label(
            "• 순서를 바꾼 뒤 평가 데이터 540장에 대한 예측은 원본과 모두 같았고, 출력값(logit)의 최대 차이는 "
            "10<sup>−14</sup> 수준(부동소수점 오차)이었습니다.<br>"
            "• 원본과 순서를 바꾼 배열의 가중치 코사인 유사도는 0.03이었습니다. 원래 순서로 잰 코사인은 출력이 같은 "
            "두 배열 사이에서도 0에 가까울 수 있으므로, 실험 B에서는 순서(와 배율)를 맞춘 뒤의 값을 함께 보고합니다.<br>"
            "• 은닉 유닛 128개를 늘어놓는 순서는 128!가지(약 10<sup>215</sup>)이므로, 이 모델과 출력이 같은 가중치 배열은 "
            "최소 128!개입니다. ReLU 신경망에서는 유닛별 배율 조정으로도 출력이 같은 배열을 만들 수 있으므로"
            "(Dinh et al., 2017) 128!은 하한입니다.<br><br>"
            "이 결과는 새로운 현상의 발견이 아니라, 피드포워드 신경망에서 항상 성립하는 은닉 유닛의 순서 대칭성"
            "(Hecht-Nielsen, 1990; Chen et al., 1993)을 이 모델에서 직접 확인한 <b>시연</b>입니다.", "cardText"))
        lay.addWidget(means)

        refs = Card("관련 문헌")
        refs.lay.addWidget(label(references_html([0, 1, 3, 6, 7]), "cardText"))
        lay.addWidget(refs)

        lay.addWidget(Legal(
            "학습된 모델 하나의 은닉 유닛 순서만 바꾸었을 때, 평가 데이터에 대한 예측은 100% 같았고 가중치 코사인 유사도는 "
            "0.03이었습니다. 이 모델과 출력이 같은 가중치 배열은 최소 128! ≈ 10<sup>215</sup>개입니다."))
        lay.addStretch(1)

    def update_results(self, out, details):
        c = out["C_permutation"]
        self.set_kpis(c["prediction_identical"], c["weight_cosine_original_vs_permuted"])
        exp = c["number_of_equivalent_arrays"].split("10^")[-1]
        self.k_count.set(f"≈ 10<sup>{exp}</sup>", None, BLUE)
        if details:
            self.details = details
            self.btn_perm.setEnabled(True)
            self.btn_rand.setEnabled(True)
            self.draw(details["C"])

    def set_kpis(self, identical, cosine):
        self.k_same.set("✓ 완전히 동일" if identical else "✗ 다름", None, GOOD if identical else ORANGE)
        self.k_cos.set(f"{cosine:.3f}", None, ORANGE)

    def random_shuffle(self):
        self.sp_perm.setValue(int(np.random.default_rng().integers(0, 1_000_000)))
        self.reshuffle(self.sp_perm.value())

    def reshuffle(self, seed):
        m0, data = self.details["B"]["models"][0], self.details["data"]
        perm = np.random.default_rng(seed).permutation(core.HIDDEN)
        res = dict(core.permutation_check(m0, data, perm), perm=perm)
        self.set_kpis(res["prediction_identical"], res["weight_cosine_original_vs_permuted"])
        self.draw(res)

    def draw(self, c):
        m0 = self.details["B"]["models"][0]
        W1 = m0.coefs_[0]
        lim = np.percentile(np.abs(W1), 99)
        n = 24
        fig = self.wmap.fig
        fig.clear()
        axes = fig.subplots(1, 2)
        for ax, idx, title in [(axes[0], np.arange(n), f"원본 배열 (앞쪽 {n}칸)"),
                               (axes[1], c["perm"][:n], f"순서를 바꾼 배열 (앞쪽 {n}칸)")]:
            im = ax.imshow(W1[:, idx], cmap=DIV, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks(range(n), [str(i) for i in idx], fontsize=7)
            ax.set_xlabel("칸에 들어 있는 은닉 유닛 번호")
            ax.set_ylabel("입력 픽셀 (64)")
            ax.tick_params(length=0, labelsize=7.5)
        cb = fig.colorbar(im, ax=axes, shrink=0.85)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8)
        self.w_slot.show_chart()

        fig = self.out.fig
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2, width_ratios=[1.1, 1])
        a, b = c["logits_orig"].ravel(), c["logits_perm"].ravel()
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax1.plot([lo, hi], [lo, hi], color=LINE, lw=2, zorder=1)
        ax1.scatter(a, b, s=9, color=BLUE, alpha=0.45, linewidths=0, zorder=2)
        ax1.set_xlabel("원본 배열의 출력")
        ax1.set_ylabel("순서를 바꾼 배열의 출력")
        ax1.set_title(f"출력값 {a.size:,}개 · 최대 차이 {c['max_abs_logit_diff']:.0e} (부동소수점 오차)")
        ax1.grid(True)
        pad = 0.05 * (hi - lo)
        ax1.set_xlim(lo - pad, hi + pad)
        ax1.set_ylim(lo - pad, hi + pad)
        px = 36  # a central pixel of the 8x8 image (border pixels are nearly constant)
        k = np.arange(core.HIDDEN)
        ax2.plot(k, W1[px], color=BLUE, lw=1.2, label="원본")
        ax2.plot(k, W1[px, c["perm"]], color=ORANGE, lw=1.2, label="순서를 바꾼 배열")
        ax2.set_title(f"입력 픽셀 {px}번에서 나가는 가중치 128개")
        ax2.set_xlabel("배열 안의 위치 (은닉 유닛 자리)")
        ax2.legend(fontsize=8.5, loc="upper right", ncols=2, framealpha=0.95, edgecolor=LINE)
        ax2.margins(y=0.25)
        ax2.tick_params(length=0)
        self.o_slot.show_chart()


class SummaryPage(QWidget):
    ROWS = [
        ("A. 학습 설정 변경", "seed 0, 50 epochs 고정. 은닉 유닛 {4, 16, 128} × 학습률 {0.1, 0.001, 0.00001}", "{a}"),
        ("B. 같은 설정, 다른 seed", "은닉 유닛 128, 학습률 0.001, 200 epochs 고정. seed 0~4", "{b}"),
        ("C. 은닉 유닛 순서 교환 (시연)", "B의 seed 0 모델에서 은닉 유닛 128개의 순서만 무작위로 바꿈", "{c}"),
    ]

    def __init__(self, save_json, save_figs):
        super().__init__()
        page, lay = new_page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page_scroll(page))

        tcard = Card("결과 요약", "실험별 설정과 측정 결과입니다.")
        self.table = new_table(["실험", "설정", "결과"], 3, [False, True, True])
        tcard.lay.addWidget(self.table)
        lay.addWidget(tcard)

        ccard = Card("실험 B의 가중치 코사인 유사도",
                     "원래 순서로 잰 코사인에는 은닉 유닛의 순서 차이와 ReLU 배율 차이가 함께 들어 있으므로, 이를 하나씩 "
                     "제거한 값을 함께 보고합니다. 값은 모델 쌍 10개의 평균(괄호 안은 최솟값~최댓값)입니다.")
        self.cos_table = new_table(["비교 방법", "값", "설명"], 4, [False, False, True])
        ccard.lay.addWidget(self.cos_table)
        lay.addWidget(ccard)

        bc = Card("실험 B와 C의 비교")
        self.bc_table = new_table(["", "실험 B", "실험 C"], 4, [False, True, True])
        bc.lay.addWidget(self.bc_table)
        lay.addWidget(bc)

        lim = Card("범위와 한계")
        lim.lay.addWidget(label(
            "• 이 실험의 모델(MLP 64-128-10)과 데이터셋(digits)은 작습니다. 결과를 대형 모델에 그대로 적용할 수는 없습니다.<br>"
            "• seed에 따라 서로 다른 해에 도달하는 현상과 은닉 유닛의 순서 대칭성은 대형 모델에서도 보고되어 있습니다"
            "(Fort et al., 2019; Entezari et al., 2022; Ainsworth et al., 2023).<br>"
            "• 실험 A의 학습률 0.1 조합은 학습이 불안정하여, 운영체제와 수치 연산 라이브러리(BLAS)에 따라 소수점 둘째 자리 "
            "정도 달라질 수 있습니다. Windows 11 + Python 3.13 환경에서는 은닉 16이 96.1%(저장값 94.8%), 은닉 128이 "
            "96.7%(저장값 97.2%)였고, 정확도 범위(11.9% ~ 97.8%)와 실험 B·C의 값은 저장값과 같았습니다.", "cardText"))
        lay.addWidget(lim)

        refs = Card("관련 문헌", "각 문헌 아래에 내용과 관련된 실험을 적었습니다.")
        refs.lay.addWidget(label(references_html(), "cardText"))
        lay.addWidget(refs)

        ex = Card("내보내기", "현재 화면의 결과를 JSON으로, 그래프를 PNG(200 dpi)로 저장합니다.")
        row = QHBoxLayout()
        b1 = QPushButton("결과 JSON 저장")
        b1.setObjectName("secondary")
        b1.clicked.connect(save_json)
        b2 = QPushButton("그래프 PNG 저장")
        b2.setObjectName("secondary")
        b2.clicked.connect(save_figs)
        for b in (b1, b2):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(b)
        row.addStretch(1)
        ex.lay.addLayout(row)
        lay.addWidget(ex)
        lay.addStretch(1)

    def update_results(self, out, details):
        accs = [r["test_acc"] for r in out["A_human_choices"]]
        b, c = out["B_seed"], out["C_permutation"]

        def val(key):
            rng = b.get(f"{key}_range")
            v = f"{b[f'{key}_mean']:.2f}"
            return f"{v} ({rng[0]:.2f} ~ {rng[1]:.2f})" if rng else v

        has_rescale = "weight_cosine_after_alignment_and_rescaling_mean" in b
        rescaled = val("weight_cosine_after_alignment_and_rescaling") if has_rescale else "—"
        first = f"{b['first_layer_cosine_after_alignment_and_rescaling_mean']:.2f}" if has_rescale else "—"
        exp = c["number_of_equivalent_arrays"].split("10^")[-1]
        same = "100% 동일" if c["prediction_identical"] else "불일치"
        vals = {
            "a": f"평가 정확도 {pct(min(accs))} ~ {pct(max(accs))}",
            "b": f"평가 정확도 {pct(b['test_acc_mean'])} ± {pct(b['test_acc_sd'])}, 예측 일치율 "
                 f"{pct(b['prediction_agreement_mean'])}. 가중치 코사인: 원래 순서 {val('weight_cosine_raw')}, "
                 f"순서를 맞춘 뒤 {val('weight_cosine_after_alignment')}, 순서·배율을 맞춘 뒤 {rescaled}",
            "c": f"예측 {same}, 가중치 코사인 {c['weight_cosine_original_vs_permuted']:.2f}, "
                 f"출력이 같은 배열 최소 128! ≈ 10{exp.translate(SUPERSCRIPT)}개",
        }
        fill_table(self.table, [(n, st, r.format(**vals)) for n, st, r in self.ROWS])
        fill_table(self.cos_table, [
            ("원래 순서", val("weight_cosine_raw"),
             "두 가중치 목록을 그대로 비교. 출력이 같은 배열 사이에서도 순서 차이만으로 0에 가까울 수 있음(실험 C: "
             f"{c['weight_cosine_original_vs_permuted']:.2f})"),
            ("순서를 맞춘 뒤", val("weight_cosine_after_alignment"),
             "헝가리안 알고리즘으로 은닉 유닛 128개를 1:1로 짝지은 뒤 비교. 순서 차이를 제거한 값"),
            ("순서·배율을 맞춘 뒤", rescaled,
             "짝짓기 후 각 유닛의 입력 가중치와 편향의 크기를 1로 표준화(출력 가중치는 반대로 조정, 출력 변화 없음). "
             "순서와 ReLU 배율 차이를 모두 제거한 값"),
            ("순서·배율을 맞춘 뒤, 첫 번째 층만", first, "입력→은닉 가중치와 편향만 비교"),
        ])
        fill_table(self.bc_table, [
            ("같게 둔 것", "학습 설정(구조·학습률·데이터·epochs)", "학습된 모델 하나"),
            ("바꾼 것", "seed(가중치 초기값과 미니배치 순서)", "은닉 유닛의 순서"),
            ("예측", f"모델 쌍 평균 {pct(b['prediction_agreement_mean'])} 일치", f"{same}"),
            ("가중치 코사인", f"순서·배율을 맞춘 뒤 {rescaled}", f"{c['weight_cosine_original_vs_permuted']:.2f}"),
        ])


# ---- main window -----------------------------------------------------------
class MainWindow(QMainWindow):
    PAGES = [("◎", "개요", "실험 목적과 세 가지 실험 한눈에 보기"),
             ("A", "실험 A · 학습 설정", "학습 설정(은닉 유닛 수·학습률)에 따른 평가 정확도"),
             ("B", "실험 B · 같은 설정", "같은 설정, 다른 seed로 학습한 모델의 예측과 가중치 비교"),
             ("C", "실험 C · 순서 교환", "은닉 유닛 순서를 바꾼 배열의 예측과 가중치 비교"),
             ("≡", "결과 요약", "결과 표, 범위와 한계, 관련 문헌, 내보내기")]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weight·Expression Lab — 같은 레시피, 다른 가중치")
        self.resize(1360, 880)
        self.setMinimumSize(1100, 720)
        self.out, self.details, self.threads = None, None, []

        root = QWidget()
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self.build_sidebar())

        content = QWidget()
        content.setObjectName("content")
        v = QVBoxLayout(content)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.build_header())
        self.stack = QStackedWidget()
        self.overview = OverviewPage(self.go)
        self.page_a = PageA(self.train_custom)
        self.page_b = PageB(self.run_experiment)
        self.page_c = PageC(self.run_experiment)
        self.summary = SummaryPage(self.save_json, self.save_figures)
        self.pages = [self.overview, self.page_a, self.page_b, self.page_c, self.summary]
        for p in self.pages:
            self.stack.addWidget(p)
        v.addWidget(self.stack)
        h.addWidget(content, 1)
        self.setCentralWidget(root)

        self.load_saved()
        self.go(0)

    def build_sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(250)
        v = QVBoxLayout(side)
        v.setContentsMargins(16, 22, 16, 18)
        v.setSpacing(4)
        v.addWidget(label("Weight·Expression Lab", "brand"))
        v.addWidget(label("같은 레시피, 다른 가중치", "brandSub"))
        v.addSpacing(22)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        for i, (icon, name, _) in enumerate(self.PAGES):
            b = QPushButton(f"  {icon}    {name}")
            b.setObjectName("navButton")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, k=i: self.go(k))
            self.nav.addButton(b, i)
            v.addWidget(b)
        v.addStretch(1)
        v.addWidget(label("데이터 · scikit-learn digits<br>모델 · MLP 64-128-10", "sideInfo"))
        return side

    def build_header(self):
        head = QFrame()
        head.setObjectName("header")
        h = QHBoxLayout(head)
        h.setContentsMargins(28, 16, 28, 16)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        trow = QHBoxLayout()
        self.title = label("", "pageTitle", wrap=False)
        self.badge = label("", "badge", wrap=False)
        trow.addWidget(self.title)
        trow.addSpacing(10)
        trow.addWidget(self.badge)
        trow.addStretch(1)
        titles.addLayout(trow)
        self.subtitle = label("", "pageSub")
        titles.addWidget(self.subtitle)
        h.addLayout(titles, 1)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.btn_run = QPushButton("▶  실험 실행")
        self.btn_run.setObjectName("primary")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setToolTip("데이터 분할 → 실험 A(9개 모델) → 실험 B(5개 모델) → 실험 C를 이 PC에서 다시 실행합니다.")
        self.btn_run.clicked.connect(self.run_experiment)
        right.addWidget(self.btn_run, alignment=Qt.AlignmentFlag.AlignRight)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(260)
        self.progress.setFixedHeight(8)
        self.progress.hide()
        right.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignRight)
        self.status = label("", "status", wrap=False)
        right.addWidget(self.status, alignment=Qt.AlignmentFlag.AlignRight)
        h.addLayout(right)
        return head

    def go(self, i):
        self.stack.setCurrentIndex(i)
        self.nav.button(i).setChecked(True)
        self.title.setText(self.PAGES[i][1])
        self.subtitle.setText(self.PAGES[i][2])

    def set_source(self, live):
        self.badge.setText("이 PC에서 재현한 결과" if live else "저장된 결과 · results.json")
        self.badge.setProperty("source", "live" if live else "saved")
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)

    def apply(self, out, details):
        self.out, self.details = out, details
        for p in self.pages:
            p.update_results(out, details)

    def load_saved(self):
        if RESULTS_JSON.exists():
            self.apply(json.loads(RESULTS_JSON.read_text(encoding="utf-8")), None)
            self.set_source(False)
            self.status.setText("가중치 시각화를 보려면 실험을 실행하세요.")
        else:
            self.status.setText("저장된 결과가 없습니다. 실험을 실행하세요.")

    # -- background jobs --
    def start(self, fn, args, on_done, on_progress):
        thread = QThread(self)
        worker = Worker(fn, *args)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_done)
        worker.failed.connect(self.on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self.cleanup_threads)
        self.threads.append((thread, worker))
        thread.start()

    def cleanup_threads(self):
        self.threads = [(t, w) for t, w in self.threads if not t.isFinished()]

    def run_experiment(self):
        if not self.btn_run.isEnabled():
            return
        self.btn_run.setEnabled(False)
        self.btn_run.setText("실행 중…")
        self.progress.setValue(0)
        self.progress.show()
        self.start(core.run_all, (), self.on_run_done, self.on_run_progress)

    def on_run_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.status.setText(msg)

    def on_run_done(self, res):
        out, details = res
        self.apply(out, details)
        self.set_source(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  다시 실행")
        self.progress.hide()
        saved = json.loads(RESULTS_JSON.read_text(encoding="utf-8")) if RESULTS_JSON.exists() else None
        if saved is None:
            self.status.setText("실험 완료")
        elif saved == out:
            self.status.setText("실험 완료 · results.json과 완전히 일치")
        else:
            self.status.setText("실험 완료 · results.json과 일부 값이 다름 (결과 요약 → 범위와 한계 참고)")

    def on_failed(self, tb):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  실험 실행")
        self.page_a.set_training(False)
        self.progress.hide()
        self.status.setText("오류가 발생했습니다.")
        QMessageBox.critical(self, "오류", tb)

    def train_custom(self, hidden, lr, seed, epochs):
        self.page_a.set_training(True)
        self.start(core.train_custom, (hidden, lr, seed, epochs), self.on_custom_done, self.on_custom_progress)

    def on_custom_progress(self, pct, msg):
        self.status.setText(msg)

    def on_custom_done(self, r):
        self.page_a.set_training(False)
        self.page_a.show_custom(r)
        self.status.setText(f"직접 해보기 완료 · 정확도 {pct(r['test_acc'])}")

    # -- export --
    def save_json(self):
        if not self.out:
            return
        path, _ = QFileDialog.getSaveFileName(self, "결과 저장", str(ROOT / "results_gui.json"), "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.status.setText(f"저장됨: {path}")

    def save_figures(self):
        folder = QFileDialog.getExistingDirectory(self, "그래프를 저장할 폴더", str(ROOT))
        if not folder:
            return
        charts = [self.overview.digits, self.page_a.heat, self.page_b.cos_algo, self.page_b.cos_toy,
                  self.page_b.mats, self.page_b.rf, self.page_c.wmap, self.page_c.out]
        if self.page_a.loss.isVisible():
            charts.append(self.page_a.loss)
        saved = 0
        for ch in charts:
            if ch.fig.axes:
                ch.fig.savefig(Path(folder) / f"{ch.name}.png", dpi=200)
                saved += 1
        msg = f"그래프 {saved}개를 저장했습니다: {folder}"
        if not self.details:
            msg += " (실험을 실행하면 가중치 그래프도 저장됩니다)"
        self.status.setText(msg)

    def closeEvent(self, e):
        for t, _ in self.threads:
            t.quit()
            t.wait()
        super().closeEvent(e)


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont(pick_qt_font(), 10))
    app.setStyleSheet(STYLE)
    setup_matplotlib()
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
