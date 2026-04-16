# -*- coding: utf-8 -*-
"""judge_dataload.py"""

from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


# ============================================
# CSV 로더 (학습 코드에서 가져옴)
# ============================================
def _make_unique_columns(cols):
    out = []
    seen = {}
    unnamed_count = 0
    for c in cols:
        name = "" if c is None else str(c).strip()
        if name == "" or name.lower() == "nan":
            unnamed_count += 1
            name = f"Unnamed_{unnamed_count}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def load_pllink_csv_safe(path, encoding="utf-16"):
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
    rows = [line.rstrip("\n").rstrip("\r").split("\t") for line in lines]
    raw = pd.DataFrame(rows)
    raw = raw.replace(r"^\s*$", np.nan, regex=True)
    raw = raw.infer_objects(copy=False)
    raw = raw.dropna(axis=1, how="all")
    
    header_keywords = ["인덱스", "SW", "측정", "시간", "속도", "Name", "Unit", "Result", "Signal", "신호", "균열"]
    header_row = None
    for i in range(len(raw)):
        row_str = " ".join(raw.iloc[i].astype(str).tolist())
        match_count = sum(k in row_str for k in header_keywords)
        if match_count >= 2:
            header_row = i
            break
    if header_row is None:
        header_row = 0
    
    data = raw.iloc[header_row + 1:].reset_index(drop=True)
    header = _make_unique_columns(raw.iloc[header_row].tolist())
    data.columns = header
    data = data.dropna(how="all")
    return data


def read_measurement_csv_smart(file_path: str) -> pd.DataFrame:
    encodings_probe = ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "cp949", "euc-kr"]
    last_err = None
    
    for enc in encodings_probe:
        try:
            with open(file_path, "r", encoding=enc) as f:
                head = f.read(4096)
            if "\t" in head:
                df = load_pllink_csv_safe(file_path, encoding=enc)
                df.columns = _make_unique_columns([str(c).strip() for c in df.columns])
                return df
        except Exception as e:
            last_err = e
    
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-16"]:
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=sep)
                if df is not None and df.shape[1] >= 1:
                    df.columns = _make_unique_columns([str(c).strip() for c in df.columns])
                    df = df.replace(r"^\s*$", np.nan, regex=True)
                    df = df.infer_objects(copy=False).dropna(how="all")
                    return df
            except Exception as e:
                last_err = e
    
    raise ValueError(f"CSV 로드 실패: {file_path}\n원인: {last_err}")


def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_wave_start_idx(df: pd.DataFrame) -> int:
    cols = list(df.columns)
    
    for i, c in enumerate(cols):
        if str(c).strip().lower() == "wave":
            return i
    
    for i, c in enumerate(cols):
        c_str = str(c).strip().lower()
        if ("signal" in c_str) or ("신호" in c_str):
            return min(i + 1, len(cols) - 1)
    
    for i, c in enumerate(cols):
        c_str = str(c).strip().lower()
        if c_str.startswith("unnamed"):
            return i
    
    for i, c in enumerate(cols):
        s = str(c).strip()
        try:
            float(s)
            return i
        except Exception:
            pass
    
    return 0


def detect_columns_and_wavecols(df: pd.DataFrame):
    label_col = pick_first_existing(df, ["균열유무", "균열여부", "label", "Label", "crack", "Crack"])
    name_col = pick_first_existing(df, ["Name", "name", "제품명", "이름"])
    
    if label_col is None:
        label_col = None
    
    wave_start = find_wave_start_idx(df)
    if label_col and label_col in df.columns:
        label_idx = list(df.columns).index(label_col)
    else:
        label_idx = len(df.columns)
    
    if wave_start >= label_idx:
        wave_start = label_idx
    
    wave_cols = list(df.columns)[wave_start:label_idx]
    
    mode = "cols"
    if "wave" in df.columns and wave_cols == ["wave"]:
        mode = "wave"
    
    return name_col, label_col, wave_cols, mode


# ============================================
# 파형 추출
# ============================================
def extract_waveform(df_row, wave_cols, mode):
    if mode == "wave" and "wave" in df_row.index:
        s = str(df_row["wave"]).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.replace(" ", "")
        w = np.fromstring(s, sep=",", dtype=np.float32)
        w = w[np.isfinite(w)]
        return w if w.size > 0 else np.zeros(2)
    else:
        amps = pd.to_numeric(df_row[wave_cols], errors="coerce").values
        amps = np.nan_to_num(amps, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return amps


# ============================================
# CSV 브라우저 다이얼로그
# ============================================
class CSVBrowserDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial_dir: str = None):
        super().__init__(parent)
        self.setWindowTitle("CSV 파일 선택")
        self.setModal(True)
        self.resize(900, 600)
        
        self.current_dir = initial_dir or os.path.expanduser("~")
        if not os.path.isdir(self.current_dir):
            self.current_dir = os.path.expanduser("~")
        
        self.selected_csv_path: Optional[str] = None
        
        self._setup_ui()
        self._load_directory(self.current_dir)
    
    def _setup_ui(self):
        nav_layout = QtWidgets.QHBoxLayout()
        
        self.btn_up = QtWidgets.QPushButton("↑")
        self.btn_up.setFixedWidth(40)
        
        self.combo_drives = QtWidgets.QComboBox()
        self.combo_drives.setFixedWidth(70)
        self._load_drives()
        
        self.label_current_path = QtWidgets.QLineEdit()
        self.label_current_path.setReadOnly(True)
        
        nav_layout.addWidget(self.btn_up)
        nav_layout.addWidget(self.combo_drives)
        nav_layout.addWidget(self.label_current_path)
        
        self.tree_files = QtWidgets.QTreeWidget()
        self.tree_files.setHeaderLabels(["이름", "수정한 날짜", "유형", "크기"])
        self.tree_files.setColumnWidth(0, 300)
        self.tree_files.setColumnWidth(1, 150)
        self.tree_files.setColumnWidth(2, 100)
        self.tree_files.setColumnWidth(3, 100)
        self.tree_files.setAlternatingRowColors(True)
        self.tree_files.setSortingEnabled(True)
        self.tree_files.setRootIsDecorated(False)
        
        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(QtWidgets.QLabel("파일 이름:"))
        
        self.label_selected_file = QtWidgets.QLineEdit()
        self.label_selected_file.setReadOnly(True)
        bottom_layout.addWidget(self.label_selected_file)
        
        self.btn_ok = QtWidgets.QPushButton("열기")
        self.btn_ok.setEnabled(False)
        self.btn_cancel = QtWidgets.QPushButton("취소")
        
        bottom_layout.addWidget(self.btn_ok)
        bottom_layout.addWidget(self.btn_cancel)
        
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.tree_files)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)
        
        self.btn_up.clicked.connect(self._go_parent_directory)
        self.combo_drives.currentTextChanged.connect(self._on_drive_changed)
        self.tree_files.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree_files.itemSelectionChanged.connect(self._on_selection_changed)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
    
    def _load_drives(self):
        if os.name == 'nt':
            import string
            from ctypes import windll
            drives = []
            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    drives.append(f"{letter}:")
                bitmask >>= 1
            self.combo_drives.addItems(drives)
            
            current_drive = os.path.splitdrive(self.current_dir)[0]
            if current_drive:
                idx = self.combo_drives.findText(current_drive)
                if idx >= 0:
                    self.combo_drives.setCurrentIndex(idx)
        else:
            self.combo_drives.addItem("/")
    
    def _on_drive_changed(self, drive: str):
        if drive:
            if os.name == 'nt':
                self._load_directory(drive + "\\")
            else:
                self._load_directory(drive)
    
    def _go_parent_directory(self):
        parent_dir = os.path.dirname(self.current_dir)
        if parent_dir and parent_dir != self.current_dir:
            self._load_directory(parent_dir)
    
    def _load_directory(self, dir_path: str):
        if not os.path.isdir(dir_path):
            return
        
        self.current_dir = os.path.abspath(dir_path)
        self.label_current_path.setText(self.current_dir)
        
        current_drive = os.path.splitdrive(self.current_dir)[0]
        if current_drive and os.name == 'nt':
            idx = self.combo_drives.findText(current_drive)
            if idx >= 0:
                self.combo_drives.blockSignals(True)
                self.combo_drives.setCurrentIndex(idx)
                self.combo_drives.blockSignals(False)
        
        self.tree_files.clear()
        
        try:
            entries = os.listdir(self.current_dir)
        except PermissionError:
            return
        
        folders = []
        csv_files = []
        
        for entry in entries:
            full_path = os.path.join(self.current_dir, entry)
            
            if os.path.isdir(full_path):
                folders.append(entry)
            elif entry.lower().endswith(".csv"):
                csv_files.append(entry)
        
        folders.sort(key=str.lower)
        for folder in folders:
            full_path = os.path.join(self.current_dir, folder)
            try:
                mtime = os.path.getmtime(full_path)
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except:
                mtime_str = ""
            
            item = QtWidgets.QTreeWidgetItem(self.tree_files)
            item.setText(0, folder)
            item.setText(1, mtime_str)
            item.setText(2, "파일 폴더")
            item.setText(3, "")
            item.setData(0, QtCore.Qt.UserRole, full_path)
            item.setData(1, QtCore.Qt.UserRole, "folder")
        
        csv_files.sort(key=str.lower)
        for csv_file in csv_files:
            full_path = os.path.join(self.current_dir, csv_file)
            try:
                size_bytes = os.path.getsize(full_path)
                size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024*1024 else f"{size_bytes / (1024*1024):.1f} MB"
                mtime = os.path.getmtime(full_path)
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except:
                size_str = ""
                mtime_str = ""
            
            item = QtWidgets.QTreeWidgetItem(self.tree_files)
            item.setText(0, csv_file)
            item.setText(1, mtime_str)
            item.setText(2, "CSV 파일")
            item.setText(3, size_str)
            item.setData(0, QtCore.Qt.UserRole, full_path)
            item.setData(1, QtCore.Qt.UserRole, "csv_file")
    
    def _on_item_double_clicked(self, item, column):
        item_type = item.data(1, QtCore.Qt.UserRole)
        
        if item_type == "folder":
            folder_path = item.data(0, QtCore.Qt.UserRole)
            self._load_directory(folder_path)
        elif item_type == "csv_file":
            self.accept()
    
    def _on_selection_changed(self):
        selected_items = self.tree_files.selectedItems()
        
        if not selected_items:
            self.btn_ok.setEnabled(False)
            self.label_selected_file.clear()
            self.selected_csv_path = None
            return
        
        item = selected_items[0]
        item_type = item.data(1, QtCore.Qt.UserRole)
        
        if item_type == "csv_file":
            csv_path = item.data(0, QtCore.Qt.UserRole)
            self.selected_csv_path = csv_path
            self.label_selected_file.setText(os.path.basename(csv_path))
            self.btn_ok.setEnabled(True)
        else:
            self.btn_ok.setEnabled(False)
            self.label_selected_file.clear()
            self.selected_csv_path = None
    
    def get_selected_csv(self) -> Optional[str]:
        return self.selected_csv_path


# ============================================
# 시각화 윈도우
# ============================================
class WaveformVisualizerWindow(QtWidgets.QDialog):
    def __init__(self, parent, waveforms: List[tuple], sampling_rate=1000000):
        super().__init__(parent)
        self.setWindowTitle("파형 시각화")
        self.resize(800, 600 * len(waveforms))
        
        # 한글 폰트 설정 (fallback 처리)
        try:
            import matplotlib.font_manager as fm
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            korean_fonts = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Gulim']
            selected_font = None
            for font in korean_fonts:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.family'] = selected_font
            else:
                plt.rcParams['font.family'] = 'sans-serif'
            
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        layout = QtWidgets.QVBoxLayout()
        
        fig, axes = plt.subplots(len(waveforms), 1, figsize=(8, 4 * len(waveforms)))
        if len(waveforms) == 1:
            axes = [axes]
        
        for ax, (title, wave) in zip(axes, waveforms):
            time = np.arange(len(wave)) / sampling_rate * 1e6
            ax.plot(time, wave)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("시간 (us)", fontsize=10)
            ax.set_ylabel("진폭", fontsize=10)
            ax.grid(True)
        
        plt.tight_layout()
        
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        btn_close = QtWidgets.QPushButton("닫기")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)
        
        self.setLayout(layout)


# ============================================
# Judge Data 컨트롤러
# ============================================
class JudgeDataController(QtCore.QObject):
    # 🎯 데이터 로드 완료 시그널 (csv_path를 전달)
    data_loaded = QtCore.pyqtSignal(str)
    
    def __init__(
        self,
        main_window,
        btn_judge_data_browse=None,
        le_judge_data_path=None,
        table_judge_data=None,
        le_judge_index_1=None,
        le_judge_index_2=None,
        le_judge_index_3=None,
        le_judge_index_4=None,
        btn_judge_visualize=None,
        btn_project_prev_2=None,
        btn_project_next_2=None,
        initial_data_dir: str = None,
        logger = None
    ):
        super().__init__(main_window)
        
        self.main_window = main_window
        self.btn_browse = btn_judge_data_browse
        self.le_path = le_judge_data_path
        self.table = table_judge_data
        self.le_indices = [le_judge_index_1, le_judge_index_2, le_judge_index_3, le_judge_index_4]
        self.btn_visualize = btn_judge_visualize
        self.btn_prev = btn_project_prev_2
        self.btn_next = btn_project_next_2
        
        self.data_dir = initial_data_dir or os.path.join(os.getcwd(), "data")
        
        self.df: Optional[pd.DataFrame] = None
        self.name_col: Optional[str] = None
        self.label_col: Optional[str] = None
        self.wave_cols: List[str] = []
        self.mode: str = "cols"
        
        self.page_size = 10
        self.page_index = 0
        self.logger = logger
        
        self._connect_signals()
    
    def _connect_signals(self):
        if self.btn_browse:
            self.btn_browse.clicked.connect(self.open_csv_browser)
        
        if self.btn_visualize:
            self.btn_visualize.clicked.connect(self.visualize_waveforms)
        
        if self.btn_prev:
            self.btn_prev.clicked.connect(self.go_prev_page)
        
        if self.btn_next:
            self.btn_next.clicked.connect(self.go_next_page)
        
        if self.table:
            self.table.itemChanged.connect(self._on_checkbox_changed)
        
        for le in self.le_indices:
            if le:
                le.textChanged.connect(self._sync_from_lineedit)
    
    def open_csv_browser(self):
        dlg = CSVBrowserDialog(parent=self.main_window, initial_dir=self.data_dir)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            csv_path = dlg.get_selected_csv()
            if csv_path:
                self.load_csv(csv_path)
    
    def load_csv(self, csv_path: str):
        try:
            self.df = read_measurement_csv_smart(csv_path)
            self.name_col, self.label_col, self.wave_cols, self.mode = detect_columns_and_wavecols(self.df)
            
            if self.le_path:
                self.le_path.setText(os.path.basename(csv_path))
            
            self.page_index = 0
            self._display_table()
            self._set_status(f"CSV 로드 완료: {os.path.basename(csv_path)} (총 {len(self.df)}행)")
            
            # 🎯 로그 추가 (먼저)
            if self.logger:
                self.logger.log_data(f"데이터 로드: {os.path.basename(csv_path)} ({len(self.df)}행)")
            
            # 🎯 데이터 로드 완료 시그널 발생
            self.data_loaded.emit(csv_path)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "CSV 로드 실패",
                f"파일을 읽을 수 없습니다:\n{csv_path}\n\n오류: {e}"
            )
    
    def _display_table(self):
        if self.df is None or self.table is None:
            return
        
        display_cols = [c for c in self.df.columns if c not in self.wave_cols][:10]
        
        total_rows = len(self.df)
        start_idx = self.page_index * self.page_size
        end_idx = min(start_idx + self.page_size, total_rows)
        
        df_page = self.df.iloc[start_idx:end_idx]
        
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setRowCount(len(df_page))
        self.table.setColumnCount(len(display_cols) + 1)
        self.table.setHorizontalHeaderLabels(["선택"] + display_cols)
        
        for display_row_idx, actual_row_idx in enumerate(range(start_idx, end_idx)):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setProperty("row_index", actual_row_idx)
            
            cell_widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(cell_widget)
            layout.addWidget(checkbox)
            layout.setAlignment(QtCore.Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            
            self.table.setCellWidget(display_row_idx, 0, cell_widget)
            
            for col_idx, col_name in enumerate(display_cols):
                value = str(self.df.iloc[actual_row_idx][col_name])
                if len(value) > 50:
                    value = value[:50] + "..."
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(display_row_idx, col_idx + 1, item)
        
        self.table.setColumnWidth(0, 60)
        self.table.blockSignals(False)
        
        total_pages = (total_rows + self.page_size - 1) // self.page_size
        self._set_status(f"페이지 {self.page_index + 1}/{total_pages} (전체 {total_rows}행, {start_idx + 1}-{end_idx}행 표시)")
    
    def go_prev_page(self):
        if self.df is None:
            return
        if self.page_index > 0:
            self.page_index -= 1
            self._display_table()
    
    def go_next_page(self):
        if self.df is None:
            return
        total_pages = (len(self.df) + self.page_size - 1) // self.page_size
        if self.page_index + 1 < total_pages:
            self.page_index += 1
            self._display_table()
    
    def _on_checkbox_changed(self, item):
        pass
    
    def _get_checked_indices(self) -> List[int]:
        if self.table is None:
            return []
        
        checked = []
        for display_row_idx in range(self.table.rowCount()):
            widget = self.table.cellWidget(display_row_idx, 0)
            if widget:
                checkbox = widget.findChild(QtWidgets.QCheckBox)
                if checkbox and checkbox.isChecked():
                    actual_row_idx = checkbox.property("row_index")
                    checked.append(actual_row_idx)
        return checked
    
    def _sync_from_lineedit(self):
        """LineEdit에서 인덱스 읽어와서 체크박스와 동기화 (1-based → 0-based 변환)"""
        indices_from_le = []
        for le in self.le_indices:
            if le:
                text = le.text().strip()
                if text:
                    try:
                        idx = int(text) - 1  # 🆕 1-based → 0-based
                        if 0 <= idx < len(self.df):
                            indices_from_le.append(idx)
                    except:
                        pass
        
        if self.table:
            self.table.blockSignals(True)
            for display_row_idx in range(self.table.rowCount()):
                widget = self.table.cellWidget(display_row_idx, 0)
                if widget:
                    checkbox = widget.findChild(QtWidgets.QCheckBox)
                    if checkbox:
                        actual_row_idx = checkbox.property("row_index")
                        checkbox.setChecked(actual_row_idx in indices_from_le)
            self.table.blockSignals(False)

    def visualize_waveforms(self):
        """시각화 (인덱스는 1-based로 표시)"""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self.main_window, "데이터 없음", "먼저 CSV 파일을 로드하세요.")
            return
        
        checked = self._get_checked_indices()
        
        # 체크박스가 없으면 LineEdit에서 수동 입력된 인덱스 사용
        if not checked:
            for le in self.le_indices:
                if le:
                    text = le.text().strip()
                    if text:
                        try:
                            idx = int(text) - 1  # 🆕 1-based → 0-based
                            if 0 <= idx < len(self.df) and idx not in checked:
                                checked.append(idx)
                        except:
                            pass
        
        if not checked:
            QtWidgets.QMessageBox.warning(self.main_window, "선택 없음", "체크박스를 선택하거나 인덱스를 입력하세요.")
            return
        
        if len(checked) > 4:
            QtWidgets.QMessageBox.warning(self.main_window, "선택 초과", "최대 4개까지 선택 가능합니다.")
            checked = checked[:4]
        
        # 🆕 LineEdit에 1-based 인덱스로 표시
        for i, idx in enumerate(checked):
            if i < len(self.le_indices) and self.le_indices[i]:
                self.le_indices[i].setText(str(idx + 1))  # 0-based → 1-based
        
        waveforms = []
        for idx in checked:
            row = self.df.iloc[idx]
            
            if self.name_col and self.name_col in row.index:
                title = str(row[self.name_col])
            else:
                title = f"Index {idx + 1}"  # 🆕 1-based 표시
            
            wave = extract_waveform(row, self.wave_cols, self.mode)
            waveforms.append((title, wave))
        
        visualizer = WaveformVisualizerWindow(self.main_window, waveforms)
        visualizer.exec_()
    
    def _set_status(self, msg: str):
        try:
            sb = self.main_window.statusBar()
            if sb:
                sb.showMessage(msg, 5000)
        except:
            pass