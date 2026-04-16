# -*- coding: utf-8 -*-
"""judge_modelload.py"""

from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from PyQt5 import QtCore, QtGui, QtWidgets


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


class ModelBrowserDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial_dir: str = None):
        super().__init__(parent)
        self.setWindowTitle("열기")
        self.setModal(True)
        self.resize(900, 600)
        
        self.current_dir = initial_dir or os.path.expanduser("~")
        if not os.path.isdir(self.current_dir):
            self.current_dir = os.path.expanduser("~")
        
        self.selected_model_path: Optional[str] = None
        self.selected_model_info: Optional[Dict[str, Any]] = None
        
        self._setup_ui()
        self._load_directory(self.current_dir)
    
    def _setup_ui(self):
        # 상단 네비게이션
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
        
        # 중앙 파일 리스트 + 정보
        content_layout = QtWidgets.QHBoxLayout()
        
        self.tree_files = QtWidgets.QTreeWidget()
        self.tree_files.setHeaderLabels(["이름", "수정한 날짜", "유형", "크기"])
        self.tree_files.setColumnWidth(0, 300)
        self.tree_files.setColumnWidth(1, 150)
        self.tree_files.setColumnWidth(2, 100)
        self.tree_files.setColumnWidth(3, 100)
        self.tree_files.setAlternatingRowColors(True)
        self.tree_files.setSortingEnabled(True)
        self.tree_files.setRootIsDecorated(False)
        
        self.text_model_info = QtWidgets.QPlainTextEdit()
        self.text_model_info.setReadOnly(True)
        self.text_model_info.setPlaceholderText(".pt 파일 선택 시 정보 표시")
        
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.tree_files)
        splitter.addWidget(self.text_model_info)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        
        content_layout.addWidget(splitter)
        
        # 하단 파일명 + 버튼
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
        
        # 전체 레이아웃
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(nav_layout)
        main_layout.addLayout(content_layout)
        main_layout.addLayout(bottom_layout)
        
        self.setLayout(main_layout)
        
        # 시그널 연결
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
            QtWidgets.QMessageBox.warning(self, "접근 거부", f"폴더에 접근할 수 없습니다:\n{self.current_dir}")
            return
        
        folders = []
        pt_files = []
        
        for entry in entries:
            full_path = os.path.join(self.current_dir, entry)
            
            if os.path.isdir(full_path):
                folders.append(entry)
            elif entry.lower().endswith(".pt"):
                pt_files.append(entry)
        
        # 폴더 추가
        folders.sort(key=str.lower)
        for folder in folders:
            full_path = os.path.join(self.current_dir, folder)
            try:
                mtime = os.path.getmtime(full_path)
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                mtime_str = ""
            
            item = QtWidgets.QTreeWidgetItem(self.tree_files)
            item.setText(0, folder)
            item.setText(1, mtime_str)
            item.setText(2, "파일 폴더")
            item.setText(3, "")
            item.setData(0, QtCore.Qt.UserRole, full_path)
            item.setData(1, QtCore.Qt.UserRole, "folder")
        
        # .pt 파일 추가
        pt_files.sort(key=str.lower)
        for pt_file in pt_files:
            full_path = os.path.join(self.current_dir, pt_file)
            try:
                size_bytes = os.path.getsize(full_path)
                size_str = self._format_size(size_bytes)
                mtime = os.path.getmtime(full_path)
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                size_str = ""
                mtime_str = ""
            
            item = QtWidgets.QTreeWidgetItem(self.tree_files)
            item.setText(0, pt_file)
            item.setText(1, mtime_str)
            item.setText(2, "PT 파일")
            item.setText(3, size_str)
            item.setData(0, QtCore.Qt.UserRole, full_path)
            item.setData(1, QtCore.Qt.UserRole, "pt_file")
    
    def _format_size(self, size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def _on_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        item_type = item.data(1, QtCore.Qt.UserRole)
        
        if item_type == "folder":
            folder_path = item.data(0, QtCore.Qt.UserRole)
            self._load_directory(folder_path)
        elif item_type == "pt_file":
            self.accept()
    
    def _on_selection_changed(self):
        selected_items = self.tree_files.selectedItems()
        
        if not selected_items:
            self.btn_ok.setEnabled(False)
            self.label_selected_file.clear()
            self.text_model_info.clear()
            self.selected_model_path = None
            self.selected_model_info = None
            return
        
        item = selected_items[0]
        item_type = item.data(1, QtCore.Qt.UserRole)
        
        if item_type == "pt_file":
            pt_path = item.data(0, QtCore.Qt.UserRole)
            self.selected_model_path = pt_path
            
            self.label_selected_file.setText(os.path.basename(pt_path))
            self.btn_ok.setEnabled(True)
            
            json_path = pt_path.rsplit(".", 1)[0] + ".json"
            
            if os.path.exists(json_path):
                self._load_model_info(json_path)
            else:
                self.text_model_info.setPlainText(f"JSON 파일을 찾을 수 없습니다.\n\n예상 경로:\n{json_path}")
                self.selected_model_info = None
        else:
            self.btn_ok.setEnabled(False)
            self.label_selected_file.clear()
            self.text_model_info.clear()
            self.selected_model_path = None
            self.selected_model_info = None
    
    def _load_model_info(self, json_path: str):
        data = read_json(json_path)
        
        if data is None:
            self.text_model_info.setPlainText(f"JSON 파일을 읽을 수 없습니다.\n\n파일: {json_path}")
            self.selected_model_info = None
            return
        
        app_info = data.get("application_info", {})
        
        if not app_info:
            self.text_model_info.setPlainText(f"'application_info' 섹션이 없습니다.")
            self.selected_model_info = data
            return
        
        info_lines = ["모델 정보\n", "=" * 40]
        
        for key in ["대상 재질", "대상 부호", "설명", "베포 날짜", "만든 사람"]:
            value = app_info.get(key, "")
            if value:
                info_lines.append(f"\n{key}: {value}")
        
        info_lines.append("\n" + "=" * 40)
        
        self.text_model_info.setPlainText("\n".join(info_lines))
        self.selected_model_info = app_info
    
    def get_selected_model(self) -> Optional[tuple[str, Dict[str, Any]]]:
        if self.selected_model_path:
            return (self.selected_model_path, self.selected_model_info or {})
        return None


class JudgeModelController(QtCore.QObject):
    model_loaded = QtCore.pyqtSignal(str, dict)
    
    def __init__(
        self,
        main_window,
        btn_judge_data_browse_2=None,
        le_judge_data_path_2=None,
        initial_model_dir: str = None,
        logger = None
    ):
        super().__init__(main_window)
        
        self.main_window = main_window
        self.btn_browse = btn_judge_data_browse_2
        self.le_path = le_judge_data_path_2
        
        self.model_dir = initial_model_dir or os.path.join(os.getcwd(), "models")
        
        self.current_model_path: Optional[str] = None
        self.current_model_info: Optional[Dict[str, Any]] = None
        self.loaded_model = None
        self.logger = logger
        
        self._connect_signals()
    
    def _connect_signals(self):
        if self.btn_browse and hasattr(self.btn_browse, "clicked"):
            self.btn_browse.clicked.connect(self.open_model_browser)
    
    def open_model_browser(self):
        dlg = ModelBrowserDialog(parent=self.main_window, initial_dir=self.model_dir)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            result = dlg.get_selected_model()
            
            if result:
                model_path, model_info = result
                self.load_model(model_path, model_info)
    
    def load_model(self, model_path: str, model_info: Dict[str, Any]):
        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "모델 파일 없음",
                f"모델 파일을 찾을 수 없습니다:\n{model_path}"
            )
            return
        
        self.current_model_path = model_path
        self.current_model_info = model_info
        
        # le_judge_data_path_2에 경로 표시
        if self.le_path:
            self.le_path.setText(model_path)
        
        self.model_loaded.emit(model_path, model_info)
        self._set_status(f"모델 로드 완료: {os.path.basename(model_path)}")
        
        # 🎯 로그 추가
        if self.logger:
            self.logger.log_model(f"모델 로드: {os.path.basename(model_path)}")
    
    def get_current_model_path(self) -> Optional[str]:
        return self.current_model_path
    
    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        return self.current_model_info
    
    def get_loaded_model(self):
        return self.loaded_model
    
    def _set_status(self, msg: str):
        try:
            sb = self.main_window.statusBar()
            if sb:
                sb.showMessage(msg, 5000)
        except Exception:
            pass