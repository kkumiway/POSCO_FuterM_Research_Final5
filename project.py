# -*- coding: utf-8 -*-
"""
project.py (v7 - UI 설정 존중 버전)
- ✅ UI 파일에서 설정한 테이블 크기, 컬럼 너비 등을 존중
- ✅ 강제로 덮어쓰지 않음
- 검색을 새 창(QInputDialog) 띄우지 않고, "현재 화면"의 검색 입력칸(QLineEdit)에서 수행

필수/권장 UI
- 프로젝트 탭(또는 메인윈도우)에 QLineEdit 하나를 두고 objectName을 아래 중 하나로 맞추면 자동 연결됨:
  1) line_project_search   (권장)
  2) edit_project_search
  3) le_project_search
  4) txt_project_search
  5) lineEdit_project_search

동작
- btn_project_export 클릭 -> 검색 입력칸의 텍스트로 필터 적용
- 검색 입력칸에서 Enter -> 동일하게 필터 적용
- (옵션) 실시간 필터: textChanged에도 연결 (기본 ON)
- 검색어가 비어있으면 전체 보기
- btn_project_prev / btn_project_next -> 페이지 이동
"""

from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# ---------------------------
# Qt binding import (PyQt5 / PySide6 / PySide2)
# ---------------------------
def _import_qt():
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets, "PyQt5"
    except Exception:
        pass
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets, "PySide6"
    except Exception:
        pass
    from PySide2 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets, "PySide2"

QtCore, QtGui, QtWidgets, QT_BINDING = _import_qt()

# ---------------------------
# Utils
# ---------------------------
def sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\\\|?*\\n\\r\\t]', "_", name)
    name = re.sub(r"\\s+", " ", name).strip()
    name = name.rstrip(". ").strip()
    return name

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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

def _is_clickable_btn(obj) -> bool:
    return obj is not None and hasattr(obj, "clicked") and hasattr(obj.clicked, "connect")

def _is_line_edit(obj) -> bool:
    return obj is not None and isinstance(obj, QtWidgets.QLineEdit)

# ---------------------------
# Dialog (프로젝트 추가)
# ---------------------------
@dataclass
class ProjectRow:
    project_name: str = ""
    process: str = ""
    target_product: str = ""
    judge_model: str = ""
    judge_data: str = ""
    measure_date: str = ""
    description: str = ""

    def to_table_values(self) -> Dict[str, str]:
        return {
            "프로젝트명": self.project_name,
            "공정": self.process,
            "대상제품": self.target_product,
            "판정모델": self.judge_model,
            "판정데이터": self.judge_data,
            "측정날짜": self.measure_date,
            "설명": self.description,
        }

class ProjectAddDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("프로젝트 추가")
        self.setModal(True)
        self.resize(520, 420)

        self.edit_project_name = QtWidgets.QLineEdit()
        self.edit_project_name.setPlaceholderText("예) Line1_2025-12-16")

        self.edit_process = QtWidgets.QLineEdit()
        self.edit_process.setPlaceholderText("예) 라인1 / 공정A")

        self.edit_target_product = QtWidgets.QLineEdit()
        self.edit_target_product.setPlaceholderText("예) RDB-63FL")

        self.edit_judge_model = QtWidgets.QLineEdit()
        self.edit_judge_model.setPlaceholderText("예) CNN1D_v003 (비워도 됨)")

        self.edit_judge_data = QtWidgets.QLineEdit()
        self.edit_judge_data.setPlaceholderText("예) 2025-12-16.csv 또는 2025-12-01~12-16")

        self.edit_measure_date = QtWidgets.QLineEdit()
        self.edit_measure_date.setPlaceholderText("예) 2025-12-16 (비워도 됨)")

        self.edit_desc = QtWidgets.QPlainTextEdit()
        self.edit_desc.setPlaceholderText("설명(선택)")

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.addRow("프로젝트명*", self.edit_project_name)
        form.addRow("공정", self.edit_process)
        form.addRow("대상제품", self.edit_target_product)
        form.addRow("판정모델", self.edit_judge_model)
        form.addRow("판정데이터", self.edit_judge_data)
        form.addRow("측정날짜", self.edit_measure_date)
        form.addRow("설명", self.edit_desc)

        self.btn_save = QtWidgets.QPushButton("저장")
        self.btn_cancel = QtWidgets.QPushButton("취소")
        self.btn_save.setDefault(True)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_save)

        root = QtWidgets.QVBoxLayout()
        root.addLayout(form)
        root.addSpacing(8)
        root.addLayout(btns)
        self.setLayout(root)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)

        self._result: Optional[ProjectRow] = None

    def _on_save(self):
        name = (self.edit_project_name.text() or "").strip()
        safe = sanitize_filename(name)
        if not safe:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "프로젝트명은 비워둘 수 없습니다.")
            return

        self._result = ProjectRow(
            project_name=name.strip(),
            process=(self.edit_process.text() or "").strip(),
            target_product=(self.edit_target_product.text() or "").strip(),
            judge_model=(self.edit_judge_model.text() or "").strip(),
            judge_data=(self.edit_judge_data.text() or "").strip(),
            measure_date=(self.edit_measure_date.text() or "").strip(),
            description=(self.edit_desc.toPlainText() or "").strip(),
        )
        self.accept()

    def get_result(self) -> Optional[ProjectRow]:
        return self._result

# ---------------------------
# Controller
# ---------------------------
class ProjectController(QtCore.QObject):
    """
    ✅ 권장 호출 (키워드 인자)
    ProjectController(
        main_window=win,
        table_projects=table_projects,
        btn_project_add=btn_project_add,
        btn_project_export=btn_project_export,
        btn_project_prev=btn_project_prev,
        btn_project_next=btn_project_next,
        line_project_search=line_project_search,     # <- 검색 입력칸(있으면)
        projects_dir=os.path.join(os.getcwd(), "projects"),
        page_size=30,
        respect_ui_settings=True  # UI 설정 존중 (기본값)
    )

    ✅ 기존 호환 호출
    - ProjectController(win, table, btn_add, projects_dir)
    """

    def __init__(
        self,
        main_window,
        table_projects,
        btn_project_add=None,
        btn_project_export=None,
        btn_project_prev=None,
        btn_project_next=None,
        line_project_search=None,
        projects_dir: str = None,
        page_size: int = 30,
        realtime_search: bool = True,
        respect_ui_settings: bool = True,  # 🎯 새 옵션
        logger = None  # 🎯 이벤트 로거
    ):
        super().__init__(main_window)

        # backward compatibility shim: (win, table, btn_add, projects_dir)
        if isinstance(btn_project_export, str) and projects_dir is None:
            projects_dir = btn_project_export
            btn_project_export = None
        if isinstance(btn_project_add, str) and projects_dir is None:
            projects_dir = btn_project_add
            btn_project_add = None

        self.main_window = main_window
        self.table = table_projects

        # buttons
        self.btn_add = btn_project_add or getattr(main_window, "btn_project_add", None)
        self.btn_export = btn_project_export or getattr(main_window, "btn_project_export", None)
        self.btn_prev = btn_project_prev or getattr(main_window, "btn_project_prev", None)
        self.btn_next = btn_project_next or getattr(main_window, "btn_project_next", None)

        # search line edit (no popup)
        self.line_search = line_project_search or self._find_search_lineedit()

        self.projects_dir = projects_dir or os.path.join(os.getcwd(), "projects")
        ensure_dir(self.projects_dir)

        self.page_size = max(1, int(page_size))
        self.page_index = 0
        self.realtime_search = bool(realtime_search)
        self.respect_ui_settings = bool(respect_ui_settings)  # 🎯 UI 설정 존중 플래그
        self.logger = logger  # 🎯 이벤트 로거

        self.current_project_path: Optional[str] = None
        self.current_project_data: Optional[Dict[str, Any]] = None

        self._all_items: List[Dict[str, Any]] = []
        self._filtered_items: List[Dict[str, Any]] = []
        self._current_keyword: str = ""

        self._setup_table()
        self.reload_from_disk()

        # 🎯 UI 설정을 존중하지 않을 때만 타이머로 컬럼 너비 적용
        if not self.respect_ui_settings:
            QtCore.QTimer.singleShot(0, self._apply_column_widths)

        # connect signals
        if _is_clickable_btn(self.btn_add):
            self.btn_add.clicked.connect(self.open_add_dialog)
        if _is_clickable_btn(self.btn_export):
            self.btn_export.clicked.connect(self.on_search_clicked)  # 현재 화면에서 검색
        if _is_clickable_btn(self.btn_prev):
            self.btn_prev.clicked.connect(self.go_prev_page)
        if _is_clickable_btn(self.btn_next):
            self.btn_next.clicked.connect(self.go_next_page)

        self._bind_search_lineedit()

        self.table.cellClicked.connect(self.on_table_clicked)

    # ---- search widgets ----
    def _find_search_lineedit(self) -> Optional[QtWidgets.QLineEdit]:
        """
        UI에서 검색 입력칸을 자동으로 찾음.
        권장 objectName: line_project_search
        """
        candidates = [
            "line_project_search",
            "edit_project_search",
            "le_project_search",
            "txt_project_search",
            "lineEdit_project_search",
        ]
        for name in candidates:
            w = getattr(self.main_window, name, None)
            if _is_line_edit(w):
                return w
            try:
                w2 = self.main_window.findChild(QtWidgets.QLineEdit, name)
                if _is_line_edit(w2):
                    return w2
            except Exception:
                pass

        # 마지막 안전장치: objectName에 "search" 포함된 QLineEdit
        try:
            for w in self.main_window.findChildren(QtWidgets.QLineEdit):
                on = (w.objectName() or "").lower()
                if "search" in on and ("project" in on or "proj" in on):
                    return w
        except Exception:
            pass
        return None

    def _bind_search_lineedit(self):
        if not _is_line_edit(self.line_search):
            self._set_status("검색 입력칸(QLineEdit)을 찾지 못했습니다. (line_project_search 권장)")
            return

        # Enter 키로 검색
        try:
            self.line_search.returnPressed.connect(self.on_search_clicked)
        except Exception:
            pass

        # 실시간 검색(원치 않으면 realtime_search=False로)
        if self.realtime_search:
            try:
                self.line_search.textChanged.connect(self.on_search_text_changed)
            except Exception:
                pass

    def on_search_text_changed(self, _text: str):
        # 타이핑마다 바로 필터링
        self.on_search_clicked()

    # ---- header helpers ----
    def _header_text(self, col: int) -> str:
        item = self.table.horizontalHeaderItem(col)
        if item is not None and item.text():
            return item.text()
        try:
            t = self.table.model().headerData(col, QtCore.Qt.Horizontal)
            return str(t) if t is not None else f"COL{col}"
        except Exception:
            return f"COL{col}"

    def _headers(self) -> Tuple[str, ...]:
        return tuple(self._header_text(c) for c in range(self.table.columnCount()))

    # ---- table setup ----
    def _setup_table(self):
        """
        테이블 기본 설정
        🎯 UI 설정을 존중할 때는 필수 설정만 적용
        """
        self.table.setRowCount(0)
        
        # 🎯 respect_ui_settings=False일 때만 강제 설정
        if not self.respect_ui_settings:
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.setAlternatingRowColors(True)
            self._apply_column_widths()

    def _apply_column_widths(self):
        """
        컬럼 너비 조정
        🎯 UI 설정을 존중할 때는 아무것도 하지 않음
        """
        if self.respect_ui_settings:
            # UI 파일에서 설정한 값 그대로 사용
            return
        
        # 🎯 respect_ui_settings=False일 때만 강제 적용
        header = self.table.horizontalHeader()
        try:
            header.setStretchLastSection(False)
        except Exception:
            pass
        try:
            header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        except Exception:
            pass

        headers = self._headers()

        if "NO" in headers:
            idx_no = headers.index("NO")
            try:
                header.setSectionResizeMode(idx_no, QtWidgets.QHeaderView.Fixed)
            except Exception:
                pass
            self.table.setColumnWidth(idx_no, 60)
            try:
                header.resizeSection(idx_no, 60)
            except Exception:
                pass

        if "설명" in headers:
            idx = headers.index("설명")
            try:
                header.setSectionResizeMode(idx, QtWidgets.QHeaderView.Fixed)
            except Exception:
                pass
            self.table.setColumnWidth(idx, 420)
            try:
                header.resizeSection(idx, 420)
            except Exception:
                pass

    # ---- disk loading ----
    def reload_from_disk(self):
        files = []
        try:
            for fn in os.listdir(self.projects_dir):
                if fn.lower().endswith(".json"):
                    files.append(os.path.join(self.projects_dir, fn))
        except Exception:
            files = []

        items = []
        for p in files:
            data = read_json(p)
            if not isinstance(data, dict):
                continue
            proj = data.get("project")
            if not isinstance(proj, dict):
                continue
            items.append({
                "path": p,
                "created_at": safe_str(data.get("created_at", "")),
                "project": proj,
                "data": data,
            })

        items.sort(key=lambda it: (it.get("created_at", ""), os.path.basename(it.get("path", "")).lower()))
        self._all_items = items
        self.apply_filter(self._current_keyword, reset_page=True)

    # ---- filtering + pagination ----
    def apply_filter(self, keyword: str, reset_page: bool = True):
        kw = (keyword or "").strip().lower()
        self._current_keyword = keyword or ""

        if not kw:
            self._filtered_items = list(self._all_items)
        else:
            filtered = []
            for it in self._all_items:
                proj = it.get("project", {}) or {}
                hay = [os.path.basename(it.get("path", ""))]
                hay.extend(safe_str(v) for v in proj.values())
                if kw in " | ".join(hay).lower():
                    filtered.append(it)
            self._filtered_items = filtered

        if reset_page:
            self.page_index = 0
        self.render_current_page()

    def total_pages(self) -> int:
        n = len(self._filtered_items)
        return (n + self.page_size - 1) // self.page_size

    def go_prev_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.render_current_page()

    def go_next_page(self):
        if self.page_index + 1 < self.total_pages():
            self.page_index += 1
            self.render_current_page()

    def render_current_page(self):
        headers = self._headers()
        self.table.setRowCount(0)

        total = len(self._filtered_items)
        if total == 0:
            self._set_status("검색 결과가 없습니다." if self._current_keyword else "프로젝트가 없습니다.")
            # 🎯 UI 설정 존중 시 컬럼 너비 조정 안 함
            if not self.respect_ui_settings:
                self._apply_column_widths()
            return

        start = self.page_index * self.page_size
        end = min(start + self.page_size, total)
        page_items = self._filtered_items[start:end]

        for i, it in enumerate(page_items):
            r = self.table.rowCount()
            self.table.insertRow(r)

            proj = it.get("project", {}) or {}
            json_path = it.get("path", "")

            for c, h in enumerate(headers):
                if h == "NO":
                    item = QtWidgets.QTableWidgetItem(str(start + i + 1))
                else:
                    item = QtWidgets.QTableWidgetItem(safe_str(proj.get(h, "")))

                if h == "프로젝트명":
                    item.setData(QtCore.Qt.UserRole, json_path)
                self.table.setItem(r, c, item)

        pages = self.total_pages()
        self._set_status(
            f"프로젝트 {total}개: {start + 1}-{end} (페이지 {self.page_index + 1}/{max(1, pages)})"
            + (f" | 검색: '{self._current_keyword}'" if self._current_keyword.strip() else "")
        )
        
        # 🎯 UI 설정 존중 시 컬럼 너비 조정 안 함
        if not self.respect_ui_settings:
            self._apply_column_widths()

    # ---- UI actions ----
    def on_search_clicked(self):
        """
        ✅ 팝업 없이 현재 화면 QLineEdit의 텍스트로만 검색
        """
        if not _is_line_edit(self.line_search):
            self._set_status("검색 입력칸(QLineEdit)을 찾지 못했습니다. (line_project_search 권장)")
            return
        text = self.line_search.text()
        self.apply_filter(text, reset_page=True)

    def open_add_dialog(self):
        dlg = ProjectAddDialog(parent=self.main_window)
        accepted = dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()
        if accepted == QtWidgets.QDialog.Accepted:
            row = dlg.get_result()
            if row:
                self.add_project(row)

    def add_project(self, row: ProjectRow):
        project_name = row.project_name.strip()
        safe_name = sanitize_filename(project_name)
        json_path = os.path.join(self.projects_dir, f"{safe_name}.json")

        if os.path.exists(json_path):
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "중복 프로젝트",
                f"이미 같은 프로젝트명이 존재합니다.\n\n- 프로젝트명: {project_name}\n- 파일: {json_path}\n\n프로젝트명을 변경하세요."
            )
            return

        values = row.to_table_values()
        payload = {
            "schema_version": 1,
            "created_at": now_iso(),
            "project": {"NO": "", **values},
            "inference": {},
            "training": {},
            "history": []
        }
        write_json(json_path, payload)
        
        # 🎯 로그 추가
        if self.logger:
            self.logger.log_success(f"프로젝트 생성: {project_name}")

        self.reload_from_disk()

        if not self._current_keyword.strip():
            self.page_index = max(0, self.total_pages() - 1)
            self.render_current_page()

        self._set_status(f"프로젝트 생성됨: {project_name} (저장됨)")

    def on_table_clicked(self, row_idx: int, col_idx: int):
        proj_item = self._item_by_header(row_idx, "프로젝트명")
        if proj_item is None:
            return

        json_path = proj_item.data(QtCore.Qt.UserRole)
        proj_name = (proj_item.text() or "").strip()

        if not json_path:
            safe = sanitize_filename(proj_name)
            json_path = os.path.join(self.projects_dir, f"{safe}.json")

        if not os.path.exists(json_path):
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "프로젝트 파일 없음",
                f"프로젝트 JSON 파일을 찾을 수 없습니다.\n\n- 프로젝트명: {proj_name}\n- 경로: {json_path}"
            )
            return

        data = read_json(json_path)
        if not isinstance(data, dict):
            QtWidgets.QMessageBox.critical(self.main_window, "로드 실패", "프로젝트 JSON을 읽지 못했습니다.")
            return

        self.current_project_path = json_path
        self.current_project_data = data
        self._set_status(f"현재 프로젝트: {proj_name}  (로드됨)")
        
        # 🎯 로그 추가
        if self.logger:
            self.logger.log_file(f"프로젝트 활성화: {proj_name}")

    def _item_by_header(self, row_idx: int, header_text: str):
        headers = self._headers()
        try:
            col = headers.index(header_text)
        except ValueError:
            return None
        return self.table.item(row_idx, col)

    def _set_status(self, msg: str):
        try:
            sb = self.main_window.statusBar()
            if sb:
                sb.showMessage(msg, 5000)
        except Exception:
            pass
    # ============================================
    # 🎯 프로젝트 자동 업데이트 메소드
    # ============================================
    def update_judge_model(self, model_path: str, model_info: Dict[str, Any]):
        """
        판정 모델이 변경되었을 때 호출
        활성화된 프로젝트 JSON 파일에 모델 정보 저장
        """
        if not self.current_project_path:
            return
        
        if not os.path.exists(self.current_project_path):
            return
        
        try:
            # 기존 데이터 읽기
            data = read_json(self.current_project_path)
            if not isinstance(data, dict):
                return
            
            # project 섹션 업데이트
            if "project" not in data:
                data["project"] = {}
            
            data["project"]["판정모델"] = os.path.basename(model_path)
            
            # inference 섹션 업데이트 (상세 정보 저장)
            if "inference" not in data:
                data["inference"] = {}
            
            data["inference"]["model_path"] = model_path
            data["inference"]["model_info"] = model_info
            data["inference"]["model_updated_at"] = now_iso()
            
            # history 추가
            if "history" not in data:
                data["history"] = []
            
            data["history"].append({
                "action": "model_changed",
                "model_path": model_path,
                "timestamp": now_iso()
            })
            
            # 파일 저장
            write_json(self.current_project_path, data)
            
            # 현재 데이터 갱신
            self.current_project_data = data
            
            # 테이블 다시 로드
            self.reload_from_disk()
            
            self._set_status(f"프로젝트 업데이트: 판정모델 = {os.path.basename(model_path)}")
            
            # 🎯 로그 추가
            if self.logger:
                self.logger.log_model(f"판정모델 변경: {os.path.basename(model_path)}")
            
        except Exception as e:
            print(f"⚠️ 프로젝트 업데이트 실패 (모델): {e}")
    
    def update_judge_data(self, csv_path: str):
        """
        판정 데이터가 변경되었을 때 호출
        활성화된 프로젝트 JSON 파일에 데이터 정보 저장
        """
        if not self.current_project_path:
            return
        
        if not os.path.exists(self.current_project_path):
            return
        
        try:
            # 기존 데이터 읽기
            data = read_json(self.current_project_path)
            if not isinstance(data, dict):
                return
            
            # project 섹션 업데이트
            if "project" not in data:
                data["project"] = {}
            
            data["project"]["판정데이터"] = os.path.basename(csv_path)
            
            # inference 섹션 업데이트 (상세 정보 저장)
            if "inference" not in data:
                data["inference"] = {}
            
            data["inference"]["data_path"] = csv_path
            data["inference"]["data_updated_at"] = now_iso()
            
            # history 추가
            if "history" not in data:
                data["history"] = []
            
            data["history"].append({
                "action": "data_changed",
                "data_path": csv_path,
                "timestamp": now_iso()
            })
            
            # 파일 저장
            write_json(self.current_project_path, data)
            
            # 현재 데이터 갱신
            self.current_project_data = data
            
            # 테이블 다시 로드
            self.reload_from_disk()
            
            self._set_status(f"프로젝트 업데이트: 판정데이터 = {os.path.basename(csv_path)}")
            
            # 🎯 로그 추가
            if self.logger:
                self.logger.log_data(f"판정데이터 변경: {os.path.basename(csv_path)}")
            
        except Exception as e:
            print(f"⚠️ 프로젝트 업데이트 실패 (데이터): {e}")