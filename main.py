# -*- coding: utf-8 -*-
"""main.py"""

import os
import sys
from PyQt5 import QtWidgets, uic

# 🎯 QRC 리소스 파일 import (이미지 표시용)
try:
    import Project_image_rc  # pyrcc5로 컴파일된 파일
except ImportError:
    print("⚠️ Project_image_rc.py를 찾을 수 없습니다. QRC 리소스를 컴파일하세요:")
    print("   pyrcc5 -o Project_image_rc.py Project_image.qrc")

from project import ProjectController
from judge_modelload import JudgeModelController
from judge_dataload import JudgeDataController
from judge_inference_integrated import JudgeInferenceController, SeedEnsemble, MultiModelEnsemble, SimpleJudgeController
from training_1 import TrainingController
from training_2 import DLTrainingController
from training_project import TrainingProjectController
# from simple_judge_controller import SimpleJudgeController  # judge_inference_integrated에 통합됨  # 간단한 판정 컨트롤러
from logger import EventLogger
import PoscoFutureM_logo
import Postech_logo_rc
import file_open_rc, file_save_rc
from PyQt5 import QtWidgets, QtCore  # PySide2면 PySide2로

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
# pickle 호환성을 위해 __main__ 네임스페이스에 추가
__main__ = sys.modules['__main__']
__main__.SeedEnsemble = SeedEnsemble
__main__.MultiModelEnsemble = MultiModelEnsemble


def _get_root_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def load_mainwindow(ui_path: str) -> QtWidgets.QMainWindow:
    win = QtWidgets.QMainWindow()
    uic.loadUi(ui_path, win)
    return win


def main():
    
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)

    root_dir = _get_root_dir()
    ui_path = os.path.join(root_dir, "Mainwindow.ui")

    if not os.path.exists(ui_path):
        QtWidgets.QMessageBox.critical(
            None,
            "파일 없음",
            f"Mainwindow.ui 파일을 찾을 수 없습니다.\n경로: {ui_path}",
        )
        sys.exit(1)

    win = load_mainwindow(ui_path)

    # ============================================
    # 1. 프로젝트 관리 컨트롤러
    # ============================================
    table_projects = win.findChild(QtWidgets.QTableWidget, "table_projects")
    btn_project_add = win.findChild(QtWidgets.QPushButton, "btn_project_add")
    btn_project_export = win.findChild(QtWidgets.QPushButton, "btn_project_export")
    btn_project_prev = win.findChild(QtWidgets.QPushButton, "btn_project_prev")
    btn_project_next = win.findChild(QtWidgets.QPushButton, "btn_project_next")
    line_project_search = win.findChild(QtWidgets.QLineEdit, "line_project_search")

    projects_dir = os.path.join(root_dir, "projects")
    models_dir = os.path.join(root_dir, "models")
    data_dir = os.path.join(root_dir, "data")


    # ============================================
    # 🎯 이벤트 로거 생성
    # ============================================
    # QTextEdit 또는 QPlainTextEdit 찾기
    text_log_project = win.findChild(QtWidgets.QTextEdit, "text_log_project") or \
                       win.findChild(QtWidgets.QPlainTextEdit, "text_log_project")
    text_log_judge = win.findChild(QtWidgets.QTextEdit, "text_log_judge") or \
                     win.findChild(QtWidgets.QPlainTextEdit, "text_log_judge")
    text_log_train = win.findChild(QtWidgets.QTextEdit, "text_log_train") or \
                     win.findChild(QtWidgets.QPlainTextEdit, "text_log_train")
    
    project_logger = EventLogger(text_log_project)
    judge_logger = EventLogger(text_log_judge)
    train_logger = EventLogger(text_log_train)
    
    print("✅ EventLogger 초기화 완료")
    
    # ProjectController 초기화 (table_projects가 있을 때만)
    if table_projects:
        win._proj_ctrl = ProjectController(
            main_window=win,
            table_projects=table_projects,
            btn_project_add=btn_project_add,
            btn_project_export=btn_project_export,
            btn_project_prev=btn_project_prev,
            btn_project_next=btn_project_next,
            line_project_search=line_project_search,
            projects_dir=projects_dir,
            page_size=30,
            realtime_search=True,
            logger=project_logger,
            respect_ui_settings=True,
        )
        table_projects.setColumnWidth(0, 60)    # NO
        table_projects.setColumnWidth(2, 100)   # 공정
        table_projects.setColumnWidth(3, 120)   # 대상제품
        table_projects.setColumnWidth(4, 250)   # 판정모델
        table_projects.setColumnWidth(5, 250)   # 판정데이터
        print("✅ ProjectController 초기화 완료")
    else:
        print("⚠️ table_projects를 찾을 수 없어 ProjectController를 초기화하지 않습니다.")
    # ============================================
    # 2. Judge 모델 로드 컨트롤러
    # ============================================
    print("\n🔍 Judge 모델 위젯 검색 중...")
    btn_judge_data_browse_2 = win.findChild(QtWidgets.QPushButton, "btn_judge_data_browse_2")
    le_judge_data_path_2 = win.findChild(QtWidgets.QLineEdit, "le_judge_data_path_2")
    
    print(f"   - btn_judge_data_browse_2: {'✅' if btn_judge_data_browse_2 is not None else '❌'}")
    print(f"   - le_judge_data_path_2: {'✅' if le_judge_data_path_2 is not None else '❌'}")

    if btn_judge_data_browse_2:        
        win._judge_model_ctrl = JudgeModelController(
            main_window=win,
            btn_judge_data_browse_2=btn_judge_data_browse_2,
            le_judge_data_path_2=le_judge_data_path_2,
            initial_model_dir=models_dir,
            logger=judge_logger
        )
        print("✅ JudgeModelController 초기화 완료")
    else:
        print("⚠️ btn_judge_data_browse_2를 찾을 수 없어 JudgeModelController를 초기화하지 않습니다.")
        print("   💡 UI 파일에서 버튼 이름 확인 필요!")

    # ============================================
    # 3. Judge 데이터 로드 컨트롤러
    # ============================================
    btn_judge_data_browse = win.findChild(QtWidgets.QPushButton, "btn_judge_data_browse")
    le_judge_data_path = win.findChild(QtWidgets.QLineEdit, "le_judge_data_path")
    table_judge_data = win.findChild(QtWidgets.QTableWidget, "table_judge_data")
    le_judge_index_1 = win.findChild(QtWidgets.QLineEdit, "le_judge_index_1")
    le_judge_index_2 = win.findChild(QtWidgets.QLineEdit, "le_judge_index_2")
    le_judge_index_3 = win.findChild(QtWidgets.QLineEdit, "le_judge_index_3")
    le_judge_index_4 = win.findChild(QtWidgets.QLineEdit, "le_judge_index_4")
    btn_judge_visualize = win.findChild(QtWidgets.QPushButton, "btn_judge_visualize")
    btn_project_prev_2 = win.findChild(QtWidgets.QPushButton, "btn_project_prev_2")
    btn_project_next_2 = win.findChild(QtWidgets.QPushButton, "btn_project_next_2")

    if btn_judge_data_browse:
        win._judge_data_ctrl = JudgeDataController(
            main_window=win,
            btn_judge_data_browse=btn_judge_data_browse,
            le_judge_data_path=le_judge_data_path,
            table_judge_data=table_judge_data,
            le_judge_index_1=le_judge_index_1,
            le_judge_index_2=le_judge_index_2,
            le_judge_index_3=le_judge_index_3,
            le_judge_index_4=le_judge_index_4,
            btn_judge_visualize=btn_judge_visualize,
            btn_project_prev_2=btn_project_prev_2,
            btn_project_next_2=btn_project_next_2,
            initial_data_dir=data_dir,
            logger=judge_logger
        )
        print("✅ JudgeDataController 초기화 완료")
    else:
        print("⚠️ btn_judge_data_browse를 찾을 수 없어 JudgeDataController를 초기화하지 않습니다.")

    # ============================================
    # 4. Judge 추론 컨트롤러
    # ============================================
    btn_judge_one = win.findChild(QtWidgets.QPushButton, "btn_judge_one")
    table_judge_stats_3 = win.findChild(QtWidgets.QTableWidget, "table_judge_stats_3")
    label_judge_chart = win.findChild(QtWidgets.QLabel, "label_judge_chart")
    table_judge_stats_2 = win.findChild(QtWidgets.QTableWidget, "table_judge_stats_2")
    label_judge_chart_placeholder = win.findChild(QtWidgets.QLabel, "label_judge_chart_placeholde")
    table_judge_stats = win.findChild(QtWidgets.QTableWidget, "table_judge_stats")
    
    # 프로젝트 정보 표시용 textBrowser들
    textBrowser_project = win.findChild(QtWidgets.QTextBrowser, "textBrowser_project")
    textBrowser_project_date = win.findChild(QtWidgets.QTextBrowser, "textBrowser_project_date")
    textBrowser_process = win.findChild(QtWidgets.QTextBrowser, "textBrowser_process")
    textBrowser_detail = None  # UI에 없음
    textBrowser_model = win.findChild(QtWidgets.QTextBrowser, "textBrowser_model")
    textBrowser_purpose = None  # UI에 없음
    textBrowser_product = win.findChild(QtWidgets.QTextBrowser, "textBrowser_product")
    textBrowser_model_data = win.findChild(QtWidgets.QTextBrowser, "textBrowser_model_data")  # date → data!
    textBrowser_traindata = win.findChild(QtWidgets.QTextBrowser, "textBrowser_traindata")
    textBrowser_validdata = win.findChild(QtWidgets.QTextBrowser, "textBrowser_validdata")
    
    # 판정 결과 표시용 textBrowser들
    textBrowser_13 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_13")
    textBrowser_15 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_15")
    textBrowser_17 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_17")
    textBrowser_34 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_34")
    
    # 불량 제품 리스트 위젯 (UI에 12개 있음)
    textEdit_product1 = win.findChild(QtWidgets.QTextEdit, "textEdit_product1")
    textEdit_product2 = win.findChild(QtWidgets.QTextEdit, "textEdit_product2")
    textEdit_product3 = win.findChild(QtWidgets.QTextEdit, "textEdit_product3")
    textEdit_product4 = win.findChild(QtWidgets.QTextEdit, "textEdit_product4")
    textEdit_product5 = win.findChild(QtWidgets.QTextEdit, "textEdit_product5")
    textEdit_product6 = win.findChild(QtWidgets.QTextEdit, "textEdit_product6")
    textEdit_product7 = win.findChild(QtWidgets.QTextEdit, "textEdit_product7")
    textEdit_product8 = win.findChild(QtWidgets.QTextEdit, "textEdit_product8")
    textEdit_product9 = win.findChild(QtWidgets.QTextEdit, "textEdit_product9")
    textEdit_product10 = win.findChild(QtWidgets.QTextEdit, "textEdit_product10")
    textEdit_product11 = win.findChild(QtWidgets.QTextEdit, "textEdit_product11")
    textEdit_product12 = win.findChild(QtWidgets.QTextEdit, "textEdit_product12")
    
    # 페이지 네비게이션 버튼
    btn_product_prev = win.findChild(QtWidgets.QPushButton, "btn_product_prev")
    btn_product_next = win.findChild(QtWidgets.QPushButton, "btn_product_next")
    
    # 판정 근거 표시용 textBrowser (groupBox_3 안)
    textBrowser_detail_name = win.findChild(QtWidgets.QTextBrowser, "textBrowser")
    textBrowser_detail_score = win.findChild(QtWidgets.QTextBrowser, "textBrowser_2")
    textBrowser_detail_reason1 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_3")
    textBrowser_detail_reason2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_4")
    
    # 누적 결과용
    textBrowser_32 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_32")
    textBrowser_30 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_30")
    textBrowser_31 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_31")
    textBrowser_37 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_37")
    label_judge_chart_placeholder_7 = win.findChild(QtWidgets.QLabel, "label_judge_chart_placeholde_7")
    
    print(f"\n   📋 프로젝트 정보 표시용 textBrowser:")
    print(f"   - textBrowser_project: {'✅' if textBrowser_project is not None else '❌'}")
    print(f"   - textBrowser_project_date: {'✅' if textBrowser_project_date is not None else '❌'}")
    print(f"   - textBrowser_process: {'✅' if textBrowser_process is not None else '❌'}")
    print(f"   - textBrowser_detail: ❌ (UI에 없음)")
    print(f"   - textBrowser_model: {'✅' if textBrowser_model is not None else '❌'}")
    print(f"   - textBrowser_purpose: ❌ (UI에 없음)")
    print(f"   - textBrowser_product: {'✅' if textBrowser_product is not None else '❌'}")
    print(f"   - textBrowser_model_data (배포날짜): {'✅' if textBrowser_model_data is not None else '❌'}")
    print(f"   - textBrowser_traindata: {'✅' if textBrowser_traindata is not None else '❌'}")
    print(f"   - textBrowser_validdata: {'✅' if textBrowser_validdata is not None else '❌'}")
    
    print(f"\n   📊 판정 결과 표시용 textBrowser:")
    print(f"   - textBrowser_13 (전체): {'✅' if textBrowser_13 is not None else '❌'}")
    print(f"   - textBrowser_15 (정상): {'✅' if textBrowser_15 is not None else '❌'}")
    print(f"   - textBrowser_17 (불량): {'✅' if textBrowser_17 is not None else '❌'}")
    print(f"   - textBrowser_34 (불량률): {'✅' if textBrowser_34 is not None else '❌'}")
    
    print(f"\n   📦 불량 제품 리스트:")
    print(f"   - textEdit_product1~3: {'✅' if textEdit_product1 is not None else '❌'}")
    print(f"   - btn_product_prev: {'✅' if btn_product_prev is not None else '❌'}")
    print(f"   - btn_product_next: {'✅' if btn_product_next is not None else '❌'}")
    print(f"   - textBrowser_detail_name: {'✅' if textBrowser_detail_name is not None else '❌'}")
    
    print(f"\n   🔘 판정 버튼:")
    print(f"   - btn_judge_one: {'✅' if btn_judge_one is not None else '❌'}")
    print("")

    # JudgeModelController가 없어도 JudgeInferenceController 초기화
    if btn_judge_one and hasattr(win, '_judge_data_ctrl'):
        results_dir = os.path.join(root_dir, "Results")
        
        # JudgeModelController가 있으면 사용, 없으면 None
        judge_model_ctrl = win._judge_model_ctrl if hasattr(win, '_judge_model_ctrl') else None
        
        win._judge_infer_ctrl = JudgeInferenceController(
            main_window=win,
            judge_model_ctrl=judge_model_ctrl,
            judge_data_ctrl=win._judge_data_ctrl,
            btn_judge_one=btn_judge_one,
            text_judge_summary=table_judge_stats_3,
            label_judge_chart=label_judge_chart,
            table_judge_stats_2=table_judge_stats_2,
            label_judge_chart_placeholder=label_judge_chart_placeholder,
            table_judge_stats=table_judge_stats,
            # 프로젝트 정보 표시용
            textBrowser_project=textBrowser_project,
            textBrowser_project_date=textBrowser_project_date,
            textBrowser_process=textBrowser_process,
            textBrowser_detail=textBrowser_detail,  # None
            textBrowser_model=textBrowser_model,
            textBrowser_purpose=textBrowser_purpose,  # None
            textBrowser_product=textBrowser_product,
            textBrowser_model_date=textBrowser_model_data,  # 🆕 배포날짜 (model_data → model_date로 매핑)
            textBrowser_traindata=textBrowser_traindata,
            textBrowser_validdata=textBrowser_validdata,
            # 판정 결과 표시용 (배치 단위)
            textBrowser_13=textBrowser_13,
            textBrowser_15=textBrowser_15,
            textBrowser_17=textBrowser_17,
            textBrowser_34=textBrowser_34,
            # 누적 통계 표시용 (오늘 전체)
            textBrowser_32=textBrowser_32,  # 누적 전체 제품 수
            textBrowser_30=textBrowser_30,  # 누적 정상 제품 수
            textBrowser_31=textBrowser_31,  # 누적 불량 제품 수
            textBrowser_37=textBrowser_37,  # 누적 불량률
            label_judge_chart_placeholder_7=label_judge_chart_placeholder_7,  # 누적 도넛 그래프
            # 불량 제품 리스트
            textEdit_product1=textEdit_product1,
            textEdit_product2=textEdit_product2,
            textEdit_product3=textEdit_product3,
            textEdit_product4=textEdit_product4,
            textEdit_product5=textEdit_product5,
            textEdit_product6=textEdit_product6,
            textEdit_product7=textEdit_product7,
            textEdit_product8=textEdit_product8,
            textEdit_product9=textEdit_product9,
            textEdit_product10=textEdit_product10,
            textEdit_product11=textEdit_product11,
            textEdit_product12=textEdit_product12,
            btn_product_prev=btn_product_prev,
            btn_product_next=btn_product_next,
            results_dir=results_dir,
            logger=judge_logger
        )
        print("✅ JudgeInferenceController 초기화 완료")
    # elif btn_judge_one and hasattr(win, '_judge_data_ctrl'):
    #     # JudgeInferenceController 사용하므로 SimpleJudgeController는 비활성화
    #     print("⚠️ JudgeModelController 없음 - SimpleJudgeController 사용")
    #     ...
    else:
        print("⚠️ Judge 추론 컨트롤러를 초기화하지 않습니다.")

    # ============================================
    # 5. TrainingProject 컨트롤러 (신규)
    # ============================================
    print("\n🔍 TrainingProject UI 위젯 검색 중...")
    
    # 1. stack_pages 찾기
    stack_pages = win.findChild(QtWidgets.QStackedWidget, "stack_pages")
    print(f"   - stack_pages: {'✅' if stack_pages else '❌'}")
    
    # 2. page_project 찾기
    page_project = None
    if stack_pages:
        for i in range(stack_pages.count()):
            widget = stack_pages.widget(i)
            if widget and widget.objectName() == "page_project":
                page_project = widget
                print(f"   - page_project: ✅ 찾음 (index={i})")
                break
    
    if not page_project:
        page_project = win.findChild(QtWidgets.QWidget, "page_project")
        print(f"   - page_project: {'✅ 직접 찾음' if page_project else '❌ 못찾음'}")
    
    # 3. 위젯들 찾기 (page_project 안에서)
    list_train_projects = None
    le_train_project_name_3 = None
    le_train_project_date = None
    le_train_project_process = None
    le_train_project_process_2 = None
    le_train_model_name_ml_2 = None
    le_train_purpose = None
    le_train_algorithm = None
    le_train_purpose_product = None  # 대상제품 (algorithm에서 변경)
    le_model_data = None  # 🆕 배포날짜 (모델 파일 날짜)
    le_train_traincsv_ml_2 = None
    le_train_validcsv_ml_2 = None
    
    # 버튼들
    btn_project_new = None
    btn_train_traincsv_load_ml_3 = None
    btn_train_traincsv_browse_ml_2 = None
    btn_train_validcsv_browse_ml_2 = None
    pushButton_3 = None
    pushButton_4 = None
    pushButton_6 = None  # 🆕 프로젝트 저장 버튼
    pushButton_7 = None  # 🆕 학습→프로젝트 이동 버튼
    pushButton_8 = None  # 🆕 판정→프로젝트 이동 버튼
    
    if page_project:
        list_train_projects = page_project.findChild(QtWidgets.QListWidget, "list_train_projects")
        le_train_project_name_3 = page_project.findChild(QtWidgets.QLineEdit, "le_train_project_name_3")
        le_train_project_date = page_project.findChild(QtWidgets.QLineEdit, "le_train_project_date")
        le_train_project_process = page_project.findChild(QtWidgets.QLineEdit, "le_train_project_process")
        le_train_model_name_ml_2 = page_project.findChild(QtWidgets.QLineEdit, "le_train_model_name_ml_2")
        le_train_purpose = page_project.findChild(QtWidgets.QLineEdit, "le_train_purpose")
        le_train_algorithm = page_project.findChild(QtWidgets.QLineEdit, "le_train_algorithm")
        le_train_purpose_product = page_project.findChild(QtWidgets.QLineEdit, "le_train_purpose_product")
        le_model_data = page_project.findChild(QtWidgets.QLineEdit, "le_model_data")  # 🆕 배포날짜
        le_train_traincsv_ml_2 = page_project.findChild(QtWidgets.QLineEdit, "le_train_traincsv_ml_2")
        le_train_validcsv_ml_2 = page_project.findChild(QtWidgets.QLineEdit, "le_train_validcsv_ml_2")
        
        # 버튼들 찾기
        btn_project_new = page_project.findChild(QtWidgets.QPushButton, "btn_project_new")
        btn_train_traincsv_load_ml_3 = page_project.findChild(QtWidgets.QPushButton, "btn_train_traincsv_load_ml_3")
        btn_train_traincsv_browse_ml_2 = page_project.findChild(QtWidgets.QPushButton, "btn_train_traincsv_browse_ml_2")
        btn_train_validcsv_browse_ml_2 = page_project.findChild(QtWidgets.QPushButton, "btn_train_validcsv_browse_ml_2")
        pushButton_3 = page_project.findChild(QtWidgets.QPushButton, "pushButton_3")
        pushButton_4 = page_project.findChild(QtWidgets.QPushButton, "pushButton_4")
        pushButton_6 = page_project.findChild(QtWidgets.QPushButton, "pushButton_6")  # 🆕
    
    # 🆕 page_judge와 page_train에서도 버튼 찾기
    stack_pages = win.findChild(QtWidgets.QStackedWidget, "stack_pages")
    if stack_pages:
        for i in range(stack_pages.count()):
            widget = stack_pages.widget(i)
            if widget:
                if widget.objectName() == "page_judge":
                    page_judge = widget
                    pushButton_8 = page_judge.findChild(QtWidgets.QPushButton, "pushButton_8")
                elif widget.objectName() == "page_train":
                    page_train = widget
                    pushButton_7 = page_train.findChild(QtWidgets.QPushButton, "pushButton_7")
    
    # 4. 결과 출력
    print(f"   - list_train_projects: {'✅' if list_train_projects is not None else '❌'} (타입: {type(list_train_projects)}, 값: {list_train_projects})")
    print(f"   - le_train_project_name_3: {'✅' if le_train_project_name_3 is not None else '❌'}")
    print(f"   - le_train_project_date: {'✅' if le_train_project_date is not None else '❌'}")
    print(f"   - le_train_project_process: {'✅' if le_train_project_process is not None else '❌'}")
    print(f"   - le_train_model_name_ml_2: {'✅' if le_train_model_name_ml_2 is not None else '❌'}")
    print(f"   - le_train_purpose: {'✅' if le_train_purpose is not None else '❌'}")
    print(f"   - le_train_algorithm: {'✅' if le_train_algorithm is not None else '❌'}")
    print(f"   - le_train_purpose_product: {'✅' if le_train_purpose_product is not None else '❌'}")
    print(f"   - le_model_data (배포날짜): {'✅' if le_model_data is not None else '❌'}")
    print(f"   - le_train_traincsv_ml_2: {'✅' if le_train_traincsv_ml_2 is not None else '❌'}")
    print(f"   - le_train_validcsv_ml_2: {'✅' if le_train_validcsv_ml_2 is not None else '❌'}")
    print(f"\n   🔘 버튼들:")
    print(f"   - btn_project_new: {'✅' if btn_project_new is not None else '❌'}")
    print(f"   - btn_train_traincsv_load_ml_3: {'✅' if btn_train_traincsv_load_ml_3 is not None else '❌'}")
    print(f"   - btn_train_traincsv_browse_ml_2: {'✅' if btn_train_traincsv_browse_ml_2 is not None else '❌'}")
    print(f"   - btn_train_validcsv_browse_ml_2: {'✅' if btn_train_validcsv_browse_ml_2 is not None else '❌'}")
    print(f"   - pushButton_3 (균열판정): {'✅' if pushButton_3 is not None else '❌'}")
    print(f"   - pushButton_4 (모델학습): {'✅' if pushButton_4 is not None else '❌'}")
    print(f"   - pushButton_6 (프로젝트저장): {'✅' if pushButton_6 is not None else '❌'}")  # 🆕
    print(f"   - pushButton_7 (학습→프로젝트): {'✅' if pushButton_7 is not None else '❌'}")  # 🆕
    print(f"   - pushButton_8 (판정→프로젝트): {'✅' if pushButton_8 is not None else '❌'}")  # 🆕
    print("")
    
    # 5. 강제로 다시 찾기 (혹시 몰라서)
    if list_train_projects is None and page_project:
        print("   ⚠️ list_train_projects가 None입니다. 다시 찾아봅니다...")
        list_train_projects = page_project.findChild(QtWidgets.QListWidget, "list_train_projects")
        print(f"      → 재검색 결과: {list_train_projects}")
    
    print(f"   🔍 최종 체크 - list_train_projects는 {'있음 ✅' if list_train_projects is not None else '없음 ❌'}")
    print("")
    
    if list_train_projects is not None:
        try:
            win._train_proj_ctrl = TrainingProjectController(
                main_window=win,
                list_train_projects=list_train_projects,
                le_train_project_name_3=le_train_project_name_3,
                le_train_project_date=le_train_project_date,
                le_train_project_process=le_train_project_process,
                le_train_project_process_2=le_model_data,  # 🆕 배포날짜 (모델 파일 날짜)
                le_train_model_name_ml_2=le_train_model_name_ml_2,
                le_train_purpose=le_train_purpose,
                le_train_purpose_product=le_train_purpose_product,  # algorithm → target_product
                le_train_deploy_date=None,  # 더 이상 사용 안 함
                le_train_traincsv_ml_2=le_train_traincsv_ml_2,
                le_train_validcsv_ml_2=le_train_validcsv_ml_2,
                btn_project_new=btn_project_new,
                btn_train_traincsv_load_ml_3=btn_train_traincsv_load_ml_3,
                btn_train_traincsv_browse_ml_2=btn_train_traincsv_browse_ml_2,
                btn_train_validcsv_browse_ml_2=btn_train_validcsv_browse_ml_2,
                pushButton_3=pushButton_3,
                pushButton_4=pushButton_4,
                pushButton_6=pushButton_6,  # 🆕 프로젝트 저장 버튼
                pushButton_7=pushButton_7,  # 🆕 학습→프로젝트
                pushButton_8=pushButton_8,  # 🆕 판정→프로젝트
                projects_dir=projects_dir,
                models_dir=os.path.join(root_dir, "Models"),  # 대문자 Models
                data_dir=os.path.join(root_dir, "Results"),  # Results 폴더
                logger=train_logger,
                train_ctrl=win._train_ctrl if hasattr(win, '_train_ctrl') else None  # 🎯 TrainingController 참조
            )
            print("✅ TrainingProjectController 초기화 완료")
            
            # 🎯 JudgeInferenceController에 TrainingProjectController 연결
            if hasattr(win, '_judge_infer_ctrl'):
                # train_proj_ctrl 참조 설정
                win._judge_infer_ctrl.train_proj_ctrl = win._train_proj_ctrl
                print("✅ JudgeInferenceController.train_proj_ctrl 설정 완료")
                
                # project_loaded 시그널 연결
                def _on_project_loaded_for_judge():
                    project_info = win._train_proj_ctrl.get_current_project_info()
                    if project_info:
                        win._judge_infer_ctrl.set_project_info(project_info)
                        print("📋 판정 탭에 프로젝트 정보 전달 완료")
                
                win._train_proj_ctrl.project_loaded.connect(_on_project_loaded_for_judge)
                print("✅ JudgeInferenceController에 TrainingProjectController 연결 완료")
                
                # 이미 로드된 프로젝트가 있으면 즉시 전달
                current_proj_info = win._train_proj_ctrl.get_current_project_info()
                if current_proj_info:
                    win._judge_infer_ctrl.set_project_info(current_proj_info)
                    print("📋 기존 프로젝트 정보 즉시 전달 완료")
            
            # 🗑️ SimpleJudgeController는 사용 안 함 (JudgeInferenceController만 사용)
            # if hasattr(win, '_simple_judge_ctrl'):
            #     win._simple_judge_ctrl.train_proj_ctrl = win._train_proj_ctrl
            #     print("✅ SimpleJudgeController에 TrainingProjectController 연결 완료")
        except Exception as e:
            print(f"⚠️ TrainingProjectController 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ list_train_projects를 찾을 수 없어 TrainingProjectController를 초기화하지 않습니다.")
        print(f"   - list_train_projects: {list_train_projects}")

    # ============================================
    # 6. Training 컨트롤러
    # ============================================
    
    # 🆕 groupBox_8 프로젝트 정보 표시용 위젯들
    textBrowser_project_2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_project_2")
    textBrowser_project_date_2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_project_date_2")
    textBrowser_process_2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_process_2")
    textBrowser_product_2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_product_2")
    
    # 🆕 groupBox_9 모델 정보 표시용 위젯들
    textBrowser_model_2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_model_2")
    textBrowser_traindata_2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_traindata_2")
    textBrowser_detail_2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_detail_2")
    textBrowser_validdata_2 = win.findChild(QtWidgets.QTextBrowser, "textBrowser_validdata_2")
    
    print(f"\n   📋 학습 탭 프로젝트 정보 표시용 위젯 (groupBox_8):")
    print(f"   - textBrowser_project_2: {'✅' if textBrowser_project_2 is not None else '❌'}")
    print(f"   - textBrowser_project_date_2: {'✅' if textBrowser_project_date_2 is not None else '❌'}")
    print(f"   - textBrowser_process_2: {'✅' if textBrowser_process_2 is not None else '❌'}")
    print(f"   - textBrowser_product_2: {'✅' if textBrowser_product_2 is not None else '❌'}")
    
    print(f"\n   📋 학습 탭 모델 정보 표시용 위젯 (groupBox_9):")
    print(f"   - textBrowser_model_2: {'✅' if textBrowser_model_2 is not None else '❌'}")
    print(f"   - textBrowser_traindata_2: {'✅' if textBrowser_traindata_2 is not None else '❌'}")
    print(f"   - textBrowser_detail_2 (배포날짜): {'✅' if textBrowser_detail_2 is not None else '❌'}")
    print(f"   - textBrowser_validdata_2: {'✅' if textBrowser_validdata_2 is not None else '❌'}")
    
    # 파일 선택 UI
    btn_model_load = win.findChild(QtWidgets.QPushButton, "btn_train_traincsv_load_ml_2")
    le_model_path = win.findChild(QtWidgets.QLineEdit, "le_train_model_name_ml")
    btn_train_csv = win.findChild(QtWidgets.QPushButton, "btn_train_traincsv_browse_ml")
    le_train_csv = win.findChild(QtWidgets.QLineEdit, "le_train_traincsv_ml")
    btn_val_csv = win.findChild(QtWidgets.QPushButton, "btn_train_validcsv_browse_ml")
    le_val_csv = win.findChild(QtWidgets.QLineEdit, "le_train_validcsv_ml")
    btn_out_dir = win.findChild(QtWidgets.QPushButton, "btn_train_outdir_browse_ml")
    le_out_dir = win.findChild(QtWidgets.QLineEdit, "le_train_outdir_ml")
    
    # 파라미터 UI
    cb_kernel = win.findChild(QtWidgets.QComboBox, "cb_train_cv_ml")
    cb_scaler = win.findChild(QtWidgets.QComboBox, "cb_train_scaler_ml")
    spin_seed = win.findChild(QtWidgets.QSpinBox, "spin_train_seed_ml")
    dspin_threshold = win.findChild(QtWidgets.QDoubleSpinBox, "dspin_train_thr_ml")
    
    # 피처 선택 UI
    list_time = win.findChild(QtWidgets.QListWidget, "listWidget")
    list_freq = win.findChild(QtWidgets.QListWidget, "listWidget_2")
    list_time_freq = win.findChild(QtWidgets.QListWidget, "listWidget_3")
    
    # 학습 제어 UI
    btn_start = win.findChild(QtWidgets.QPushButton, "btn_train_start_ml")
    btn_stop = win.findChild(QtWidgets.QPushButton, "btn_train_stop_ml")
    btn_save = win.findChild(QtWidgets.QPushButton, "pushButton_5")  # 저장 버튼
    progress_bar = win.findChild(QtWidgets.QProgressBar, "progress_train_ml_2")  # 수정: _2 추가
    
    # 결과 표시 UI
    label_cm = win.findChild(QtWidgets.QLabel, "label_train_plot_loss_ml_2")
    label_roc = win.findChild(QtWidgets.QLabel, "label_train_plot_val_ml_2")
    table_result = win.findChild(QtWidgets.QTableWidget, "tableWidget_result")
    text_shap = win.findChild(QtWidgets.QTextEdit, "text_shap_ml_1")
    text_importance = win.findChild(QtWidgets.QTextEdit, "text_featureimportance_1")
    text_report = win.findChild(QtWidgets.QTextEdit, "text_train_report_1")
    
    # Training 컨트롤러 초기화 (필수 UI가 있을 때만)
    if btn_start:
        try:
            win._train_ctrl = TrainingController(
                main_window=win,
                # 🆕 groupBox_8 프로젝트 정보 표시용 위젯
                textBrowser_project_2=textBrowser_project_2,
                textBrowser_project_date_2=textBrowser_project_date_2,
                textBrowser_process_2=textBrowser_process_2,
                textBrowser_product_2=textBrowser_product_2,
                # 🆕 groupBox_9 모델 정보 표시용 위젯
                textBrowser_model_2=textBrowser_model_2,
                textBrowser_traindata_2=textBrowser_traindata_2,
                textBrowser_detail_2=textBrowser_detail_2,
                textBrowser_validdata_2=textBrowser_validdata_2,
                # 파일 선택
                btn_model_load=btn_model_load,
                le_model_path=le_model_path,
                btn_train_csv=btn_train_csv,
                le_train_csv=le_train_csv,
                btn_val_csv=btn_val_csv,
                le_val_csv=le_val_csv,
                btn_out_dir=btn_out_dir,
                le_out_dir=le_out_dir,
                # 파라미터
                cb_kernel=cb_kernel,
                cb_scaler=cb_scaler,
                spin_seed=spin_seed,
                dspin_threshold=dspin_threshold,
                # 피처 선택
                list_time=list_time,
                list_freq=list_freq,
                list_time_freq=list_time_freq,
                # 학습 제어
                btn_start=btn_start,
                btn_stop=btn_stop,
                btn_save=btn_save,  # 저장 버튼
                progress_bar=progress_bar,
                # 결과 표시
                label_cm=label_cm,
                label_roc=label_roc,
                table_result=table_result,
                text_shap=text_shap,
                text_importance=text_importance,
                text_report=text_report,
                logger=train_logger,
                train_proj_ctrl=None,  # 🎯 나중에 연결
                models_dir=os.path.join(root_dir, "Models"),  # 모델 저장 디렉토리
                projects_dir=projects_dir  # 프로젝트 저장 디렉토리
            )
            print("✅ TrainingController 초기화 완료")
        except Exception as e:
            print(f"⚠️ TrainingController 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================
    # 🎯 TrainingController와 TrainingProjectController 연결
    # ============================================
    if hasattr(win, '_train_ctrl') and hasattr(win, '_train_proj_ctrl'):
        win._train_ctrl.train_proj_ctrl = win._train_proj_ctrl
        print("✅ TrainingController ↔ TrainingProjectController 연결 완료")
    
    # ============================================
    # 🎯 시그널 연결 - 프로젝트 자동 업데이트
    # ============================================
    # 판정 모델이 로드되면 -> 프로젝트 JSON 업데이트
    if hasattr(win, "_judge_model_ctrl") and hasattr(win, "_proj_ctrl"):
        win._judge_model_ctrl.model_loaded.connect(
            win._proj_ctrl.update_judge_model
        )
        print("✅ 시그널 연결: JudgeModel -> Project (모델 자동 업데이트)")
    
    # 판정 데이터가 로드되면 -> 프로젝트 JSON 업데이트
    if hasattr(win, "_judge_data_ctrl") and hasattr(win, "_proj_ctrl"):
        win._judge_data_ctrl.data_loaded.connect(
            win._proj_ctrl.update_judge_data
        )
        print("✅ 시그널 연결: JudgeData -> Project (데이터 자동 업데이트)")
    
    # ============================================
    # 컨트롤러 상태 출력
    # ============================================
    print("=" * 60)
    print("컨트롤러 초기화 완료")
    print("=" * 60)
    print(f"  - ProjectController:        {'✅' if hasattr(win, '_proj_ctrl') else '❌'}")
    print(f"  - JudgeModelController:     {'✅' if hasattr(win, '_judge_model_ctrl') else '❌'}")
    print(f"  - JudgeDataController:      {'✅' if hasattr(win, '_judge_data_ctrl') else '❌'}")
    print(f"  - JudgeInferController:     {'✅' if hasattr(win, '_judge_infer_ctrl') else '❌'}")
    print(f"  - SimpleJudgeController:    {'✅' if hasattr(win, '_simple_judge_ctrl') else '❌'}")
    print(f"  - TrainingProjectController: {'✅' if hasattr(win, '_train_proj_ctrl') else '❌'}")
    print(f"  - TrainingController:       {'✅' if hasattr(win, '_train_ctrl') else '❌'}")
    print("=" * 60)
    
    # ============================================
    # 🆕 headerBar 버튼 연결 (프로젝트 탭 이동)
    # ============================================
    print("\n🔍 headerBar 버튼 검색 중...")
    stackedWidget = win.findChild(QtWidgets.QStackedWidget, "stackedWidget")
    btn_home = win.findChild(QtWidgets.QPushButton, "pushButton")  # Home 버튼
    btn_exit = win.findChild(QtWidgets.QPushButton, "pushButton_2")  # 종료 버튼
    
    print(f"   - stackedWidget: {'✅' if stackedWidget is not None else '❌'}")
    print(f"   - btn_home (Home): {'✅' if btn_home is not None else '❌'}")
    print(f"   - btn_exit (종료): {'✅' if btn_exit is not None else '❌'}")
    
    # Home 버튼 -> 프로젝트 탭(page_project)으로 이동
    if btn_home and stackedWidget:
        def go_to_project_page():
            print("\n🏠 Home 버튼 클릭 -> 프로젝트 설정 탭으로 이동")
            page_project = win.findChild(QtWidgets.QWidget, "page_project")
            if page_project:
                stackedWidget.setCurrentWidget(page_project)
                print("✅ 프로젝트 탭으로 이동 완료")
        
        btn_home.clicked.connect(go_to_project_page)
        print("✅ Home 버튼 연결 완료: 프로젝트 탭으로 이동")
    
    # 종료 버튼 -> 프로그램 종료
    if btn_exit:
        btn_exit.clicked.connect(win.close)
        print("✅ 종료 버튼 연결 완료")
    
    # 🆕 페이지 전환 시 프로젝트 정보 업데이트
    # stackedWidget (판정 탭 내부 페이지)
    if stackedWidget and hasattr(win, '_judge_infer_ctrl') and hasattr(win, '_train_proj_ctrl'):
        def on_page_changed(index):
            """페이지 전환 시 프로젝트 정보 업데이트"""
            current_widget = stackedWidget.widget(index)
            if not current_widget:
                return
            
            page_name = current_widget.objectName()
            print(f"\n📄 stackedWidget 페이지 전환: {page_name}")
            
            # 판정 탭으로 이동 시
            if page_name == "page_judge":
                print("   🎯 판정 탭으로 이동 -> 프로젝트 정보 + 모델 전달")
                
                # 프로젝트 정보 업데이트
                if hasattr(win, '_train_proj_ctrl') and hasattr(win, '_judge_infer_ctrl'):
                    project_info = win._train_proj_ctrl.get_current_project_info()
                    if project_info:
                        win._judge_infer_ctrl.update_project_display(project_info)
                
                # 🆕 모델 + 데이터 무조건 전달 시도
                try:
                    train_proj_ctrl = win._train_proj_ctrl
                    judge_infer_ctrl = win._judge_infer_ctrl
                    
                    print(f"   🔍 모델: {train_proj_ctrl.current_model is not None}")
                    print(f"   🔍 경로: {train_proj_ctrl.current_model_path}")
                    
                    # 모델 전달
                    if train_proj_ctrl.current_model and train_proj_ctrl.current_model_path:
                        import os
                        judge_infer_ctrl.loaded_model = train_proj_ctrl.current_model
                        judge_infer_ctrl.model_path = train_proj_ctrl.current_model_path
                        print(f"   ✅ 모델 전달 완료: {os.path.basename(train_proj_ctrl.current_model_path)}")
                    
                    
                except Exception as e:
                    print(f"   ❌ 모델 전달 실패: {e}")
            
            # 학습 탭으로 이동 시
            elif page_name == "page_train":
                print("   🎯 학습 탭으로 이동 -> 프로젝트 정보 업데이트")
                print(f"   - win._train_proj_ctrl 존재: {hasattr(win, '_train_proj_ctrl')}")
                print(f"   - win._train_ctrl 존재: {hasattr(win, '_train_ctrl')}")
                
                if hasattr(win, '_train_proj_ctrl'):
                    project_info = win._train_proj_ctrl.get_current_project_info()
                    print(f"   - project_info: {project_info is not None}")
                    
                    if project_info and hasattr(win, '_train_ctrl'):
                        print(f"   - update_project_display 호출")
                        win._train_ctrl.update_project_display(project_info)
                    else:
                        if not project_info:
                            print(f"   ❌ project_info가 None")
                        if not hasattr(win, '_train_ctrl'):
                            print(f"   ❌ _train_ctrl이 없음")
                else:
                    print(f"   ❌ _train_proj_ctrl이 없음")
        
        stackedWidget.currentChanged.connect(on_page_changed)
        print("✅ stackedWidget 페이지 전환 이벤트 연결 완료")
    
    # 🆕 stack_pages (메인 탭: 프로젝트/판정/학습)
    stack_pages = win.findChild(QtWidgets.QStackedWidget, "stack_pages")
    if stack_pages and hasattr(win, '_judge_infer_ctrl') and hasattr(win, '_train_proj_ctrl'):
        def on_main_page_changed(index):
            """메인 페이지 전환 시 프로젝트 정보 업데이트"""
            current_widget = stack_pages.widget(index)
            if not current_widget:
                return
            
            page_name = current_widget.objectName()
            print(f"\n📄 stack_pages 페이지 전환: {page_name} (index={index})")
            
            # 판정 탭으로 이동 시
            if page_name == "page_judge":
                print("   🎯 판정 탭으로 이동 -> 프로젝트 정보 + 모델 전달")
                
                # 프로젝트 정보 업데이트
                if hasattr(win, '_train_proj_ctrl') and hasattr(win, '_judge_infer_ctrl'):
                    project_info = win._train_proj_ctrl.get_current_project_info()
                    if project_info:
                        win._judge_infer_ctrl.update_project_display(project_info)
                
                # 🆕 모델 + 데이터 무조건 전달 시도
                try:
                    train_proj_ctrl = win._train_proj_ctrl
                    judge_infer_ctrl = win._judge_infer_ctrl
                    
                    print(f"   🔍 모델: {train_proj_ctrl.current_model is not None}")
                    print(f"   🔍 경로: {train_proj_ctrl.current_model_path}")
                    
                    # 모델 전달
                    if train_proj_ctrl.current_model and train_proj_ctrl.current_model_path:
                        import os
                        judge_infer_ctrl.loaded_model = train_proj_ctrl.current_model
                        judge_infer_ctrl.model_path = train_proj_ctrl.current_model_path
                        print(f"   ✅ 모델 전달 완료: {os.path.basename(train_proj_ctrl.current_model_path)}")
                    
                    
                except Exception as e:
                    print(f"   ❌ 모델 전달 실패: {e}")
            
            # 학습 탭으로 이동 시
            elif page_name == "page_train":
                print("   🎯 학습 탭으로 이동 -> 프로젝트 정보 업데이트")
                if hasattr(win, '_train_proj_ctrl') and hasattr(win, '_train_ctrl'):
                    project_info = win._train_proj_ctrl.get_current_project_info()
                    if project_info:
                        win._train_ctrl.update_project_display(project_info)
        
        stack_pages.currentChanged.connect(on_main_page_changed)
        print("✅ stack_pages 페이지 전환 이벤트 연결 완료")
    
    print("✅ 페이지 전환 시 프로젝트 정보 자동 업데이트 연결 완료")
    
    print("=" * 60)


        # ============================================
        # 🎨 사이드바(list_nav) 간격 설정
        # ============================================
        
    print("\n🎨 사이드바 간격 설정 중...")
    list_nav = win.findChild(QtWidgets.QListWidget, "list_nav")

    if list_nav:
        # =========================
        # 1) 왼쪽 끝까지 붙이기 (레이아웃 margin 제거)
        # =========================
        sidebar = list_nav.parentWidget()  # 보통 sidebar(QFrame)
        if sidebar and sidebar.layout():
            sidebar.layout().setContentsMargins(0, 0, 0, 0)
            sidebar.layout().setSpacing(0)

        # 중앙 위젯/상위 레이아웃도 필요하면 같이 0
        cw = win.centralWidget()
        if cw and cw.layout():
            cw.layout().setContentsMargins(0, 0, 0, 0)

        # =========================
        # 2) 메뉴를 "밑으로 내리기"
        # =========================
        top_offset = 35  # 메뉴 시작 위치(아래로 내림)

        # =========================
        # 3) 아이템 간 세로 간격
        # =========================
        item_gap = 36
        list_nav.setSpacing(item_gap)

        # =========================
        # 4) 프레임/뷰포트 여백 제거 (깔끔)
        # =========================
        list_nav.setViewportMargins(0, 0, 0, 0)
        list_nav.viewport().setContentsMargins(0, 0, 0, 0)
        list_nav.setFrameShape(QtWidgets.QFrame.NoFrame)

        # =========================
        # 5) 스크롤은 가능(필요할 때만) + 스크롤바는 숨김(B)
        # =========================
        list_nav.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        list_nav.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # =========================
        # 6) 스타일 적용 (입체감 + top highlight 복구 + 스크롤바 숨김 포함)
        # =========================
        list_nav.setStyleSheet(f"""
            /* ✅ 스크롤바 완전 숨김(스크롤은 유지) */
            QScrollBar:vertical {{
                width: 0px;
                background: transparent;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                width: 0px;
                background: transparent;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
                background: transparent;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}

            QListWidget {{
                background: white;
                border: 0px;
                outline: 0px;
                padding-top: {top_offset}px;
                padding-left: 0px;
                padding-right: 0px;
                padding-bottom: 0px;
            }}

            QListWidget::viewport {{
                background: transparent;
                border: 0px;
                padding: 0px;
                margin: 0px;
            }}

            QListWidget::item {{
                margin: 6px 10px;
                padding: 40px 15px;
                border-radius: 12px;

                color: #111;

                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff,
                    stop:1 #f5f6f8
                );

                /* ✅ '4선' 느낌 복구: 윗줄(12시)을 흰색 대신 옅은 회색으로 */
                border: 1px solid #d9dde3;
                border-top: 2px solid #eef1f5;      /* ✅ 12시 하이라이트 */
                border-bottom: 1px solid #cfd5dd;   /* 눌림 느낌 */
            }}

            QListWidget::item:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff,
                    stop:1 #eef2f7
                );
                border: 1px solid #c7cdd6;
                border-top: 2px solid #eef1f5;
                border-bottom: 1px solid #bfc7d2;
            }}

            QListWidget::item:selected {{
                color: #1d4ed8;
                font-weight: 700;

                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #eaf2ff,
                    stop:1 #d7e7ff
                );

                border: 1px solid #3b82f6;
                border-top: 2px solid #3b82f6;
                border-bottom: 1px solid #2563eb;
            }}

            QListWidget::item:selected:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e1eeff,
                    stop:1 #cfe3ff
                );
                border: 1px solid #2563eb;
                border-top: 2px solid #3b82f6;;
                border-bottom: 1px solid #1d4ed8;
            }}
        """)

        print("✅ list_nav 스타일/레이아웃/스크롤바 숨김(B) 적용 완료")
    else:
        print("⚠️ list_nav를 찾을 수 없습니다")

        print("=" * 60)

    # ============================================
    # 7. DL Training 컨트롤러 (ResNet18 - tab_train_dl)
    # ============================================
    print("\n🔍 DL Training 위젯 검색 중...")
    tab_train_dl = win.findChild(QtWidgets.QWidget, "tab_train_dl")

    if tab_train_dl:
        def _find(name, cls=None):
            cls = cls or QtWidgets.QWidget
            return win.findChild(cls, name)

        win._dl_train_ctrl = DLTrainingController(
            main_window=win,
            # 프로젝트 정보
            textBrowser_project_6      = _find("textBrowser_project_6",      QtWidgets.QTextBrowser),
            textBrowser_project_date_6 = _find("textBrowser_project_date_6", QtWidgets.QTextBrowser),
            textBrowser_process_6      = _find("textBrowser_process_6",      QtWidgets.QTextBrowser),
            textBrowser_product_6      = _find("textBrowser_product_6",      QtWidgets.QTextBrowser),
            textBrowser_model_6        = _find("textBrowser_model_6",        QtWidgets.QTextBrowser),
            textBrowser_traindata_6    = _find("textBrowser_traindata_6",    QtWidgets.QTextBrowser),
            textBrowser_validdata_6    = _find("textBrowser_validdata_6",    QtWidgets.QTextBrowser),
            textBrowser_detail_6       = _find("textBrowser_detail_6",       QtWidgets.QTextBrowser),
            # 학습 제어
            btn_start     = _find("btn_train_start_ml_2", QtWidgets.QPushButton),
            progress_bar  = _find("progress_train_ml_3",  QtWidgets.QProgressBar),
            # 실시간 그래프
            label_loss    = _find("label_train_plot_loss_dl", QtWidgets.QLabel),
            label_val     = _find("label_train_plot_val_dl",  QtWidgets.QLabel),
            # 로그
            text_log      = _find("text_train_log_dl", QtWidgets.QTextEdit),
            # 결과 위젯
            widget_cm     = _find("label_train_plot_loss_ml_2"),
            widget_roc    = _find("label_train_plot_val_ml_2"),
            widget_grad   = _find("widget_Featureimportance_ml_1"),
            text_report   = _find("text_train_report_1", QtWidgets.QTextEdit),
            # 디렉토리
            models_dir    = os.path.join(root_dir, "Models"),
            projects_dir  = projects_dir,
            # 연동
            train_proj_ctrl = win._train_proj_ctrl if hasattr(win, '_train_proj_ctrl') else None,
            logger          = train_logger,
        )
        # 프로젝트 로드 시 DL 탭도 자동 업데이트
        if hasattr(win, '_train_proj_ctrl'):
            win._train_proj_ctrl.project_loaded.connect(
                lambda info: win._dl_train_ctrl.update_project_display(info)
            )
        print("✅ DLTrainingController 초기화 완료")
    else:
        print("⚠️ tab_train_dl 없음 - DLTrainingController 스킵")

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()