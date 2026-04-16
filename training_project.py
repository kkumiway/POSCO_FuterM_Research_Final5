# -*- coding: utf-8 -*-
"""training_project.py - 모델 학습용 프로젝트 관리 (v2 - UI 위젯 이름 업데이트)"""

import os
import json
import pickle
import torch
from datetime import datetime
from typing import Optional, Dict, Any
from PyQt5 import QtCore, QtWidgets

# SeedEnsemble import (모델 로드용)
try:
    from judge_inference_integrated import SeedEnsemble
    # PyTorch 2.6+ safe_globals 등록
    torch.serialization.add_safe_globals([SeedEnsemble])
except ImportError:
    SeedEnsemble = None
    print("⚠️ SeedEnsemble을 import할 수 없습니다. judge_inference.py를 확인하세요.")


class TrainingProjectController(QtCore.QObject):
    """
    모델 학습 탭의 프로젝트 관리 컨트롤러
    - 프로젝트 리스트 표시 (projects 폴더의 JSON 파일)
    - 프로젝트 선택 시 자동 세팅
    - 모델 자동 로드
    """
    
    project_loaded = QtCore.pyqtSignal(dict)  # 프로젝트 로드 시그널
    model_loaded = QtCore.pyqtSignal(object, dict)  # 모델 로드 시그널 (model_obj, metadata)
    
    def __init__(
        self,
        main_window,
        # 프로젝트 리스트
        list_train_projects=None,
        # 프로젝트 메타 정보
        le_train_project_name_3=None,
        le_train_project_date=None,
        le_train_project_process=None,
        le_train_project_process_2=None,  # 상세정보 (QLineEdit 또는 QPlainTextEdit)
        # 모델 정보
        le_train_model_name_ml_2=None,
        le_train_purpose=None,
        le_train_purpose_product=None,  # algorithm → target_product로 변경
        le_train_deploy_date=None,
        # 데이터 정보
        le_train_traincsv_ml_2=None,
        le_train_validcsv_ml_2=None,
        # 버튼들
        btn_project_new=None,  # 새 프로젝트
        btn_train_traincsv_load_ml_3=None,  # 모델 로드
        btn_train_traincsv_browse_ml_2=None,  # 학습 데이터 찾기
        btn_train_validcsv_browse_ml_2=None,  # 검증 데이터 찾기
        pushButton_3=None,  # 균열 판정으로 전환
        pushButton_4=None,  # 모델 학습으로 전환
        pushButton_6=None,  # 🆕 프로젝트 저장 버튼
        pushButton_7=None,  # 🆕 학습 → 프로젝트 설정 이동
        pushButton_8=None,  # 🆕 판정 → 프로젝트 설정 이동
        # 디렉토리
        projects_dir: str = None,
        models_dir: str = None,
        data_dir: str = None,
        logger=None,
        # 🎯 TrainingController 참조
        train_ctrl=None
    ):
        super().__init__(main_window)
        
        self.main_window = main_window
        
        # UI 위젯
        self.list_projects = list_train_projects
        
        # 🎯 리스트 위젯 초기 설정
        if self.list_projects is not None:
            # 가로 스크롤바 활성화
            self.list_projects.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            # 세로 스크롤바 항상 표시
            self.list_projects.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            # 텍스트 줄바꿈 끄기
            self.list_projects.setWordWrap(False)
            # 균일한 항목 크기
            self.list_projects.setUniformItemSizes(True)
            # 아이콘 영역 최소화
            self.list_projects.setIconSize(QtCore.QSize(0, 0))
            # 🎯 폰트 크기 키우기
            font = self.list_projects.font()
            font.setPointSize(11)  # 9 → 11로 변경 (더 크게: 12, 13 등)
            # font.setBold(True)  # 굵게 (선택사항)
            self.list_projects.setFont(font)
        
        self.le_name = le_train_project_name_3
        self.le_date = le_train_project_date
        self.le_process = le_train_project_process
        self.le_model_date = le_train_project_process_2  # 🆕 상세내용 → 모델 날짜로 사용
        
        self.le_model = le_train_model_name_ml_2
        self.le_purpose = le_train_purpose
        self.le_target_product = le_train_purpose_product  # algorithm → target_product
        self.le_deploy = le_train_deploy_date
        
        self.le_train_data = le_train_traincsv_ml_2
        self.le_valid_data = le_train_validcsv_ml_2
        
        # 버튼들
        self.btn_new_project = btn_project_new
        self.btn_load_model = btn_train_traincsv_load_ml_3
        self.btn_browse_train = btn_train_traincsv_browse_ml_2
        self.btn_browse_valid = btn_train_validcsv_browse_ml_2
        self.btn_to_judge = pushButton_3
        self.btn_to_training = pushButton_4
        self.btn_save_project = pushButton_6  # 🆕 프로젝트 저장 버튼
        self.btn_to_project_from_train = pushButton_7  # 🆕 학습 → 프로젝트
        self.btn_to_project_from_judge = pushButton_8  # 🆕 판정 → 프로젝트
        
        # 디렉토리
        self.projects_dir = projects_dir or os.path.join(os.getcwd(), "projects")
        self.models_dir = models_dir or os.path.join(os.getcwd(), "Models")  # 대문자 M
        self.data_dir = data_dir or os.path.join(os.getcwd(), "Results")  # Results 폴더로 변경
        
        # 디렉토리 생성
        os.makedirs(self.projects_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger = logger
        
        # 🎯 TrainingController 참조
        self.train_ctrl = train_ctrl
        
        # 디버깅 로그
        if self.logger:
            self.logger.log_info("=" * 50)
            self.logger.log_info("🚀 TrainingProjectController 초기화 시작")
            self.logger.log_info(f"   - list_projects: {'✅ 있음' if list_train_projects is not None else '❌ 없음'}")
            self.logger.log_info(f"   - projects_dir: {self.projects_dir}")
            self.logger.log_info(f"   - models_dir: {self.models_dir}")
            self.logger.log_info(f"   - data_dir: {self.data_dir}")
            self.logger.log_info("=" * 50)
        
        # 현재 선택된 프로젝트
        self.current_project_file: Optional[str] = None
        self.current_project_data: Optional[Dict[str, Any]] = None
        self.current_model = None  # 로드된 모델 객체
        self.current_model_path = None  # 로드된 모델 파일 경로 (문자열)
        self.current_train_data = None  # 로드된 학습 데이터 (DataFrame)
        self.current_valid_data = None  # 로드된 검증 데이터 (DataFrame)
        
        # 초기화
        self._connect_signals()
        self._load_project_list()
    
    def _connect_signals(self):
        """시그널 연결"""
        print("\n🔗 TrainingProjectController 시그널 연결 중...")
        
        if self.list_projects is not None:
            self.list_projects.itemClicked.connect(self._on_project_selected)
            print("   ✅ list_train_projects 연결됨")
        else:
            print("   ❌ list_train_projects 없음")
        
        # 버튼 시그널 연결
        if self.btn_new_project is not None:
            self.btn_new_project.clicked.connect(self._on_new_project)
            print("   ✅ btn_project_new 연결됨")
        else:
            print("   ❌ btn_project_new 없음")
        
        if self.btn_load_model is not None:
            self.btn_load_model.clicked.connect(self._on_browse_model)
            print("   ✅ btn_train_traincsv_load_ml_3 연결됨")
        else:
            print("   ❌ btn_train_traincsv_load_ml_3 없음")
        
        if self.btn_browse_train is not None:
            self.btn_browse_train.clicked.connect(self._on_browse_train_data)
            print("   ✅ btn_train_traincsv_browse_ml_2 연결됨")
        else:
            print("   ❌ btn_train_traincsv_browse_ml_2 없음")
        
        if self.btn_browse_valid is not None:
            self.btn_browse_valid.clicked.connect(self._on_browse_valid_data)
            print("   ✅ btn_train_validcsv_browse_ml_2 연결됨")
        else:
            print("   ❌ btn_train_validcsv_browse_ml_2 없음")
        
        if self.btn_to_judge is not None:
            self.btn_to_judge.clicked.connect(self._on_goto_judge)
            print("   ✅ pushButton_3 (균열판정 이동) 연결됨")
        else:
            print("   ❌ pushButton_3 없음")
        
        if self.btn_to_training is not None:
            self.btn_to_training.clicked.connect(self._on_goto_training)
            print("   ✅ pushButton_4 (모델학습 이동) 연결됨")
        else:
            print("   ❌ pushButton_4 없음")
        
        if self.btn_save_project is not None:
            self.btn_save_project.clicked.connect(self.save_current_project)
            print("   ✅ pushButton_6 (프로젝트 저장) 연결됨")
        else:
            print("   ❌ pushButton_6 없음")
        
        if self.btn_to_project_from_train is not None:
            self.btn_to_project_from_train.clicked.connect(self._on_goto_project)
            print("   ✅ pushButton_7 (학습→프로젝트) 연결됨")
        else:
            print("   ❌ pushButton_7 없음")
        
        if self.btn_to_project_from_judge is not None:
            self.btn_to_project_from_judge.clicked.connect(self._on_goto_project)
            print("   ✅ pushButton_8 (판정→프로젝트) 연결됨")
        else:
            print("   ❌ pushButton_8 없음")
        
        print("🔗 TrainingProjectController 시그널 연결 완료\n")
    
    def _load_project_list(self):
        """프로젝트 리스트 로드"""
        if self.logger:
            self.logger.log_info(f"🔍 프로젝트 리스트 로드 시작...")
            self.logger.log_info(f"   - list_projects 위젯: {'있음' if self.list_projects is not None else '없음'}")
            self.logger.log_info(f"   - projects_dir: {self.projects_dir}")
        
        if self.list_projects is None:
            if self.logger:
                self.logger.log_error("❌ list_projects 위젯이 None입니다!")
            return
        
        self.list_projects.clear()
        
        if not os.path.exists(self.projects_dir):
            if self.logger:
                self.logger.log_warning(f"⚠️ projects 폴더가 없습니다: {self.projects_dir}")
            return
        
        # JSON 파일만 필터링
        json_files = [f for f in os.listdir(self.projects_dir) if f.endswith('.json')]
        json_files.sort(reverse=True)  # 최신순 정렬
        
        if self.logger:
            self.logger.log_info(f"   - 발견된 JSON 파일: {len(json_files)}개")
            if json_files:
                for jf in json_files[:5]:  # 최대 5개만 출력
                    self.logger.log_info(f"      • {jf}")
                if len(json_files) > 5:
                    self.logger.log_info(f"      ... (외 {len(json_files) - 5}개)")
        
        for json_file in json_files:
            # 확장자 제거한 이름으로 표시
            project_name = os.path.splitext(json_file)[0]
            item = QtWidgets.QListWidgetItem(project_name)
            
            # 🆕 딕셔너리로 저장 (안전성 향상)
            item.setData(QtCore.Qt.UserRole, {
                'json_file': json_file,
                'json_path': os.path.join(self.projects_dir, json_file),
                'project_name': project_name
            })
            
            # 항목 크기 설정 (폰트 크기에 맞춰 높이 증가)
            item.setSizeHint(QtCore.QSize(-1, 35))  # 30 → 35px (더 크게: 40, 45 등)
            # 툴팁 추가 (전체 이름 보기)
            item.setToolTip(project_name)
            
            self.list_projects.addItem(item)
        
        if self.logger:
            self.logger.log_success(f"✅ 프로젝트 {len(json_files)}개 로드 완료")
    
    def _on_project_selected(self, item: QtWidgets.QListWidgetItem):
        """프로젝트 선택 시"""
        item_data = item.data(QtCore.Qt.UserRole)
        
        # 🆕 안전하게 데이터 가져오기
        if isinstance(item_data, dict):
            json_file = item_data.get('json_file', '')
            project_path = item_data.get('json_path', '')
            if not project_path:
                project_path = os.path.join(self.projects_dir, json_file)
        else:
            # 레거시: 문자열인 경우
            json_file = item_data
            project_path = os.path.join(self.projects_dir, json_file)
        
        self._load_project(project_path)
    
    def _load_project(self, project_path: str):
        """프로젝트 JSON 로드 및 UI 세팅"""
        try:
            print(f"\n📂 프로젝트 로드 시작: {os.path.basename(project_path)}")
            
            with open(project_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_project_file = os.path.basename(project_path)
            self.current_project_data = data
            
            # 🆕 두 가지 JSON 구조 지원
            # 1. Flat 구조 (save_current_project가 저장)
            # 2. meta/model 구조 (기존 구조)
            
            if "meta" in data or "model" in data:
                # 기존 구조: meta/model
                print("   📋 기존 구조 (meta/model) 감지")
                meta = data.get("meta", {})
                model = data.get("model", {})
            else:
                # 새 구조: flat
                print("   📋 새 구조 (flat) 감지")
                meta = data
                model = data
            
            # 프로젝트 메타 정보
            if self.le_name:
                project_filename = os.path.splitext(os.path.basename(project_path))[0]
                self.le_name.setText(project_filename)
                print(f"   ✅ 프로젝트명: {project_filename}")
            
            if self.le_date:
                date_value = meta.get("created_date", "")
                self.le_date.setText(date_value)
                print(f"   ✅ 생성날짜: {date_value}")
            
            if self.le_process:
                process_value = meta.get("target_process", "")
                self.le_process.setText(process_value)
                print(f"   ✅ 대상공정: {process_value}")
            
            if self.le_target_product:
                product_value = model.get("target_product", "")
                self.le_target_product.setText(product_value)
                print(f"   ✅ 대상제품: {product_value}")
            
            if self.le_purpose:
                purpose_value = model.get("purpose", "")
                self.le_purpose.setText(purpose_value)
                print(f"   ✅ 용도: {purpose_value}")
            
            if self.le_model_date:
                model_date_value = model.get("model_date", "")
                self.le_model_date.setText(model_date_value)
                print(f"   ✅ 모델날짜: {model_date_value}")
            
            # 모델 정보
            model_path = model.get("model_path", "") or model.get("model_filename", "")
            if model_path:
                model_filename = os.path.basename(model_path)
                
                if self.le_model:
                    self.le_model.setText(model_filename)
                
                print(f"   ✅ 모델파일: {model_filename}")
                
                full_model_path = os.path.join(self.models_dir, model_filename)
                if not os.path.exists(full_model_path):
                    full_model_path = model_path
                    if not os.path.isabs(full_model_path):
                        full_model_path = os.path.join(os.getcwd(), full_model_path)
                
                if os.path.exists(full_model_path):
                    self._load_model(full_model_path, model)
                else:
                    print(f"   ⚠️ 모델 파일 없음: {full_model_path}")
            else:
                if self.le_model:
                    self.le_model.setText("")
                print(f"   ℹ️ 모델파일 정보 없음")
            
            # 데이터 경로
            # train_data_path (전체경로) 또는 train_data_filename (파일명만)
            train_data = model.get("train_data_path", "") or model.get("train_data_filename", "")
            if self.le_train_data:
                # 파일명만 표시 (없어도 빈 값 표시)
                train_filename = os.path.basename(train_data) if train_data else ""
                self.le_train_data.setText(train_filename)
                if train_filename:
                    print(f"   ✅ 학습데이터: {train_filename}")
                else:
                    print(f"   ℹ️ 학습데이터 정보 없음")
                
                # 🎯 학습 데이터 자동 로드 (있으면)
                if os.path.isabs(train_data):
                    full_train_path = train_data
                else:
                    full_train_path = os.path.join(self.data_dir, train_data)
                
                if os.path.exists(full_train_path):
                    self._load_data(full_train_path, data_type="train")
            
            # valid_data_path (전체경로) 또는 valid_data_filename (파일명만)
            valid_data = model.get("valid_data_path", "") or model.get("valid_data_filename", "")
            if self.le_valid_data:
                # 파일명만 표시 (없어도 빈 값 표시)
                valid_filename = os.path.basename(valid_data) if valid_data else ""
                self.le_valid_data.setText(valid_filename)
                if valid_filename:
                    print(f"   ✅ 검증데이터: {valid_filename}")
                else:
                    print(f"   ℹ️ 검증데이터 정보 없음")
                
                # 🎯 검증 데이터 자동 로드 (있으면)
                if os.path.isabs(valid_data):
                    full_valid_path = valid_data
                else:
                    full_valid_path = os.path.join(self.data_dir, valid_data)
                
                if os.path.exists(full_valid_path):
                    self._load_data(full_valid_path, data_type="valid")
            
            # 시그널 emit
            self.project_loaded.emit(data)
            
            # 프로젝트명 출력
            proj_name = meta.get("project_name", "")
            print(f"🎉 프로젝트 '{proj_name}' 로드 완료!\n")
            
            # 🎯 데이터 로드 상태 확인
            if self.logger:
                self.check_data_loaded()
            else:
                # logger가 없어도 콘솔에는 출력
                self.check_data_loaded()
            
            # 🎯 TrainingController에 프로젝트 정보 전달
            if self.train_ctrl:
                model_name = os.path.basename(model_path) if model_path else ""
                train_csv_name = os.path.basename(train_data) if train_data else ""
                valid_csv_name = os.path.basename(valid_data) if valid_data else ""
                target_product = model.get("target_product", "")  # algorithm → target_product
                
                self.train_ctrl.set_project_info(
                    model_name=model_name,
                    train_csv=train_csv_name,
                    valid_csv=valid_csv_name,
                    target_product=target_product  # algorithm → target_product
                )
                
                # 🆕 학습 탭 프로젝트 정보 표시
                project_filename = os.path.splitext(os.path.basename(project_path))[0]
                project_info = {
                    "project_filename": project_filename,
                    "created_date": meta.get("created_date", ""),
                    "target_process": meta.get("target_process", ""),
                    "target_product": model.get("target_product", "")
                }
                self.train_ctrl.update_project_display(project_info)
        
        except Exception as e:
            print(f"❌ 프로젝트 로드 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "프로젝트 로드 실패",
                f"프로젝트를 로드할 수 없습니다:\n{str(e)}"
            )
            if self.logger:
                self.logger.log_error(f"❌ 프로젝트 로드 실패: {str(e)}")
    
    def _load_model(self, model_path: str, model_info: Dict[str, Any]):
        """
        모델 자동 로드
        .pt 파일과 .json 메타데이터 로드
        """
        if not os.path.isabs(model_path):
            # 상대 경로면 models_dir 기준으로 변환
            model_path = os.path.join(self.models_dir, model_path)
        
        if not os.path.exists(model_path):
            print(f"⚠️ 모델 파일 없음: {model_path}")
            if self.logger:
                self.logger.log_warning(f"⚠️ 모델 파일 없음: {model_path}")
            return
        
        try:
            # 🎯 모델 파일의 수정 날짜를 배포일로 설정
            file_mtime = os.path.getmtime(model_path)
            deploy_date = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")  # yyyy-MM-dd 형식
            
            print(f"📅 파일 날짜 추출: {deploy_date}")
            print(f"📅 le_model_date 위젯: {self.le_model_date is not None}")
            
            if self.le_model_date:
                self.le_model_date.setText(deploy_date)
                print(f"📅 배포일 자동 설정: {deploy_date} (파일 수정 날짜)")
                print(f"📅 설정 후 확인: le_model_date.text() = '{self.le_model_date.text()}'")
            else:
                print(f"⚠️ le_model_date 위젯이 None입니다!")
            
            # 1. .pt 파일 로드
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                print(f"🔧 모델 파일 로드 시작: {os.path.basename(model_path)}")
                
                # PyTorch 2.6+ 호환: weights_only=False (신뢰할 수 있는 파일)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # SeedEnsemble 또는 단일 모델 처리
                if isinstance(checkpoint, dict):
                    # checkpoint 형태
                    if 'model_state_dict' in checkpoint:
                        model_obj = checkpoint.get('model_state_dict')
                    elif 'model' in checkpoint:
                        model_obj = checkpoint.get('model')
                    else:
                        model_obj = checkpoint
                else:
                    model_obj = checkpoint
                
                self.current_model = model_obj
                self.current_model_path = model_path  # 모델 파일 경로 저장
                
                # 🆕 디버깅 로그
# 시그널 emit
                self.model_loaded.emit(model_obj, model_info)
                
                print(f"✅ 모델 로드 완료: {os.path.basename(model_path)}")
                if self.logger:
                    self.logger.log_model(f"🔧 모델 로드 완료: {os.path.basename(model_path)}")
            
            # 2. .json 메타데이터 로드 (선택) - deploy_date 제외
            json_path = model_path.rsplit('.', 1)[0] + '.json'
            if os.path.exists(json_path):
                print(f"📄 메타데이터 로드: {os.path.basename(json_path)}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # UI 업데이트 (metadata 우선, deploy_date는 제외)
                if self.le_purpose and 'purpose' in metadata:
                    self.le_purpose.setText(metadata['purpose'])
                if self.le_target_product and 'target_product' in metadata:  # algorithm → target_product
                    self.le_target_product.setText(metadata['target_product'])
                # deploy_date는 파일 날짜 사용하므로 JSON에서 읽지 않음
                
                if self.logger:
                    self.logger.log_file(f"📄 메타데이터 로드: {os.path.basename(json_path)}")
        
        except Exception as e:
            print(f"❌ 모델 로드 실패: {str(e)}")
            if self.logger:
                self.logger.log_error(f"❌ 모델 로드 실패: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _load_data(self, data_path: str, data_type: str = "train"):
        """
        CSV 데이터 자동 로드
        
        Args:
            data_path: 데이터 파일 경로 (파일명 또는 전체 경로)
            data_type: "train" 또는 "valid"
        """
        try:
            import pandas as pd
            
            # 경로 처리
            if not os.path.isabs(data_path):
                # 상대 경로면 data_dir (Results) 기준으로 변환
                full_path = os.path.join(self.data_dir, data_path)
            else:
                full_path = data_path
            
            # 파일명만 있는 경우 data_dir에서 찾기
            if not os.path.exists(full_path):
                full_path = os.path.join(self.data_dir, os.path.basename(data_path))
            
            if not os.path.exists(full_path):
                print(f"⚠️ 데이터 파일 없음: {full_path}")
                print(f"   시도한 경로1: {os.path.join(self.data_dir, data_path)}")
                print(f"   시도한 경로2: {os.path.join(self.data_dir, os.path.basename(data_path))}")
                if self.logger:
                    self.logger.log_warning(f"⚠️ 데이터 파일 없음: {full_path}")
                    self.logger.log_info(f"   시도한 경로1: {os.path.join(self.data_dir, data_path)}")
                    self.logger.log_info(f"   시도한 경로2: {os.path.join(self.data_dir, os.path.basename(data_path))}")
                return
            
            # CSV 로드
            print(f"📁 CSV 파일 읽기 시작: {os.path.basename(full_path)}")
            if self.logger:
                self.logger.log_info(f"📁 CSV 파일 읽기 시작: {os.path.basename(full_path)}")
            
            df = pd.read_csv(full_path, encoding='utf-8-sig')
            
            # 저장
            if data_type == "train":
                self.current_train_data = df
                print(f"✅ 학습 데이터 로드 완료!")
                print(f"   📊 Shape: {len(df)}행 × {len(df.columns)}열")
                print(f"   📂 파일: {os.path.basename(full_path)}")
                print(f"   🗂️ 컬럼 수: {len(df.columns)}개")
                if len(df) > 0:
                    cols_preview = ', '.join(df.columns[:5].tolist())
                    if len(df.columns) > 5:
                        cols_preview += '...'
                    print(f"   🔍 첫 행 확인: OK (컬럼: {cols_preview})")
                
                if self.logger:
                    self.logger.log_success(f"✅ 학습 데이터 로드 완료!")
                    self.logger.log_info(f"   📊 Shape: {len(df)}행 × {len(df.columns)}열")
                    self.logger.log_info(f"   📂 파일: {os.path.basename(full_path)}")
                    self.logger.log_info(f"   🗂️ 컬럼 수: {len(df.columns)}개")
                    if len(df) > 0:
                        self.logger.log_info(f"   🔍 첫 행 확인: OK (컬럼: {', '.join(df.columns[:5].tolist())}{'...' if len(df.columns) > 5 else ''})")
            else:  # valid
                self.current_valid_data = df
                print(f"✅ 검증 데이터 로드 완료!")
                print(f"   📊 Shape: {len(df)}행 × {len(df.columns)}열")
                print(f"   📂 파일: {os.path.basename(full_path)}")
                print(f"   🗂️ 컬럼 수: {len(df.columns)}개")
                if len(df) > 0:
                    print(f"   🔍 첫 행 확인: OK")
                
                if self.logger:
                    self.logger.log_success(f"✅ 검증 데이터 로드 완료!")
                    self.logger.log_info(f"   📊 Shape: {len(df)}행 × {len(df.columns)}열")
                    self.logger.log_info(f"   📂 파일: {os.path.basename(full_path)}")
                    self.logger.log_info(f"   🗂️ 컬럼 수: {len(df.columns)}개")
                    if len(df) > 0:
                        self.logger.log_info(f"   🔍 첫 행 확인: OK")
        
        except Exception as e:
            print(f"❌ 데이터 로드 실패 ({data_type}): {str(e)}")
            if self.logger:
                self.logger.log_error(f"❌ 데이터 로드 실패 ({data_type}): {str(e)}")
            import traceback
            traceback.print_exc()
            if self.logger:
                self.logger.log_error(f"   상세: {traceback.format_exc()}")
    
    def get_current_project_info(self) -> Optional[Dict[str, str]]:
        """
        현재 프로젝트 정보를 균열판정 페이지에서 사용할 수 있도록 반환
        🆕 프로젝트 탭의 QLineEdit에서 현재 입력된 값을 실시간으로 읽음
        """
        print(f"\n📋 프로젝트 정보 읽기 (UI 위젯에서 직접):")
        
        # 🆕 UI 위젯에서 현재 값 직접 읽기
        project_filename = self.le_name.text() if self.le_name else ""
        created_date = self.le_date.text() if self.le_date else ""
        target_process = self.le_process.text() if self.le_process else ""
        model_filename = self.le_model.text() if self.le_model else ""
        purpose = self.le_purpose.text() if self.le_purpose else ""
        target_product = self.le_target_product.text() if self.le_target_product else ""
        model_date = self.le_model_date.text() if self.le_model_date else ""
        train_data_filename = self.le_train_data.text() if self.le_train_data else ""
        valid_data_filename = self.le_valid_data.text() if self.le_valid_data else ""
        
        print(f"   - 프로젝트명: {project_filename}")
        print(f"   - 생성날짜: {created_date}")
        print(f"   - 대상공정: {target_process}")
        print(f"   - 모델파일: {model_filename}")
        print(f"   - 용도: {purpose}")
        print(f"   - 대상제품: {target_product}")
        print(f"   - 모델날짜: {model_date}")
        print(f"   - 학습데이터: {train_data_filename}")
        print(f"   - 검증데이터: {valid_data_filename}")
        
        return {
            "project_name": project_filename,  # 🆕 프로젝트명 = 파일명
            "project_filename": project_filename,
            "created_date": created_date,
            "target_process": target_process,
            "project_detail": "-",  # 상세 설명 (사용 안 함)
            "model_filename": model_filename,
            "purpose": purpose,
            "target_product": target_product,
            "model_date": model_date,
            "train_data_filename": train_data_filename,
            "valid_data_filename": valid_data_filename
        }
    
    def get_current_project_data(self) -> Optional[Dict[str, Any]]:
        """현재 프로젝트 데이터 반환"""
        return self.current_project_data
    
    def get_current_model(self):
        """현재 로드된 모델 반환"""
        return self.current_model
    
    def get_train_data(self):
        """현재 로드된 학습 데이터 반환"""
        return self.current_train_data
    
    def get_valid_data(self):
        """현재 로드된 검증 데이터 반환"""
        return self.current_valid_data
    
    def check_data_loaded(self):
        """데이터 로드 상태 확인 및 로그 출력"""
        print("\n" + "="*50)
        print("🔍 데이터 로드 상태 확인")
        
        # 학습 데이터
        if self.current_train_data is not None:
            print(f"✅ 학습 데이터: 로드됨 ({len(self.current_train_data)}행 × {len(self.current_train_data.columns)}열)")
            cols_preview = ', '.join(self.current_train_data.columns[:5].tolist())
            if len(self.current_train_data.columns) > 5:
                cols_preview += '...'
            print(f"   컬럼: {cols_preview}")
        else:
            print("❌ 학습 데이터: 로드 안됨")
        
        # 검증 데이터
        if self.current_valid_data is not None:
            print(f"✅ 검증 데이터: 로드됨 ({len(self.current_valid_data)}행 × {len(self.current_valid_data.columns)}열)")
        else:
            print("❌ 검증 데이터: 로드 안됨")
        
        # 모델
        if self.current_model is not None:
            print("✅ 모델: 로드됨")
        else:
            print("❌ 모델: 로드 안됨")
        
        print("="*50 + "\n")
        
        if self.logger:
            self.logger.separator()
            self.logger.log_info("🔍 데이터 로드 상태 확인")
            
            # 학습 데이터
            if self.current_train_data is not None:
                self.logger.log_success(f"✅ 학습 데이터: 로드됨 ({len(self.current_train_data)}행 × {len(self.current_train_data.columns)}열)")
                self.logger.log_info(f"   컬럼: {', '.join(self.current_train_data.columns[:5].tolist())}{'...' if len(self.current_train_data.columns) > 5 else ''}")
            else:
                self.logger.log_warning("❌ 학습 데이터: 로드 안됨")
            
            # 검증 데이터
            if self.current_valid_data is not None:
                self.logger.log_success(f"✅ 검증 데이터: 로드됨 ({len(self.current_valid_data)}행 × {len(self.current_valid_data.columns)}열)")
            else:
                self.logger.log_warning("❌ 검증 데이터: 로드 안됨")
            
            # 모델
            if self.current_model is not None:
                self.logger.log_success("✅ 모델: 로드됨")
            else:
                self.logger.log_warning("❌ 모델: 로드 안됨")
            
            self.logger.separator()
        
        return {
            "train_data": self.current_train_data is not None,
            "valid_data": self.current_valid_data is not None,
            "model": self.current_model is not None
        }
    
    def _on_new_project(self):
        """새 프로젝트 - 모든 필드 초기화"""
        print("\n📝 새 프로젝트 생성 모드")
        
        # 모든 필드 초기화
        if self.le_name:
            self.le_name.clear()
        if self.le_date:
            self.le_date.setText(datetime.now().strftime("%Y-%m-%d"))
        if self.le_process:
            self.le_process.clear()
        if self.le_model_date:  # 🆕 le_detail → le_model_date
            self.le_model_date.clear()
        
        if self.le_model:
            self.le_model.clear()
        if self.le_purpose:
            self.le_purpose.clear()
        if self.le_target_product:  # algorithm → target_product
            self.le_target_product.clear()
        # le_deploy는 더 이상 사용 안 함
        
        if self.le_train_data:
            self.le_train_data.clear()
        if self.le_valid_data:
            self.le_valid_data.clear()
        
        # 현재 데이터 초기화
        self.current_project_file = None
        self.current_project_data = None
        self.current_model = None
        self.current_model_path = None
        self.current_train_data = None
        self.current_valid_data = None
        
        # 🆕 학습 탭 프로젝트 정보 초기화
        if self.train_ctrl:
            empty_info = {
                "project_filename": "",
                "created_date": "",
                "target_process": "",
                "target_product": ""
            }
            self.train_ctrl.update_project_display(empty_info)
        
        if self.logger:
            self.logger.log_info("📝 새 프로젝트: 모든 필드가 초기화되었습니다.")
        
        print("✅ 새 프로젝트 준비 완료 - 정보를 입력하세요")
    
    def _on_browse_model(self):
        """모델 파일 찾기"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window,
            "모델 파일 선택",
            self.models_dir,
            "PyTorch 모델 (*.pt *.pth);;All Files (*.*)"
        )
        
        if file_path:
            print(f"📂 모델 파일 선택: {file_path}")
            
            # 파일명만 표시
            filename = os.path.basename(file_path)
            if self.le_model:
                self.le_model.setText(filename)
            
            # 🆕 파일 수정 날짜 자동 추출 및 설정
            try:
                file_mtime = os.path.getmtime(file_path)
                file_date = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
                if self.le_model_date:
                    self.le_model_date.setText(file_date)
                    print(f"📅 모델 파일 날짜: {file_date}")
            except Exception as e:
                print(f"⚠️ 파일 날짜 추출 실패: {e}")
            
            # 모델 로드
            self._load_model(file_path, {})
            
            if self.logger:
                self.logger.log_file(f"📂 모델 파일 선택: {filename}")
    
    def _on_browse_train_data(self):
        """학습 데이터 CSV 찾기"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window,
            "학습 데이터 선택",
            self.data_dir,
            "CSV 파일 (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            print(f"📂 학습 데이터 선택: {file_path}")
            
            # 파일명만 표시
            filename = os.path.basename(file_path)
            if self.le_train_data:
                self.le_train_data.setText(filename)
            
            # 데이터 로드
            self._load_data(file_path, data_type="train")
            
            if self.logger:
                self.logger.log_file(f"📂 학습 데이터 선택: {filename}")
    
    def _on_browse_valid_data(self):
        """검증 데이터 CSV 찾기"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window,
            "검증 데이터 선택",
            self.data_dir,
            "CSV 파일 (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            print(f"📂 검증 데이터 선택: {file_path}")
            
            # 파일명만 표시
            filename = os.path.basename(file_path)
            if self.le_valid_data:
                self.le_valid_data.setText(filename)
            
            # 데이터 로드
            self._load_data(file_path, data_type="valid")
            
            if self.logger:
                self.logger.log_file(f"📂 검증 데이터 선택: {filename}")
    
    def _on_goto_judge(self):
        """균열 판정 페이지로 전환"""
        print("\n" + "="*50)
        print("🔘 pushButton_3 클릭됨!")
        
        # 🆕 탭 전환 전 자동 저장
        print("💾 자동 저장 시도...")
        self.save_current_project(silent=True)
        
        print("🔄 균열 판정 페이지로 전환 시도")
        print("="*50)
        
        # 프로젝트 정보 전달
        project_info = self.get_current_project_info()
        print(f"📋 프로젝트 정보 가져오기: {project_info is not None}")
        
        if project_info:
            print(f"   프로젝트명: {project_info.get('project_name', 'None')}")
            print(f"   생성날짜: {project_info.get('created_date', 'None')}")
            print(f"   모델파일: {project_info.get('model_filename', 'None')}")
            
            # JudgeInferenceController 확인 및 정보 전달
            if hasattr(self.main_window, '_judge_infer_ctrl'):
                print("✅ _judge_infer_ctrl 존재함")
                if hasattr(self.main_window._judge_infer_ctrl, 'set_project_info'):
                    print("✅ set_project_info 메서드 존재함")
                    self.main_window._judge_infer_ctrl.set_project_info(project_info)
                else:
                    print("❌ set_project_info 메서드 없음!")
            else:
                print("⚠️ _judge_infer_ctrl 없음 - 프로젝트 정보만 textBrowser에 직접 표시")
                # textBrowser에 직접 표시
                self._set_project_info_to_textbrowser(project_info)
            
            # 🆕 모델 자동 전달
            print(f"🔍 [DEBUG] _on_goto_judge 모델 확인:")
            print(f"   - self.current_model: {self.current_model is not None}")
            print(f"   - self.current_model_path: {self.current_model_path}")
            
            if self.current_model and self.current_model_path:
                print(f"📦 현재 로드된 모델: {os.path.basename(self.current_model_path)}")
                
                # JudgeModelController에 모델 전달
                if hasattr(self.main_window, '_judge_model_ctrl'):
                    judge_model_ctrl = self.main_window._judge_model_ctrl
                    judge_model_ctrl.model_path = self.current_model_path
                    judge_model_ctrl.loaded_model = self.current_model
                    print(f"✅ JudgeModelController에 모델 전달: {os.path.basename(self.current_model_path)}")
                    
                    # 모델 로드 시그널 emit
                    if hasattr(judge_model_ctrl, 'model_loaded'):
                        judge_model_ctrl.model_loaded.emit(self.current_model_path)
                        print("✅ model_loaded 시그널 emit")
                else:
                    print("⚠️ JudgeModelController 없음")
                    # JudgeInferenceController에 직접 전달
                    if hasattr(self.main_window, '_judge_infer_ctrl'):
                        judge_infer_ctrl = self.main_window._judge_infer_ctrl
                        judge_infer_ctrl.loaded_model = self.current_model
                        judge_infer_ctrl.model_path = self.current_model_path
                        print(f"✅ JudgeInferenceController에 직접 모델 전달: {os.path.basename(self.current_model_path)}")
                    else:
                        print("❌ JudgeInferenceController도 없음!")
            else:
                if not self.current_model:
                    print("⚠️ 로드된 모델 없음")
                if not self.current_model_path:
                    print("⚠️ 모델 경로 없음")
        else:
            print("⚠️ 프로젝트 정보가 없습니다. 먼저 프로젝트를 선택하세요.")
        
        # stack_pages 찾기
        stack_pages = self.main_window.findChild(QtWidgets.QStackedWidget, "stack_pages")
        if stack_pages:
            # page_judge 찾기
            for i in range(stack_pages.count()):
                widget = stack_pages.widget(i)
                if widget and widget.objectName() == "page_judge":
                    stack_pages.setCurrentIndex(i)
                    print(f"✅ 균열 판정 페이지로 전환 완료 (index={i})")
                    if self.logger:
                        self.logger.log_info("🔄 균열 판정 페이지로 전환")
                    return
        
        print("⚠️ page_judge를 찾을 수 없습니다.")
    
    def _set_project_info_to_textbrowser(self, project_info: dict):
        """textBrowser에 직접 프로젝트 정보 표시 (JudgeInferenceController 없을 때)"""
        print("\n📋 textBrowser에 직접 프로젝트 정보 표시")
        
        # page_judge에서 textBrowser들 찾기
        page_judge = None
        stack_pages = self.main_window.findChild(QtWidgets.QStackedWidget, "stack_pages")
        if stack_pages:
            for i in range(stack_pages.count()):
                widget = stack_pages.widget(i)
                if widget and widget.objectName() == "page_judge":
                    page_judge = widget
                    break
        
        if not page_judge:
            print("❌ page_judge를 찾을 수 없습니다")
            return
        
        # textBrowser들 찾아서 정보 표시
        textbrowsers = {
            "textBrowser_project": project_info.get("project_name", ""),
            "textBrowser_project_date": project_info.get("created_date", ""),
            "textBrowser_process": project_info.get("target_process", ""),
            "textBrowser_model": project_info.get("model_filename", ""),
            "textBrowser_purpose": project_info.get("purpose", ""),
            "textBrowser_algorithm": project_info.get("algorithm", ""),
            "textBrowser_model_date": project_info.get("model_date", "")
        }
        
        for name, value in textbrowsers.items():
            widget = page_judge.findChild(QtWidgets.QTextBrowser, name)
            if widget:
                widget.setText(value)
                print(f"   ✅ {name}: {value}")
            else:
                print(f"   ❌ {name} 못 찾음")
        
        print("📋 프로젝트 정보 표시 완료")
    
    def _on_goto_training(self):
        """모델 학습 페이지로 전환"""
        
        # 🆕 탭 전환 전 자동 저장
        print("💾 자동 저장 시도...")
        self.save_current_project(silent=True)
        
        print("🔄 모델 학습 페이지로 전환")
        
        # stack_pages 찾기
        stack_pages = self.main_window.findChild(QtWidgets.QStackedWidget, "stack_pages")
        if stack_pages:
            # page_train 찾기
            for i in range(stack_pages.count()):
                widget = stack_pages.widget(i)
                if widget and widget.objectName() == "page_train":
                    stack_pages.setCurrentIndex(i)
                    print(f"✅ 모델 학습 페이지로 전환 완료 (index={i})")
                    if self.logger:
                        self.logger.log_info("🔄 모델 학습 페이지로 전환")
                    return
        
        print("⚠️ page_train을 찾을 수 없습니다.")
    
    def reload_project_list(self):
        """프로젝트 리스트 새로고침"""
        self._load_project_list()    

    def save_current_project(self, silent=False):
        """
        현재 입력된 프로젝트 정보를 JSON으로 저장
        
        Args:
            silent: True면 메시지 없이 조용히 저장
        """
        print("\n💾 프로젝트 저장 중...")
        
        # 1. 필수 정보 확인 (프로젝트명만 필수!)
        project_name = self.le_name.text().strip() if self.le_name else ""
        
        if not project_name:
            if not silent:
                QtWidgets.QMessageBox.warning(
                    self.main_window,
                    "저장 불가",
                    "프로젝트명을 입력하세요."
                )
            return False
        
        # 2. 프로젝트 정보 수집 (안전하게!)
        from datetime import datetime
        
        def safe_get_text(widget):
            """위젯에서 안전하게 텍스트 가져오기"""
            if widget is None:
                return ""
            try:
                if hasattr(widget, 'text'):
                    text = widget.text()
                    return text.strip() if text else ""
                elif hasattr(widget, 'toPlainText'):
                    text = widget.toPlainText()
                    return text.strip() if text else ""
                else:
                    return ""
            except:
                return ""
        
        # 정보 수집
        model_filename = safe_get_text(self.le_model)
        
        project_data = {
            "project_filename": project_name,
            "project_name": project_name,
            "created_date": safe_get_text(self.le_date) or datetime.now().strftime("%Y-%m-%d"),
            "target_process": safe_get_text(self.le_process),
            "target_product": safe_get_text(self.le_target_product),
            "project_detail": "-",
            "model_filename": model_filename,
            "purpose": safe_get_text(self.le_purpose),
            "model_date": safe_get_text(self.le_model_date),
            "train_data_filename": os.path.basename(safe_get_text(self.le_train_data)),
            "valid_data_filename": os.path.basename(safe_get_text(self.le_valid_data))
        }
        
        # 3. JSON 파일명 결정 (프로젝트명 기준!)
        json_filename = f"{project_name}.json"
        json_path = os.path.join(self.projects_dir, json_filename)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 프로젝트 저장 완료: {json_filename}")
            
            # 4. 로그 추가
            if self.logger:
                self.logger.log_success(f"프로젝트 저장: {project_name}")
            
            # 5. 리스트 갱신
            self._load_project_list()
            
            # 6. 방금 저장한 프로젝트 자동 선택 (silent가 아닐 때만)
            if not silent:
                self._select_project_by_filename(json_filename)
            
            # 7. 성공 메시지 (silent가 아닐 때만)
            if not silent:
                QtWidgets.QMessageBox.information(
                    self.main_window,
                    "저장 완료",
                    f"프로젝트가 저장되었습니다!\n\n"
                    f"프로젝트: {project_name}\n"
                    f"파일: {json_filename}"
                )
            
            return True
            
        except Exception as e:
            print(f"❌ 프로젝트 저장 실패: {e}")
            if self.logger:
                self.logger.log_error(f"프로젝트 저장 실패: {e}")
            
            if not silent:
                QtWidgets.QMessageBox.critical(
                    self.main_window,
                    "저장 실패",
                    f"프로젝트 저장 중 오류가 발생했습니다:\n{str(e)}"
                )
            
            return False
    
    def _select_project_by_filename(self, json_filename: str):
        """
        파일명으로 프로젝트 리스트에서 찾아서 선택
        """
        if not self.list_projects:
            return
        
        # 리스트에서 해당 파일명을 가진 항목 찾기
        for i in range(self.list_projects.count()):
            item = self.list_projects.item(i)
            item_data = item.data(QtCore.Qt.UserRole)
            
            # 안전하게 체크
            if not item_data:
                continue
            
            try:
                # 딕셔너리인 경우
                if isinstance(item_data, dict):
                    json_path = item_data.get('json_path', '')
                    if json_path and os.path.basename(json_path) == json_filename:
                        self.list_projects.setCurrentItem(item)
                        self._on_project_selected(item)
                        print(f"✅ 프로젝트 자동 선택: {json_filename}")
                        break
                # 문자열인 경우 (레거시)
                elif isinstance(item_data, str):
                    if os.path.basename(item_data) == json_filename:
                        self.list_projects.setCurrentItem(item)
                        self._on_project_selected(item)
                        print(f"✅ 프로젝트 자동 선택: {json_filename}")
                        break
            except Exception as e:
                print(f"⚠️ 항목 확인 중 오류: {e}")
                continue
    def _on_goto_project(self):
        """프로젝트 설정 페이지로 전환"""
        print("\n" + "="*50)
        print("🔘 pushButton_7 또는 pushButton_8 클릭됨!")
        print("🔄 프로젝트 설정 페이지로 전환 시도")
        print("="*50)
        
        # stack_pages 찾기
        stack_pages = self.main_window.findChild(QtWidgets.QStackedWidget, "stack_pages")
        if stack_pages:
            # page_project 찾기
            for i in range(stack_pages.count()):
                widget = stack_pages.widget(i)
                if widget and widget.objectName() == "page_project":
                    stack_pages.setCurrentIndex(i)
                    print(f"✅ 프로젝트 설정 페이지로 전환 완료 (index={i})")
                    if self.logger:
                        self.logger.log_info("🔄 프로젝트 설정 페이지로 전환")
                    return
        
        print("⚠️ page_project를 찾을 수 없습니다.")