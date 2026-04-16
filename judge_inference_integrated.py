# -*- coding: utf-8 -*-
"""judge_inference.py - 범용 모델 추론 엔진 (JSON 내장 코드 지원 + 신호 전처리)"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional, Dict, Any, List
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QHeaderView
import importlib.util

from scipy.signal import hilbert, find_peaks, stft
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis, entropy
from scipy.integrate import trapezoid

# sklearn imports (pickle 호환성)
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# matplotlib for charts
# matplotlib for charts
import matplotlib
matplotlib.use('Qt5Agg')  # 🎯 백엔드 명시
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# 🎯 한글 폰트 설정 (전역 - 프로그램 시작 시 한 번만)
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        import matplotlib.font_manager as fm
        import platform
        import warnings
        
        # 한글 폰트 경고 무시
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
        
        system = platform.system()
        
        # Windows에서 폰트 파일 직접 지정
        if system == 'Windows':
            import os
            font_dirs = [
                r'C:\Windows\Fonts',
                os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
            ]
            
            font_files = {
                'Malgun Gothic': ['malgun.ttf', 'malgunbd.ttf'],
                'Gulim': ['gulim.ttc'],
                'Batang': ['batang.ttc'],
                'Dotum': ['dotum.ttc']
            }
            
            font_found = False
            for font_name, files in font_files.items():
                for font_file in files:
                    for font_dir in font_dirs:
                        font_path = os.path.join(font_dir, font_file)
                        if os.path.exists(font_path):
                            # 폰트 파일을 matplotlib에 추가
                            fm.fontManager.addfont(font_path)
                            plt.rcParams['font.family'] = font_name
                            print(f"✅ 한글 폰트 설정: {font_name} ({font_path})")
                            font_found = True
                            break
                    if font_found:
                        break
                if font_found:
                    break
            
            if not font_found:
                print("⚠️ Windows 폰트를 찾지 못했습니다. 기본 설정 사용")
                plt.rcParams['font.family'] = 'Malgun Gothic'
        
        elif system == 'Darwin':  # Mac
            plt.rcParams['font.family'] = 'AppleGothic'
            print("✅ 한글 폰트 설정: AppleGothic")
        
        else:  # Linux
            plt.rcParams['font.family'] = 'NanumGothic'
            print("✅ 한글 폰트 설정: NanumGothic")
        
        plt.rcParams['axes.unicode_minus'] = False
        
        # 폰트 캐시 재구성
        # fm._rebuild()
        
    except Exception as e:
        print(f"⚠️ 한글 폰트 설정 실패: {e}")
        import traceback
        traceback.print_exc()

# 프로그램 시작 시 한 번만 실행
setup_korean_font()


# ============================================
# 신호 전처리 함수
# ============================================
def preprocess_signal(signal, gain=1.0, clip_max=None, clip_min=None):
    """
    신호 전처리 - 2.5× 증폭/클리핑 제거, float32 변환만 수행
    """
    return np.asarray(signal, dtype=np.float32)


# ============================================
# 특징 추출 (기본 함수 - 폴백용)
# ============================================
def extract_all_features(signal, sampling_rate=1000000):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size < 2:
        return [0.0] * 23
    
    N = len(signal)
    
    peak_amp = np.max(signal)
    backwall_amp = signal[-1]
    TOF = np.argmax(signal) / sampling_rate * 1e6
    std_val = np.std(signal)
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    crest_factor = peak_amp / (rms_val + 1e-12)
    hilbert_auc = trapezoid(np.abs(hilbert(signal)))
    
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)[:N // 2]
    spectrum = 2.0 / N * np.abs(yf[:N // 2])
    
    spectrum_norm = spectrum / np.sum(spectrum + 1e-12)
    dominant_freq = xf[np.argmax(spectrum)]
    
    band_mask = (xf >= 0.2e6) & (xf <= 0.5e6)
    band_energy_ratio = np.sum(spectrum[band_mask] ** 2) / (np.sum(spectrum**2) + 1e-12)
    
    spectral_entropy = entropy(spectrum_norm)
    spectral_centroid = np.sum(xf * spectrum) / (np.sum(spectrum) + 1e-12)
    spectral_spread = np.sqrt(np.sum(((xf - spectral_centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-12))
    spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-12))) / (np.mean(spectrum) + 1e-12)
    peak_count = len(find_peaks(spectrum, height=np.max(spectrum) * 0.1)[0])
    spectral_kurtosis = kurtosis(spectrum)
    
    f, _, Zxx = stft(signal, fs=sampling_rate, nperseg=256, noverlap=96)
    Z = np.abs(Zxx)
    Z_norm = Z / (np.sum(Z, axis=0, keepdims=True) + 1e-12)
    
    dom_freq_t = np.mean(f[np.argmax(Z, axis=0)])
    energy_t = np.mean(np.sum(Z**2, axis=0))
    entropy_t = np.mean(-np.sum(Z_norm * np.log(Z_norm + 1e-12), axis=0))
    peak_trend_t = np.mean(np.sum(Z > (np.max(Z, axis=0, keepdims=True) * 0.3), axis=0))
    freq_shift_t = np.mean(np.abs(np.diff(f[np.argmax(Z, axis=0)], prepend=0)))
    spec_kurtosis_t = np.mean(kurtosis(Z, axis=0, fisher=True, bias=False))
    
    high_band_mask = f >= 100_000
    high_band_activity = np.mean(np.sum(Z[high_band_mask, :], axis=0) / (np.sum(Z, axis=0) + 1e-12))
    
    return [
        peak_amp, backwall_amp, TOF, std_val, skew_val, kurt_val, crest_factor,
        hilbert_auc, dominant_freq, band_energy_ratio, spectral_entropy,
        spectral_centroid, spectral_spread, spectral_flatness, peak_count,
        spectral_kurtosis, dom_freq_t, energy_t, entropy_t, peak_trend_t,
        freq_shift_t, spec_kurtosis_t, high_band_activity
    ]


feature_cols = [
    "peak_amp", "backwall_amp", "TOF", "std", "skew", "kurtosis",
    "crest_factor", "hilbert_auc", "dominant_freq", "band_energy_ratio",
    "spectral_entropy", "spectral_centroid", "spectral_spread", "spectral_flatness",
    "peak_count", "spectral_kurtosis", "dominant_freq(t)", "energy(t)", "entropy(t)",
    "peak_trend(t)", "freq_shift(t)", "spectral_kurtosis(t)", "high_band_activity(t)"
]


# ============================================
# SVM Seed Ensemble (기존 - 하위 호환성 유지)
# ============================================
class SeedEnsemble:
    """SVM 앙상블 래퍼 클래스 (기존 .pt 파일 호환용)"""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probs = []
        for m in self.models:
            probs.append(m.predict_proba(X))
        return np.mean(np.stack(probs, axis=0), axis=0)

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)


# ============================================
# Multi-Model Ensemble (신규 - LR/SVM/RF/XGB/MLP × N seeds)
# ============================================
class MultiModelEnsemble:
    """
    LR / SVM / RF / XGB / MLP × N seeds 앙상블
    training_1.py의 MultiModelEnsemble과 동일 구조 (pickle 호환)
    """
    MODEL_NAMES = ["LR", "SVM", "RF", "XGB", "MLP"]

    def __init__(self, ensemble_by_model: dict, model_names: list, scaler=None):
        self.ensemble_by_model = ensemble_by_model
        self.model_names = model_names
        self.scaler = scaler  # 외부 스케일러 (이미 fit된 상태)

    def _scale(self, X):
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def predict_proba_per_model(self, X):
        X_s = self._scale(X)
        model_probs = {}
        for name in self.model_names:
            pipes = self.ensemble_by_model.get(name, [])
            if not pipes:
                continue
            probs = np.mean(
                [p.predict_proba(X_s)[:, 1] for p in pipes], axis=0
            )
            model_probs[name] = probs
        return model_probs

    def predict_proba(self, X):
        """전체 앙상블 확률값 반환 (N×2 array) - SeedEnsemble과 동일 인터페이스"""
        model_probs = self.predict_proba_per_model(X)
        if not model_probs:
            n = len(X)
            return np.stack([np.ones(n) * 0.5, np.ones(n) * 0.5], axis=1)
        ensemble_probs = np.mean(list(model_probs.values()), axis=0)
        return np.stack([1 - ensemble_probs, ensemble_probs], axis=1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ============================================
# 파형 추출 헬퍼
# ============================================
def extract_waveform(df_row, wave_cols, mode, apply_preprocess=True):
    """
    파형 데이터 추출
    
    Parameters:
    -----------
    df_row : pd.Series
        DataFrame의 한 행
    wave_cols : list
        파형 컬럼 리스트
    mode : str
        "wave" 또는 "cols"
    apply_preprocess : bool
        전처리 적용 여부 (기본 True)
    """
    if mode == "wave" and "wave" in df_row.index:
        s = str(df_row["wave"]).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.replace(" ", "")
        w = np.fromstring(s, sep=",", dtype=np.float32)
        w = w[np.isfinite(w)]
        signal = w if w.size > 0 else np.zeros(2)
    else:
        amps = pd.to_numeric(df_row[wave_cols], errors="coerce").values
        amps = np.nan_to_num(amps, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        signal = amps
    
    # 🎯 전처리 적용 (2.5배 증폭 + 클리핑)
    if apply_preprocess:
        signal = preprocess_signal(signal)
    
    return signal


# ============================================
# 추론 엔진
# ============================================
class InferenceEngine:
    def __init__(self, model_path: str, json_path: str):
        self.model_path = model_path
        self.json_path = json_path
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.json_config = json.load(f)
        
        self.bundle = torch.load(model_path, map_location="cpu", weights_only=False)
        
        self.model_family = self._detect_model_family()

        # bundle이 raw state_dict이면 .get()이 없으므로 JSON에서 먼저 읽고 bundle은 폴백
        def _get(key, default=None):
            # JSON 중첩 구조 지원 (inference_config, stft_config 등)
            for section in [self.json_config,
                            self.json_config.get("inference_config", {}),
                            self.json_config.get("stft_config", {}),
                            self.json_config.get("model_config", {})]:
                if key in section:
                    return section[key]
            # bundle에서 폴백 (dict인 경우만)
            if isinstance(self.bundle, dict):
                return self.bundle.get(key, default)
            return default

        self.threshold     = _get("threshold",     0.5)
        self.sampling_rate = _get("sampling_rate", 1000000)
        self.wave_mode     = _get("wave_mode",     "cols")
        self.amp_cols      = _get("amp_cols",      [])
        self.speed_col     = _get("speed_col",     "속도")
        self.feature_cols  = _get("feature_cols",  feature_cols)
        
        # 기본 피처 추출 함수
        self.feature_extractor = None
        
        if self.model_family == "svm":
            self.ensemble = self.bundle.get("ensemble")
            # 🎯 SVM도 피처 로더 호출
            self._load_feature_extractor()
        elif self.model_family == "pytorch":
            self._load_pytorch_model()
    
    def _detect_model_family(self) -> str:
        """모델 종류 자동 감지"""
        # JSON에서 명시적으로 지정한 경우
        if "model_family" in self.json_config:
            family = self.json_config["model_family"]
            
            # "auto"면 자동 감지
            if family.lower() == "auto":
                print("📝 모델 타입 자동 감지 시작...")
                return self._auto_detect_from_bundle()
            
            if "svm" in family.lower() or "sklearn" in family.lower() or "multi_model" in family.lower():
                return "svm"
            elif "pytorch" in family.lower() or "torch" in family.lower():
                return "pytorch"
        
        # bundle에서 감지
        return self._auto_detect_from_bundle()
    
    def _auto_detect_from_bundle(self) -> str:
        """bundle 구조로부터 모델 타입 자동 감지"""
        # bundle의 model_family 확인
        bundle_family = self.bundle.get("model_family", "")
        if ("svm" in bundle_family.lower() or "sklearn" in bundle_family.lower()
                or "multi_model" in bundle_family.lower()):
            print(f"✅ Sklearn 계열 모델 감지 (bundle.model_family: {bundle_family})")
            return "svm"
        
        # ensemble 객체로 감지
        if "ensemble" in self.bundle:
            print("✅ SVM 모델 감지 (ensemble 존재)")
            return "svm"
        
        # state_dict 확인 (PyTorch)
        if isinstance(self.bundle, dict):
            # PyTorch state_dict 키 패턴 확인
            keys = list(self.bundle.keys())
            pytorch_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'loss']
            if any(k in keys for k in pytorch_keys):
                print("✅ PyTorch 모델 감지 (state_dict 패턴)")
                return "pytorch"
        
        # 기본값은 PyTorch
        print("⚠️  모델 타입 불명확 - PyTorch로 가정")
        return "pytorch"
    
    def _load_feature_extractor(self):
        """
        피처 추출 함수 로드 (JSON 내장 코드 우선)
        1. JSON의 feature_extraction.code (새 방식)
        2. JSON의 feature_extraction.file (기존 방식)
        3. 기본 함수 (폴백)
        """
        feature_config = self.json_config.get("feature_extraction", {})
        
        # 🎯 방법 1: JSON에 코드 내장 (우선)
        if "code" in feature_config:
            try:
                print("📝 JSON 내장 피처 코드 실행...")
                
                # 안전한 namespace 준비
                safe_globals = {
                    "np": np,
                    "numpy": np,
                    "scipy": __import__("scipy"),
                    "hilbert": hilbert,
                    "find_peaks": find_peaks,
                    "stft": stft,
                    "fft": fft,
                    "fftfreq": fftfreq,
                    "skew": skew,
                    "kurtosis": kurtosis,
                    "entropy": entropy,
                    "trapezoid": trapezoid,
                }
                
                # 코드 실행
                exec(feature_config["code"], safe_globals)
                
                # 함수 이름 가져오기 (기본: extract_features)
                func_name = feature_config.get("function", "extract_features")
                
                if func_name in safe_globals:
                    self.feature_extractor = safe_globals[func_name]
                    print(f"✅ JSON 내장 피처 함수 로드: {func_name}")
                else:
                    print(f"⚠️  함수 '{func_name}'를 찾을 수 없음")
                    print(f"ℹ️  기본 피처 함수 사용")
                    self.feature_extractor = None
                    
            except Exception as e:
                print(f"❌ JSON 피처 코드 실행 실패: {e}")
                print(f"ℹ️  기본 피처 함수 사용")
                import traceback
                traceback.print_exc()
                self.feature_extractor = None
                
        # 방법 2: 파일에서 로드 (하위 호환)
        elif "file" in feature_config:
            feature_file = feature_config["file"]
            feature_function = feature_config.get("function", "extract_features")
            
            try:
                # 상대 경로 처리
                if not os.path.isabs(feature_file):
                    json_dir = os.path.dirname(self.json_path)
                    feature_file = os.path.join(json_dir, feature_file)
                
                if not os.path.exists(feature_file):
                    print(f"⚠️  피처 파일을 찾을 수 없음: {feature_file}")
                    print(f"ℹ️  기본 피처 함수 사용")
                    self.feature_extractor = None
                    return
                
                print(f"📂 피처 파일 로드: {feature_file}")
                
                # .py 파일 동적 로드
                spec = importlib.util.spec_from_file_location("feature_module", feature_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 함수 가져오기
                if hasattr(module, feature_function):
                    self.feature_extractor = getattr(module, feature_function)
                    print(f"✅ 파일에서 피처 함수 로드: {feature_function}")
                else:
                    print(f"⚠️  함수 '{feature_function}'를 찾을 수 없음")
                    print(f"ℹ️  기본 피처 함수 사용")
                    self.feature_extractor = None
                    
            except Exception as e:
                print(f"❌ 피처 파일 로드 실패: {e}")
                print(f"ℹ️  기본 피처 함수 사용")
                self.feature_extractor = None
                
        # 방법 3: 기본 함수 사용
        else:
            self.feature_extractor = None
            if self.model_family == "svm":
                print(f"ℹ️  SVM: 기본 피처 함수 사용 (extract_all_features)")
    
    def _load_pytorch_model(self):
        """PyTorch 모델 동적 로드 (JSON 내장 코드 우선)"""
        try:
            # ── 아키텍처 클래스 결정 ─────────────────────────────
            # 방법 1: JSON에 architecture 코드 내장
            if "architecture" in self.json_config and "code" in self.json_config["architecture"]:
                print("📝 JSON 내장 아키텍처 코드 실행...")
                safe_globals = {
                    "torch": torch, "nn": nn, "np": np, "numpy": np,
                }
                exec(self.json_config["architecture"]["code"], safe_globals)
                model_class_name = self.json_config["architecture"].get("class_name", "CustomNet")
                if model_class_name not in safe_globals:
                    raise AttributeError(f"클래스 '{model_class_name}'를 찾을 수 없습니다.")
                ModelClass = safe_globals[model_class_name]
                print(f"✅ JSON 내장 모델 클래스 로드: {model_class_name}")
                self._load_feature_extractor()

            # 방법 2: ResNet18 자동 빌드 (JSON에 "backbone": "resnet18" 명시)
            elif self.json_config.get("backbone", "").lower() in ("resnet18", "resnet"):
                print("📝 ResNet18 자동 빌드...")
                import torchvision.models as tv_models

                mc = self.json_config.get("model_config", {})
                use_single_ch = mc.get("use_single_channel",
                                self.json_config.get("use_single_channel", True))
                num_classes   = mc.get("num_classes",
                                self.json_config.get("num_classes", 2))

                base = tv_models.resnet18(weights=None)

                # 1채널 패치
                if use_single_ch:
                    old = base.conv1
                    new_conv = nn.Conv2d(
                        1, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=(old.bias is not None)
                    )
                    with torch.no_grad():
                        new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                    base.conv1 = new_conv

                in_features = base.fc.in_features
                base.fc = nn.Sequential(
                    nn.Dropout(0.5), nn.Linear(in_features, 128), nn.ReLU(),
                    nn.Dropout(0.3), nn.Linear(128, num_classes),
                )

                # state_dict 로드 ─────────────────────────────────
                # ① bundle이 raw state_dict인 경우 (.pth 직접 저장)
                # ② bundle['state_dict'] 형식
                # ③ bundle['state_dicts'] 앙상블 형식
                raw_keys = list(self.bundle.keys()) if isinstance(self.bundle, dict) else []
                is_raw_sd = (isinstance(self.bundle, dict) and
                             any(k.startswith(("conv1", "bn1", "layer", "fc"))
                                 for k in raw_keys))

                if is_raw_sd:
                    print("🎯 raw state_dict 포맷 감지 (.pth 직접 저장)")
                    base.load_state_dict(self.bundle, strict=True)
                    base.eval()
                    for m in base.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()
                    self.models = [base]
                elif "state_dicts" in self.bundle:
                    self.models = []
                    for sd in self.bundle["state_dicts"]:
                        m = tv_models.resnet18(weights=None)
                        if use_single_ch:
                            m.conv1 = new_conv
                        m.fc = nn.Sequential(
                            nn.Dropout(0.5), nn.Linear(in_features, 128), nn.ReLU(),
                            nn.Dropout(0.3), nn.Linear(128, num_classes),
                        )
                        m.load_state_dict(sd, strict=False)
                        m.eval()
                        self.models.append(m)
                    print(f"🎯 앙상블 {len(self.models)}개 로드")
                else:
                    sd = self.bundle.get("state_dict", self.bundle)
                    base.load_state_dict(sd, strict=False)
                    base.eval()
                    self.models = [base]

                # ResNet18은 STFT 이미지 입력 → feature_extractor = None 으로 표시
                self.feature_extractor = None
                self.input_type = "stft"   # ← 추론 분기용

                # STFT 파라미터 (JSON stft_config 섹션 우선, 없으면 최상위, 없으면 기본값)
                sc = self.json_config.get("stft_config", {})
                self.stft_params = {
                    "nperseg":  sc.get("nperseg",  self.json_config.get("nperseg",  128)),
                    "noverlap": sc.get("noverlap", self.json_config.get("noverlap",  96)),
                    "nfft":     sc.get("nfft",     self.json_config.get("nfft",     256)),
                    "window":   sc.get("window",   self.json_config.get("window",  "hann")),
                    "img_size": sc.get("img_size", self.json_config.get("img_size", 224)),
                    "use_single_channel": use_single_ch,
                }
                print(f"✅ ResNet18 로드 완료 | STFT 파라미터: {self.stft_params}")
                return

            # 방법 3: architecture_file 참조 (하위 호환)
            else:
                arch_file = self.json_config.get("architecture_file")
                if arch_file is None:
                    raise ValueError("JSON에 'backbone', 'architecture_file', 또는 'architecture.code'가 필요합니다.")
                if not os.path.isabs(arch_file):
                    arch_file = os.path.join(os.path.dirname(self.json_path), arch_file)
                if not os.path.exists(arch_file):
                    raise FileNotFoundError(f"Architecture 파일을 찾을 수 없습니다: {arch_file}")
                spec = importlib.util.spec_from_file_location("model_arch", arch_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                feature_func_name = self.json_config.get("feature_extractor_function")
                self.feature_extractor = (getattr(module, feature_func_name)
                                          if feature_func_name and hasattr(module, feature_func_name)
                                          else None)
                model_class_name = self.json_config.get("model_class_name", "CustomNet")
                if not hasattr(module, model_class_name):
                    available = [n for n in dir(module) if not n.startswith("_")]
                    raise AttributeError(f"'{model_class_name}' 없음. 사용 가능: {available}")
                ModelClass = getattr(module, model_class_name)
                print(f"✅ 모델 클래스 로드: {model_class_name}")

            # ── 공통: state_dict 로드 (방법 1 / 방법 3 경로) ────
            init_args = (self.json_config.get("model_init_args")
                         or self.json_config.get("architecture", {}).get("init_args", {}))

            raw_keys = list(self.bundle.keys()) if isinstance(self.bundle, dict) else []
            is_raw_sd = (isinstance(self.bundle, dict) and
                         "state_dict" not in self.bundle and
                         "state_dicts" not in self.bundle and
                         "ensemble" not in self.bundle)

            if is_raw_sd:
                print("🎯 raw state_dict 포맷 감지")
                m = ModelClass(**init_args)
                m.load_state_dict(self.bundle, strict=False)
                m.eval()
                self.models = [m]
            elif "state_dicts" in self.bundle:
                self.models = []
                for idx, sd in enumerate(self.bundle["state_dicts"]):
                    m = ModelClass(**init_args)
                    m.load_state_dict(sd)
                    m.eval()
                    self.models.append(m)
                    print(f"  ✓ 모델 {idx+1} 로드")
                self.scaler_min   = self.bundle.get("scaler_min")
                self.scaler_scale = self.bundle.get("scaler_scale")
            elif "state_dict" in self.bundle:
                m = ModelClass(**init_args)
                m.load_state_dict(self.bundle["state_dict"])
                m.eval()
                self.models = [m]
                self.scaler_min   = self.bundle.get("scaler_min")
                self.scaler_scale = self.bundle.get("scaler_scale")
            else:
                raise KeyError("Bundle에 'state_dict' 또는 'state_dicts'가 없습니다.")

            self.input_type = self.json_config.get("input_type", "features")
            print(f"✅ PyTorch 모델 로드 완료 (총 {len(self.models)}개)")

        except Exception as e:
            print(f"❌ PyTorch 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """모델 추론"""
        if self.model_family == "svm":
            return self._predict_svm(features)
        elif self.model_family == "pytorch":
            return self._predict_pytorch(features)
        else:
            raise ValueError(f"알 수 없는 모델 패밀리: {self.model_family}")
    
    def _predict_svm(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SVM 앙상블 예측"""
        probs = self.ensemble.predict_proba(features)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        return preds, probs
    
    def _predict_pytorch(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """PyTorch 모델 예측 (STFT 이미지 모드 / 피처 모드 자동 분기)"""

        input_type = getattr(self, "input_type", "features")
        print(f"🔬 _predict_pytorch 진입: input_type={input_type}")
        
        # 🎯 backbone이 resnet18이면 무조건 stft로 강제
        backbone = getattr(self, "json_config", {}).get("backbone", "")
        print(f"🔬 backbone={backbone}")
        if backbone in ("resnet18", "resnet34", "resnet50") and input_type != "stft":
            print(f"⚠️ backbone={backbone}인데 input_type={input_type} → stft로 강제 변경")
            input_type = "stft"

        # ── STFT 이미지 모드 (ResNet18 등) ──────────────────────
        if input_type == "stft":
            return self._predict_stft(features)

        # ── 피처 모드 (기존) ─────────────────────────────────────
        is_raw_mode = (self.feature_extractor is None)

        if is_raw_mode:
            print("  Raw 파형 모드: 스케일링 생략")
            X_tensor = torch.FloatTensor(features)
        else:
            if self.scaler_min is not None and self.scaler_scale is not None:
                X_scaled = features * self.scaler_scale + self.scaler_min
            else:
                X_scaled = features
            X_tensor = torch.FloatTensor(X_scaled)

        probs_list = []
        with torch.no_grad():
            for model in self.models:
                probs_list.append(model(X_tensor).numpy())

        probs = np.mean(np.stack(probs_list, axis=0), axis=0).flatten()
        preds = (probs >= self.threshold).astype(int)
        print(f"🔬 _predict_pytorch 결과: probs 범위=[{probs.min():.4f}, {probs.max():.4f}], threshold={self.threshold}, crack={preds.sum()}/{len(preds)}")
        return preds, probs

    def _predict_stft(self, raw_signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        ResNet18 STFT 추론
        raw_signals: (N, signal_len) 원본 파형 배열
        """
        print(f"🔬 _predict_stft 호출됨: 입력 shape={raw_signals.shape}, min={raw_signals.min():.4f}, max={raw_signals.max():.4f}")
        from scipy.signal import stft as scipy_stft
        import torchvision.transforms as T

        p = getattr(self, "stft_params", {})
        nperseg  = p.get("nperseg",  128)
        noverlap = p.get("noverlap",  96)
        nfft     = p.get("nfft",     256)
        window   = p.get("window",  "hann")
        img_size = p.get("img_size", 224)
        use_single_ch = p.get("use_single_channel", True)

        resize    = T.Resize((img_size, img_size), antialias=True)
        if use_single_ch:
            normalize = T.Normalize(mean=[0.5], std=[0.5])
        else:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        tensors = []
        for sig in raw_signals:
            sig = np.asarray(sig, dtype=np.float32)
            _, _, Zxx = scipy_stft(sig, fs=self.sampling_rate,
                                   window=window, nperseg=nperseg,
                                   noverlap=noverlap, nfft=nfft)
            mag = 20 * np.log10(np.abs(Zxx) + 1e-10)
            s_min, s_max = mag.min(), mag.max()
            mag = (mag - s_min) / (s_max - s_min + 1e-8)

            t = torch.tensor(mag, dtype=torch.float32).unsqueeze(0)  # (1, F, T)
            t = resize(t)
            t = normalize(t)
            if not use_single_ch:
                t = t.repeat(3, 1, 1)
            tensors.append(t)

        X_tensor = torch.stack(tensors, dim=0)  # (N, C, H, W)

        probs_list = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(X_tensor)                         # (N, 2)
                prob   = torch.softmax(logits, dim=1)[:, 1]     # crack 확률
                probs_list.append(prob.numpy())

        probs = np.mean(np.stack(probs_list, axis=0), axis=0)
        preds = (probs >= self.threshold).astype(int)
        print(f"🔬 _predict_stft 결과: probs 범위=[{probs.min():.4f}, {probs.max():.4f}], threshold={self.threshold}, crack={preds.sum()}/{len(preds)}")
        return preds, probs


# ============================================
# Judge Inference 컨트롤러
# ============================================
class JudgeInferenceController(QtCore.QObject):
    def __init__(
        self,
        main_window,
        judge_model_ctrl,
        judge_data_ctrl,
        train_proj_ctrl=None,  # 🆕 TrainingProjectController 추가
        btn_judge_one=None,
        text_judge_summary=None,
        label_judge_chart=None,
        table_judge_stats_2=None,
        label_judge_chart_placeholder=None,
        table_judge_stats=None,
        # 프로젝트 정보 표시용 textBrowser들
        textBrowser_project=None,
        textBrowser_project_date=None,
        textBrowser_process=None,
        textBrowser_detail=None,  # 🆕 상세 설명
        textBrowser_model=None,
        textBrowser_purpose=None,
        textBrowser_product=None,  # 🆕 대상 제품 (algorithm → product)
        textBrowser_model_date=None,
        textBrowser_traindata=None,  # 🆕 학습 데이터
        textBrowser_validdata=None,  # 🆕 검증 데이터
        # 판정 결과 표시용 textBrowser들 (배치 단위)
        textBrowser_13=None,  # 전체 개수
        textBrowser_15=None,  # 정상 개수
        textBrowser_17=None,  # 불량 개수
        textBrowser_34=None,  # 불량률
        # 누적 통계 표시용 textBrowser들 (오늘 전체)
        textBrowser_32=None,  # 누적 전체 제품 수
        textBrowser_30=None,  # 누적 정상 제품 수
        textBrowser_31=None,  # 누적 불량 제품 수
        textBrowser_37=None,  # 누적 불량률
        label_judge_chart_placeholder_7=None,  # 누적 도넛 그래프
        # 불량 제품 리스트 위젯 (12개)
        textEdit_product1=None,
        textEdit_product2=None,
        textEdit_product3=None,
        textEdit_product4=None,
        textEdit_product5=None,
        textEdit_product6=None,
        textEdit_product7=None,
        textEdit_product8=None,
        textEdit_product9=None,
        textEdit_product10=None,
        textEdit_product11=None,
        textEdit_product12=None,
        btn_product_prev=None,
        btn_product_next=None,
        results_dir: str = None,
        respect_ui_settings: bool = True,  # 🎯 UI 설정 존중
        logger = None
    ):
        super().__init__(main_window)
        
        self.main_window = main_window
        self.model_ctrl = judge_model_ctrl
        self.data_ctrl = judge_data_ctrl
        self.train_proj_ctrl = train_proj_ctrl  # 🆕 TrainingProjectController
        self.btn_run = btn_judge_one
        self.text_summary = text_judge_summary
        self.label_chart = label_judge_chart
        self.table_stats_2 = table_judge_stats_2
        self.label_chart_all = label_judge_chart_placeholder
        self.table_stats = table_judge_stats
        
        # 프로젝트 정보 표시용
        self.textBrowser_project = textBrowser_project
        self.textBrowser_project_date = textBrowser_project_date
        self.textBrowser_process = textBrowser_process
        self.textBrowser_detail = textBrowser_detail  # 🆕 상세 설명
        self.textBrowser_model = textBrowser_model
        self.textBrowser_purpose = textBrowser_purpose
        self.textBrowser_product = textBrowser_product  # 🆕 대상 제품
        self.textBrowser_model_date = textBrowser_model_date
        self.textBrowser_traindata = textBrowser_traindata  # 🆕 학습 데이터
        self.textBrowser_validdata = textBrowser_validdata  # 🆕 검증 데이터
        
        # 판정 결과 표시용 (배치 단위)
        self.textBrowser_13 = textBrowser_13
        self.textBrowser_15 = textBrowser_15
        self.textBrowser_17 = textBrowser_17
        self.textBrowser_34 = textBrowser_34
        
        # 누적 통계 표시용 (오늘 전체)
        self.textBrowser_32 = textBrowser_32  # 누적 전체 제품 수
        self.textBrowser_30 = textBrowser_30  # 누적 정상 제품 수
        self.textBrowser_31 = textBrowser_31  # 누적 불량 제품 수
        self.textBrowser_37 = textBrowser_37  # 누적 불량률
        self.label_chart_all_cumulative = label_judge_chart_placeholder_7  # 누적 도넛 그래프
        
        # 불량 제품 리스트
        self.textEdit_products = [
            textEdit_product1, textEdit_product2, textEdit_product3, textEdit_product4,
            textEdit_product5, textEdit_product6, textEdit_product7, textEdit_product8,
            textEdit_product9, textEdit_product10, textEdit_product11, textEdit_product12
        ]
        self.btn_product_prev = btn_product_prev
        self.btn_product_next = btn_product_next
        self._defect_list = []
        self._defect_page = 0
        self._defect_per_page = 12
        
        self.results_dir = results_dir or os.path.join(os.getcwd(), "Results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.respect_ui_settings = bool(respect_ui_settings)  # 🎯
        self.logger = logger
        
        # 🔍 초기화 상태 확인
        print(f"\n🔍 JudgeInferenceController 초기화 상태:")
        print(f"   - btn_run (btn_judge_one): {self.btn_run is not None}")
        print(f"   - textBrowser_project: {self.textBrowser_project is not None}")
        print(f"   - textBrowser_13: {self.textBrowser_13 is not None}")
        
        self._connect_signals()
    
    def _set_text_centered(self, widget, text: str):
        """QTextBrowser에 수직 가운데 정렬된 텍스트 설정
        
        Args:
            widget: QTextBrowser 위젯
            text: 표시할 텍스트
        """
        if widget is None:
            return
        
        # HTML로 수직 가운데 정렬
        html = f'''
        <table width="100%" height="100%" cellpadding="0" cellspacing="0" border="0">
            <tr>
                <td align="left" valign="middle" style="padding-left: 5px;">
                    <font>{text}</font>
                </td>
            </tr>
        </table>
        '''
        widget.setHtml(html)
    
    
    def set_project_info(self, project_info: dict):
        """프로젝트 정보 표시"""
        print("\n📋 프로젝트 정보 표시 시작")
        print(f"   받은 정보: {project_info}")
        
        # textBrowser들 상태 확인
        print(f"\n   textBrowser 위젯 상태:")
        print(f"   - textBrowser_project: {self.textBrowser_project is not None}")
        print(f"   - textBrowser_project_date: {self.textBrowser_project_date is not None}")
        print(f"   - textBrowser_process: {self.textBrowser_process is not None}")
        print(f"   - textBrowser_detail: {self.textBrowser_detail is not None}")
        print(f"   - textBrowser_model: {self.textBrowser_model is not None}")
        print(f"   - textBrowser_purpose: {self.textBrowser_purpose is not None}")
        print(f"   - textBrowser_product: {self.textBrowser_product is not None}")
        print(f"   - textBrowser_model_date: {self.textBrowser_model_date is not None}")
        
        print(f"\n   정보 표시 중:")
        # textBrowser들에 정보 표시
        if self.textBrowser_project:
            # 🎯 프로젝트 파일명 표시 (project_name 대신)
            self._set_text_centered(self.textBrowser_project, project_info.get("project_filename", ""))
            print(f"   ✅ 프로젝트 파일명: {project_info.get('project_filename', '')}")
        else:
            print(f"   ❌ textBrowser_project가 None")
        
        if self.textBrowser_project_date:
            self._set_text_centered(self.textBrowser_project_date, project_info.get("created_date", ""))
            print(f"   ✅ 생성날짜: {project_info.get('created_date', '')}")
        else:
            print(f"   ❌ textBrowser_project_date가 None")
        
        if self.textBrowser_process:
            self._set_text_centered(self.textBrowser_process, project_info.get("target_process", ""))
            print(f"   ✅ 공정: {project_info.get('target_process', '')}")
        else:
            print(f"   ❌ textBrowser_process가 None")
        
        # textBrowser_detail은 UI에 없음 (None)
        if self.textBrowser_detail:
            detail_text = project_info.get("project_detail", "-")
            self._set_text_centered(self.textBrowser_detail, detail_text)
            print(f"   ✅ 상세 설명: {detail_text}")
        
        if self.textBrowser_model:
            self._set_text_centered(self.textBrowser_model, project_info.get("model_filename", ""))
            print(f"   ✅ 모델: {project_info.get('model_filename', '')}")
        else:
            print(f"   ❌ textBrowser_model가 None")
        
        # textBrowser_purpose는 UI에 없음 (None)
        if self.textBrowser_purpose:
            self._set_text_centered(self.textBrowser_purpose, project_info.get("purpose", ""))
            print(f"   ✅ 용도: {project_info.get('purpose', '')}")
        
        if self.textBrowser_product:
            # 🆕 대상 제품 표시 (algorithm → target_product)
            self._set_text_centered(self.textBrowser_product, project_info.get("target_product", ""))
            print(f"   ✅ 대상 제품: {project_info.get('target_product', '')}")
        else:
            print(f"   ❌ textBrowser_product가 None")
        
        if self.textBrowser_model_date:
            self._set_text_centered(self.textBrowser_model_date, project_info.get("model_date", ""))
            print(f"   ✅ 모델날짜: {project_info.get('model_date', '')}")
        else:
            print(f"   ❌ textBrowser_model_date가 None")
        
        if self.textBrowser_traindata:
            # 🆕 학습 데이터 파일명 표시
            self._set_text_centered(self.textBrowser_traindata, project_info.get("train_data_filename", ""))
            print(f"   ✅ 학습 데이터: {project_info.get('train_data_filename', '')}")
        else:
            print(f"   ❌ textBrowser_traindata가 None")
        
        if self.textBrowser_validdata:
            # 🆕 검증 데이터 파일명 표시
            self._set_text_centered(self.textBrowser_validdata, project_info.get("valid_data_filename", ""))
            print(f"   ✅ 검증 데이터: {project_info.get('valid_data_filename', '')}")
        else:
            print(f"   ❌ textBrowser_validdata가 None")
        
        print("📋 프로젝트 정보 표시 완료\n")
    
    def update_project_display(self, project_info: dict):
        """
        프로젝트 정보 업데이트 (set_project_info의 별칭)
        TrainingController와 동일한 인터페이스 제공
        """
        self.set_project_info(project_info)
    
    def create_donut_chart(self, normal_count: int, crack_count: int):
        """도넛 그래프 생성 및 표시"""
        import io
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import Qt
        
        total = normal_count + crack_count
        if total == 0:
            return
        
        crack_percent = (crack_count / total) * 100
        
        # 🎯 고정 크기로 차트 생성 (4x4 인치, 정사각형)
        fig = Figure(figsize=(4, 4), facecolor='white', dpi=100)
        ax = fig.add_subplot(111)
        
        # 데이터
        sizes = [normal_count, crack_count]
        colors = ['#4472C4', '#FF0000']  # 파란색, 빨간색
        
        # 도넛 차트 (레이블과 퍼센트 제거)
        wedges = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.6)  # 🎯 두께 조절
        )[0]  # wedges만 받기
        
        # 가운데 원 (더 작게 = 더 두꺼운 도넛)
        centre_circle = plt.Circle((0,0), 0.60, fc='white')  # 🎯 더 두껍게
        ax.add_artist(centre_circle)
        
        # 가운데 텍스트 (균열 퍼센트만 크게)
        ax.text(0, 0, f'{crack_percent:.1f}%', 
                ha='center', va='center', 
                fontsize=48, weight='bold', color= '#FF0000')  # 🎯 더 크게
        
        ax.axis('equal')
        fig.tight_layout(pad=0)  # 🎯 여백 제거
        
        # QPixmap으로 변환
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        
        # label_chart_all에 표시 (비율 유지하며 크기 조정)
        if self.label_chart_all:
            # 🎯 레이블 크기 확인
            label_size = self.label_chart_all.size()
            target_width = label_size.width() if label_size.width() > 0 else 400
            target_height = label_size.height() if label_size.height() > 0 else 400
            
            scaled_pixmap = pixmap.scaled(
                target_width, target_height,
                Qt.KeepAspectRatio,  # 🎯 비율 유지
                Qt.SmoothTransformation
            )
            self.label_chart_all.setPixmap(scaled_pixmap)
            self.label_chart_all.setAlignment(Qt.AlignCenter)  # 🎯 중앙 정렬
            
            print(f"   차트 크기: 원본 {pixmap.width()}x{pixmap.height()} → 조정 {scaled_pixmap.width()}x{scaled_pixmap.height()}")
        
        plt.close(fig)
        
        print(f"📊 도넛 그래프 생성 완료: 정상 {normal_count}개, 균열 {crack_count}개 ({crack_percent:.1f}%)")
    
    def _display_defect_products(self, crack_df: pd.DataFrame, name_col: Optional[str]):
        """불량 제품 리스트를 textEdit_product1~12에 페이지네이션으로 표시"""
        print(f"\n📋 불량 제품 리스트 표시 시작")
        print(f"   - crack_df 크기: {len(crack_df)}")
        print(f"   - name_col: {name_col}")
        print(f"   - textEdit_products 개수: {len(self.textEdit_products)}")

        self._defect_list = []
        for _, row in crack_df.iterrows():
            if name_col and name_col in row.index:
                product_name = str(row[name_col])
            else:
                product_name = f"Index {row.name}"
            self._defect_list.append(product_name)

        print(f"   - 생성된 불량 제품 개수: {len(self._defect_list)}")

        self._defect_page = 0
        self._update_defect_page()

    def _update_defect_page(self):
        """현재 페이지 표시 + 버튼 상태 업데이트"""
        total = len(self._defect_list)
        per_page = self._defect_per_page
        total_pages = max(1, (total + per_page - 1) // per_page)

        start = self._defect_page * per_page
        end = min(start + per_page, total)
        page_items = self._defect_list[start:end]

        for i in range(per_page):
            widget = self.textEdit_products[i] if i < len(self.textEdit_products) else None
            if widget is not None:
                if i < len(page_items):
                    widget.setPlainText(page_items[i])
                    print(f"   ✅ textEdit_product{i+1}: {page_items[i]}")
                else:
                    widget.clear()
                    print(f"   ⬜ textEdit_product{i+1}: (빈칸)")

        if self.btn_product_prev:
            self.btn_product_prev.setEnabled(self._defect_page > 0)
        if self.btn_product_next:
            self.btn_product_next.setEnabled(self._defect_page < total_pages - 1)

        print(f"📋 불량 제품 리스트 표시 완료: {len(page_items)}개 (페이지 {self._defect_page+1}/{total_pages})\n")

    def _prev_defect_page(self):
        if self._defect_page > 0:
            self._defect_page -= 1
            self._update_defect_page()

    def _next_defect_page(self):
        total = len(self._defect_list)
        total_pages = max(1, (total + self._defect_per_page - 1) // self._defect_per_page)
        if self._defect_page < total_pages - 1:
            self._defect_page += 1
            self._update_defect_page()
    
    def _connect_signals(self):
        print("\n🔗 JudgeInferenceController 시그널 연결 중...")
        if self.btn_run:
            self.btn_run.clicked.connect(self.run_inference)
            print("   ✅ btn_judge_one (판정하기) 연결됨")
        else:
            print("   ❌ btn_judge_one 없음")
        if self.btn_product_prev:
            self.btn_product_prev.clicked.connect(self._prev_defect_page)
            print("   ✅ btn_product_prev 연결됨")
        if self.btn_product_next:
            self.btn_product_next.clicked.connect(self._next_defect_page)
            print("   ✅ btn_product_next 연결됨")
        print("🔗 JudgeInferenceController 시그널 연결 완료\n")
    
    def run_inference(self):
        print("\n" + "="*50)
        print("🔘 btn_judge_one 클릭됨!")
        print("📊 판정 시작")
        print("="*50)
        
        # 🎯 모델 경로 가져오기 (우선순위: self.model_path > model_ctrl > train_proj_ctrl)
        model_path = None
        
        # 1. 직접 전달된 모델 경로 확인 (최우선)
        if hasattr(self, 'model_path') and self.model_path:
            model_path = self.model_path
            print(f"✅ 직접 전달된 모델 사용: {model_path}")
        
        # 2. model_ctrl에서 확인
        elif self.model_ctrl:
            model_path = self.model_ctrl.get_current_model_path()
            if model_path:
                print(f"✅ model_ctrl에서 모델 로드: {model_path}")
        
        # 3. train_proj_ctrl에서 확인
        elif self.train_proj_ctrl:
            project_data = self.train_proj_ctrl.get_current_project_data()
            if project_data:
                model_info = project_data.get("model", {})
                model_path = model_info.get("model_path", "")
                if model_path:
                    print(f"✅ train_proj_ctrl에서 모델 로드: {model_path}")
        
        if not model_path or not os.path.exists(model_path):
            print("❌ 모델이 로드되지 않음")
            print(f"   - self.model_path: {getattr(self, 'model_path', None)}")
            print(f"   - self.model_ctrl: {self.model_ctrl is not None if self.model_ctrl else False}")
            print(f"   - self.train_proj_ctrl: {self.train_proj_ctrl is not None if self.train_proj_ctrl else False}")
            QtWidgets.QMessageBox.warning(
                self.main_window, 
                "모델 없음", 
                "먼저 프로젝트를 로드하거나 모델을 로드하세요.\n\n"
                "프로젝트 탭에서 프로젝트를 선택하면 자동으로 모델이 로드됩니다."
            )
            return
        
        # 데이터 확인 (판정 탭에서 직접 불러온 데이터만 사용)
        df = None
        if self.data_ctrl:
            df = self.data_ctrl.df
            if df is not None:
                print(f"✅ 판정 데이터 로드됨: {len(df)}행")
        
        if df is None:
            print("❌ 데이터가 로드되지 않음")
            print("   판정 탭에서 CSV 파일을 직접 로드하세요.")
            QtWidgets.QMessageBox.warning(
                self.main_window, 
                "데이터 없음", 
                "판정 탭에서 CSV 데이터를 먼저 로드하세요.\n\n"
                "데이터 로드 버튼을 클릭하여 판정할 데이터를 불러오세요."
            )
            return
        
        print(f"✅ 모델: {model_path}")
        print(f"✅ 데이터: {len(df)}행")
        
        json_path = model_path.rsplit(".", 1)[0] + ".json"
        if not os.path.exists(json_path):
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "JSON 파일 없음",
                f"모델과 동일 이름의 JSON 파일이 필요합니다:\n{json_path}"
            )
            return
        
        try:
            start_time = datetime.now()
            self._set_status("추론 시작...")
            
            # 🎯 로그 추가
            if self.logger:
                self.logger.log_start("추론 시작...")
            
            engine = InferenceEngine(model_path, json_path)
            
            wave_cols = self.data_ctrl.wave_cols
            mode = self.data_ctrl.mode
            name_col = self.data_ctrl.name_col
            
            # ============================================
            # 🎯 핵심: SVM/PyTorch 모두 커스텀 피처 지원
            # ============================================
            
            if engine.model_family == "svm":
                # SVM: 피처 추출
                print("📊 SVM 모드: 피처 추출 진행...")
                
                if engine.speed_col and engine.speed_col in df.columns:
                    speed = pd.to_numeric(df[engine.speed_col], errors="coerce").fillna(0).values
                else:
                    speed = np.zeros(len(df), dtype=np.float32)
                
                feats = []
                for i in range(len(df)):
                    # 🎯 전처리 적용 (2.5배 + 클리핑)
                    wave = extract_waveform(df.iloc[i], wave_cols, mode, apply_preprocess=True)
                    
                    # 🎯 커스텀 피처 함수가 있으면 사용, 없으면 기본 함수
                    if engine.feature_extractor is not None:
                        feat = engine.feature_extractor(wave, sampling_rate=engine.sampling_rate)
                    else:
                        feat = extract_all_features(wave, sampling_rate=engine.sampling_rate)
                    
                    feats.append(feat)
                
                feat_df = pd.DataFrame(np.asarray(feats), columns=engine.feature_cols)
                X = feat_df.values
            
            elif engine.model_family == "pytorch":
                if engine.feature_extractor is not None:
                    print("📊 PyTorch 피처 추출 모드...")
                    feats = []
                    for i in range(len(df)):
                        wave = extract_waveform(df.iloc[i], wave_cols, mode, apply_preprocess=True)
                        feat = engine.feature_extractor(wave, sampling_rate=engine.sampling_rate)
                        feats.append(feat)
                    feat_df = pd.DataFrame(np.asarray(feats), columns=engine.feature_cols)
                    X = feat_df.values
                
                else:
                    # Raw 파형 모드 (CNN/LSTM/Transformer 등)
                    print("📊 PyTorch Raw 파형 모드...")
                    
                    # 🎯 첫 번째 파형 추출해서 실제 길이 파악 (JSON에 없으면 실제 신호 길이 사용)
                    first_wave = extract_waveform(df.iloc[0], wave_cols, mode, apply_preprocess=True)
                    actual_length = len(first_wave)
                    waveform_length = engine.json_config.get(
                        "waveform_length",
                        engine.json_config.get("inference_config", {}).get("waveform_length", actual_length)
                    )
                    print(f"  파형 길이: {waveform_length} (실제 신호 길이: {actual_length})")
                    
                    waves = []
                    for i in range(len(df)):
                        # 🎯 전처리 적용
                        wave = extract_waveform(df.iloc[i], wave_cols, mode, apply_preprocess=True)
                        
                        # 길이 통일 (패딩 또는 자르기)
                        if len(wave) < waveform_length:
                            wave = np.pad(wave, (0, waveform_length - len(wave)), mode='constant')
                        elif len(wave) > waveform_length:
                            wave = wave[:waveform_length]
                        
                        waves.append(wave)
                    
                    X = np.array(waves, dtype=np.float32)  # (N, waveform_length)
                    print(f"✅ Raw 파형 배열 생성: {X.shape}")
            
            else:
                raise ValueError(f"알 수 없는 모델 패밀리: {engine.model_family}")
            
            self._set_status("모델 추론 중...")
            preds, probs = engine.predict(X)
            
            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            
            results_df = df.copy()
            results_df["판정시간"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
            results_df["판정"] = ["균열" if p == 1 else "정상" for p in preds]
            results_df["확률"] = probs
            results_df["판정근거"] = [self._generate_reason(p, engine.threshold) for p in probs]
            
            # 일별 CSV 저장 (append 모드)
            date_str = datetime.now().strftime("%Y%m%d")
            result_path = os.path.join(self.results_dir, f"result_{date_str}.csv")
            
            if os.path.exists(result_path):
                results_df.to_csv(result_path, mode='a', header=False, index=False, encoding="utf-8-sig")
            else:
                results_df.to_csv(result_path, mode='w', header=True, index=False, encoding="utf-8-sig")
            
            crack_df = results_df[results_df["판정"] == "균열"].reset_index(drop=True)
            
            # 현재 배치 통계
            total_count = len(df)
            crack_count = len(crack_df)
            normal_count = total_count - crack_count
            defect_rate = crack_count / total_count if total_count > 0 else 0
            
            self._display_summary(crack_df, name_col)
            self._display_chart(self.label_chart, normal_count, crack_count, defect_rate)
            self._display_stats(self.table_stats_2, inference_time, total_count, normal_count, crack_count, defect_rate)
            
            # 🎯 판정 결과를 textBrowser에 표시
            if self.textBrowser_13:  # 전체 개수
                self._set_text_centered(self.textBrowser_13, str(total_count))
            if self.textBrowser_15:  # 정상 개수
                self._set_text_centered(self.textBrowser_15, str(normal_count))
            if self.textBrowser_17:  # 불량 개수
                self._set_text_centered(self.textBrowser_17, str(crack_count))
            if self.textBrowser_34:  # 불량률
                self._set_text_centered(self.textBrowser_34, f"{defect_rate*100:.2f}%")
            
            # 🎯 도넛 그래프 생성 및 표시
            print(f"\n📊 현재 Batch 도넛 그래프 생성:")
            print(f"   - 전체: {total_count}개")
            print(f"   - 정상: {normal_count}개") 
            print(f"   - 불량: {crack_count}개")
            print(f"   - 불량률: {defect_rate*100:.2f}%")
            self.create_donut_chart(normal_count, crack_count)
            
            # 🆕 불량 제품 리스트 표시 (batch 기준)
            self._display_defect_products(crack_df, name_col)
            
            print(f"\n📊 판정 결과:")
            print(f"   - 전체: {total_count}개")
            print(f"   - 정상: {normal_count}개")
            print(f"   - 불량: {crack_count}개")
            print(f"   - 불량률: {defect_rate*100:.2f}%")
            
            # 전체 누적 통계
            self._display_cumulative_stats()
            
            self._set_status(f"추론 완료! 균열 {crack_count}개 / 전체 {total_count}개 검출")
            
            # 🎯 로그: 판정 완료 메시지
            if self.logger:
                self.logger.separator()
                self.logger.log_success(f"✅ 판정 완료!")
                self.logger.log_info(f"전체: {total_count}개 | 정상: {normal_count}개 | 불량: {crack_count}개 | 불량률: {defect_rate*100:.2f}%")
                self.logger.separator()
            
            QtWidgets.QMessageBox.information(
                self.main_window,
                "추론 완료",
                f"추론이 완료되었습니다.\n\n"
                f"전체: {total_count}개\n"
                f"균열: {crack_count}개\n"
                f"정상: {normal_count}개\n"
                f"불량률: {defect_rate*100:.2f}%\n\n"
                f"결과 저장: {result_path}"
            )
            
        except Exception as e:
            # 🎯 로그 추가
            if self.logger:
                self.logger.log_error(f"판정 실패: {str(e)}")
            
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "추론 실패",
                f"추론 중 오류가 발생했습니다:\n{e}"
            )
            import traceback
            traceback.print_exc()
            self._set_status("추론 실패")
    
    def _generate_reason(self, prob: float, threshold: float) -> str:
        if prob >= threshold:
            return f"균열 가능성 높음 (확률: {prob:.1%}, 기준: {threshold:.1%})"
        else:
            return f"정상 범위 (확률: {prob:.1%}, 기준: {threshold:.1%})"
    
    def _display_summary(self, crack_df: pd.DataFrame, name_col: Optional[str]):
        """균열 판정 결과를 table_judge_stats_3에 표시"""
        if self.text_summary is None:
            return
        
        if len(crack_df) == 0:
            self.text_summary.setRowCount(1)
            self.text_summary.setColumnCount(1)
            self.text_summary.setHorizontalHeaderLabels(["메시지"])
            item = QtWidgets.QTableWidgetItem("균열로 판정된 데이터가 없습니다.")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.text_summary.setItem(0, 0, item)
            return
        
        # 테이블 설정
        self.text_summary.setRowCount(len(crack_df))
        self.text_summary.setColumnCount(2)
        # self.text_summary.setSectionResizeMode(1,QHeaderView.ResizeToContents)
        self.text_summary.setHorizontalHeaderLabels(["부분 이름", "판정 근거"])
        
        
        # 데이터 채우기
        for idx, (_, row) in enumerate(crack_df.iterrows()):
            if name_col and name_col in row.index:
                name = str(row[name_col])
            else:
                name = f"Index {idx}"
            
            reason = str(row.get("판정근거", ""))
            
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.text_summary.setItem(idx, 0, name_item)
            
            reason_item = QtWidgets.QTableWidgetItem(reason)
            reason_item.setFlags(reason_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.text_summary.setItem(idx, 1, reason_item)
        
        # 🎯 UI 설정 존중 시 자동 조정 안 함
        if not self.respect_ui_settings:
            self.text_summary.resizeColumnsToContents()
            self.text_summary.resizeRowsToContents()
    
    def _display_chart(self, label_widget, normal_count: int, crack_count: int, defect_rate: float):
        if label_widget is None:
            return
        
        # 기존 위젯 제거
        layout = label_widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            layout = QtWidgets.QVBoxLayout(label_widget)
            label_widget.setLayout(layout)
        
        # 🎯 UI 설정을 존중하지 않을 때만 한글 폰트 설정
        if not self.respect_ui_settings:
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
        
        # 🎯 고정 크기로 차트 생성 (정사각형)
        fig = Figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        sizes = [normal_count, crack_count]
        colors = ['#3b82f6', '#FF6600']
        labels = ['정상', '균열']
        
        # 퍼센트와 레이블 분리
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val}개)'
            return my_autopct
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=make_autopct(sizes),
            startangle=90,
            wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2),
            pctdistance=0.75,
            labeldistance=1.1
        )
        
        # 레이블 폰트
        for text in texts:
            text.set_fontsize(11)
            text.set_weight('bold')
        
        # 퍼센트 폰트
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
        
        # 중앙에 정확도 표시
        accuracy = 1 - defect_rate
        ax.text(0, 0.05, f'{accuracy*100:.1f}%', ha='center', va='center', 
                fontsize=24, weight='bold', color='#1f2937')
        ax.text(0, -0.15, '정확도', ha='center', va='center', 
                fontsize=11, color='#6b7280')
        
        # 여백 조정
        fig.tight_layout(pad=0.5)  # 🎯 여백 최소화
        
        canvas = FigureCanvasQTAgg(fig)
        
        # 🎯 canvas 크기 고정 (정사각형 유지)
        canvas.setFixedSize(400, 400)  # 고정 크기
        
        layout.addWidget(canvas)
    
    def _display_stats(self, table_widget, inference_time: float, total: int, normal: int, crack: int, defect_rate: float):
        if table_widget is None:
            return
        
        table_widget.setRowCount(5)
        table_widget.setColumnCount(2)
        table_widget.setHorizontalHeaderLabels(["항목", "값"])
        
        stats_data = [
            ("판정 시간", f"{inference_time:.2f}초"),
            ("총 제품", f"{total}개"),
            ("정상 제품수", f"{normal}개"),
            ("균열 제품수", f"{crack}개"),
            ("불량률", f"{defect_rate*100:.2f}%")
        ]
        
        for row_idx, (key, value) in enumerate(stats_data):
            table_widget.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(key))
            table_widget.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(value))
        
        # 🎯 UI 설정 존중 시 자동 조정 안 함
        if not self.respect_ui_settings:
            table_widget.resizeColumnsToContents()
    
    def _display_cumulative_stats(self):
        """Results 폴더에서 오늘 날짜의 모든 CSV 파일을 불러와서 누적 통계 표시"""
        try:
            # 🎯 오늘 날짜 문자열 생성 (예: 20250112)
            today_str = datetime.now().strftime("%Y%m%d")
            
            print(f"\n📊 누적 통계 계산 중... (오늘 날짜: {today_str})")
            
            # 🆕 DAY 계산: Results 폴더의 고유 날짜 개수
            unique_dates = set()
            all_results = []
            
            for filename in os.listdir(self.results_dir):
                if filename.endswith(".csv") and "result_" in filename:
                    try:
                        # result_20250112_... 형식에서 날짜 추출
                        date_part = filename.split("_")[1][:8]
                        if len(date_part) == 8 and date_part.isdigit():
                            unique_dates.add(date_part)
                            
                            # 🎯 오늘 날짜가 포함된 CSV 파일만 불러오기
                            if today_str in filename:
                                filepath = os.path.join(self.results_dir, filename)
                                df = pd.read_csv(filepath, encoding="utf-8-sig")
                                all_results.append(df)
                                print(f"   ✅ 파일 로드: {filename} ({len(df)}행)")
                    except Exception as e:
                        print(f"   ⚠️ 파일 처리 실패: {filename} - {e}")
                        pass
            
            day_count = len(unique_dates)
            print(f"   📅 누적 DAY: {day_count}일")
            
            # 오늘 날짜의 결과가 없으면 초기화
            if not all_results:
                print("   ⚠️ 오늘 날짜의 판정 결과가 없습니다.")
                
                if self.label_chart_all_cumulative:
                    self.label_chart_all_cumulative.clear()
                    self.label_chart_all_cumulative.setText("오늘 날짜의 판정 결과가 없습니다.")
                
                if self.textBrowser_32:
                    self._set_text_centered(self.textBrowser_32, "0")
                if self.textBrowser_30:
                    self._set_text_centered(self.textBrowser_30, "0")
                if self.textBrowser_31:
                    self._set_text_centered(self.textBrowser_31, "0")
                if self.textBrowser_37:
                    self._set_text_centered(self.textBrowser_37, "0.00%")
                
                return
            
            # 🎯 오늘 날짜의 모든 결과 합치기
            combined_df = pd.concat(all_results, ignore_index=True)
            
            print(f"\n📊 누적 데이터 확인:")
            print(f"   - 로드한 CSV 파일 수: {len(all_results)}")
            print(f"   - 합친 DataFrame 행 수: {len(combined_df)}")
            
            # 🎯 누적 통계 계산
            total_count = len(combined_df)
            crack_count = len(combined_df[combined_df["판정"] == "균열"])
            normal_count = total_count - crack_count
            defect_rate = crack_count / total_count if total_count > 0 else 0
            
            print(f"\n📊 오늘 누적 통계:")
            print(f"   - 전체: {total_count}개")
            print(f"   - 정상: {normal_count}개")
            print(f"   - 불량: {crack_count}개")
            print(f"   - 불량률: {defect_rate*100:.2f}%")
            
            # 🎯 textBrowser에 누적 통계 표시
            if self.textBrowser_32:  # 누적 전체 제품 수
                self._set_text_centered(self.textBrowser_32, str(total_count))
            
            if self.textBrowser_30:  # 누적 정상 제품 수
                self._set_text_centered(self.textBrowser_30, str(normal_count))
            
            if self.textBrowser_31:  # 누적 불량 제품 수
                self._set_text_centered(self.textBrowser_31, str(crack_count))
            
            if self.textBrowser_37:  # 누적 불량률
                self._set_text_centered(self.textBrowser_37, f"{defect_rate*100:.2f}%")
            
            # 🎯 누적 도넛 그래프 생성 (label_judge_chart_placeholder_7)
            print(f"\n📊 누적 도넛 그래프 생성:")
            print(f"   - 전체: {total_count}개")
            print(f"   - 정상: {normal_count}개")
            print(f"   - 불량: {crack_count}개")
            print(f"   - 불량률: {defect_rate*100:.2f}%")
            self._create_cumulative_donut_chart(normal_count, crack_count)
            
            print("✅ 누적 통계 표시 완료\n")
        
        except Exception as e:
            print(f"❌ 누적 통계 표시 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_cumulative_donut_chart(self, normal_count: int, crack_count: int):
        """누적 도넛 그래프 생성 및 표시 (label_judge_chart_placeholder_7)"""
        import io
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import Qt
        
        total = normal_count + crack_count
        if total == 0:
            if self.label_chart_all_cumulative:
                self.label_chart_all_cumulative.clear()
                self.label_chart_all_cumulative.setText("데이터 없음")
            return
        
        crack_percent = (crack_count / total) * 100
        
        # 🎯 고정 크기로 차트 생성 (4x4 인치, 정사각형)
        fig = Figure(figsize=(4, 4), facecolor='white', dpi=100)
        ax = fig.add_subplot(111)
        
        # 데이터
        sizes = [normal_count, crack_count]
        colors = ['#4472C4', '#FF0000']  # 파란색, 빨간색
        
        # 도넛 차트 (레이블과 퍼센트 제거)
        wedges = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.6)
        )[0]
        
        # 가운데 원
        centre_circle = plt.Circle((0,0), 0.60, fc='white')
        ax.add_artist(centre_circle)
        
        # 가운데 텍스트 (균열 퍼센트만 크게)
        ax.text(0, 0, f'{crack_percent:.1f}%', 
                ha='center', va='center', 
                fontsize=48, weight='bold', color='#FF0000')
        
        ax.axis('equal')
        fig.tight_layout(pad=0)
        
        # QPixmap으로 변환
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        
        # label_chart_all_cumulative에 표시
        if self.label_chart_all_cumulative:
            label_size = self.label_chart_all_cumulative.size()
            target_width = label_size.width() if label_size.width() > 0 else 400
            target_height = label_size.height() if label_size.height() > 0 else 400
            
            scaled_pixmap = pixmap.scaled(
                target_width, target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label_chart_all_cumulative.setPixmap(scaled_pixmap)
            self.label_chart_all_cumulative.setAlignment(Qt.AlignCenter)
            
            print(f"   📊 누적 도넛 그래프 생성 완료: 정상 {normal_count}개, 균열 {crack_count}개 ({crack_percent:.1f}%)")
        
        plt.close(fig)
    
    def _set_status(self, msg: str):
        try:
            sb = self.main_window.statusBar()
            if sb:
                sb.showMessage(msg, 5000)
        except:
            pass

# ============================================
# SimpleJudgeController - 통합 판정 컨트롤러
# (CSV 저장, 누적 결과, 불량 제품 페이지네이션 포함)
# ============================================
class SimpleJudgeController(QtCore.QObject):
    """
    프로젝트에서 모델 경로를 가져와 판정하는 통합 컨트롤러
    - 원본 CSV 형식 유지하며 판정 컬럼 추가 (result_YYYYMMDD.csv)
    - 누적 결과 표시 (오늘 날짜 기준)
    - 불량 제품 페이지네이션 (8개씩)
    """
    
    def __init__(
        self,
        main_window,
        train_proj_ctrl,  # TrainingProjectController
        judge_data_ctrl,  # JudgeDataController
        btn_judge_one=None,
        model_ctrl=None,  # JudgeModelController (판정 탭 모델 로드 버튼)
        label_chart_placeholder=None,
        label_judge_chart_placeholder_7=None,
        textBrowser_13=None,
        textBrowser_15=None,
        textBrowser_17=None,
        textBrowser_34=None,
        # 누적 결과용
        textBrowser_32=None,
        textBrowser_30=None,
        textBrowser_28=None,
        textBrowser_37=None,
        # 불량 제품 리스트
        textEdit_product1=None,
        textEdit_product2=None,
        textEdit_product3=None,
        textEdit_product4=None,
        textEdit_product5=None,
        textEdit_product6=None,
        textEdit_product7=None,
        textEdit_product8=None,
        textEdit_score1=None,
        textEdit_score2=None,
        textEdit_score3=None,
        textEdit_score4=None,
        textEdit_score5=None,
        textEdit_score6=None,
        textEdit_score7=None,
        textEdit_score8=None,
        stackedWidget_defect=None,
        pushButton_after=None,
        pushButton_prev=None,
        logger=None
    ):
        super().__init__(main_window)
        
        self.main_window = main_window
        self.train_proj_ctrl = train_proj_ctrl
        self.data_ctrl = judge_data_ctrl
        self.model_ctrl = model_ctrl  # JudgeModelController 폴백
        self.btn_judge_one = btn_judge_one
        self.label_chart_placeholder = label_chart_placeholder
        self.label_judge_chart_placeholder_7 = label_judge_chart_placeholder_7
        
        # 현재 판정 결과용
        self.textBrowser_13 = textBrowser_13
        self.textBrowser_15 = textBrowser_15
        self.textBrowser_17 = textBrowser_17
        self.textBrowser_34 = textBrowser_34
        
        # 누적 결과용
        self.textBrowser_32 = textBrowser_32
        self.textBrowser_30 = textBrowser_30
        self.textBrowser_28 = textBrowser_28
        self.textBrowser_37 = textBrowser_37
        
        # 불량 제품 리스트
        self.textEdit_products = [
            textEdit_product1, textEdit_product2, textEdit_product3, textEdit_product4,
            textEdit_product5, textEdit_product6, textEdit_product7, textEdit_product8
        ]
        self.textEdit_scores = [
            textEdit_score1, textEdit_score2, textEdit_score3, textEdit_score4,
            textEdit_score5, textEdit_score6, textEdit_score7, textEdit_score8
        ]
        
        self.stackedWidget_defect = stackedWidget_defect
        self.pushButton_after = pushButton_after
        self.pushButton_prev = pushButton_prev
        
        self.logger = logger
        
        # 불량 제품 데이터
        self.defect_list = []  # [(제품명, 점수), ...]
        self.current_page = 0
        self.items_per_page = 8
        
        # 시그널 연결
        if self.btn_judge_one:
            self.btn_judge_one.clicked.connect(self.run_simple_inference)
            print("✅ SimpleJudgeController: btn_judge_one 연결됨")
        
        if self.pushButton_after:
            self.pushButton_after.clicked.connect(self.next_page)
        
        if self.pushButton_prev:
            self.pushButton_prev.clicked.connect(self.prev_page)
    
    def run_simple_inference(self):
        """간단한 판정 실행 + CSV 저장 + 누적 표시"""
        print("\n" + "="*50)
        print("🔘 btn_judge_one 클릭됨! (SimpleJudgeController)")
        print("📊 간단 판정 시작")
        print("="*50)
        
        # 1. 모델 경로 가져오기 (우선순위: train_proj_ctrl → model_ctrl)
        model_path = None

        if self.train_proj_ctrl and self.train_proj_ctrl.current_model_path:
            model_path = self.train_proj_ctrl.current_model_path
            print(f"✅ 프로젝트에서 모델 경로: {model_path}")
        elif self.model_ctrl and self.model_ctrl.get_current_model_path():
            model_path = self.model_ctrl.get_current_model_path()
            print(f"✅ 모델 로드 버튼에서 경로: {model_path}")

        if not model_path or not os.path.exists(model_path):
            print("❌ 프로젝트에서 모델을 찾을 수 없습니다")
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "모델 없음",
                "먼저 프로젝트를 선택하고 모델을 로드하세요."
            )
            return
        
        # 2. 데이터 확인
        if not self.data_ctrl or self.data_ctrl.df is None:
            print("❌ 판정 데이터가 없습니다")
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "데이터 없음",
                "먼저 판정 데이터를 로드하세요."
            )
            return
        
        df = self.data_ctrl.df
        print(f"✅ 데이터: {len(df)}행")
        
        # 3. JSON 파일 확인
        json_path = model_path.rsplit(".", 1)[0] + ".json"
        if not os.path.exists(json_path):
            print(f"❌ JSON 파일 없음: {json_path}")
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "JSON 파일 없음",
                f"모델과 동일 이름의 JSON 파일이 필요합니다:\n{json_path}"
            )
            return
        
        print(f"✅ JSON: {json_path}")
        
        try:
            # 4. InferenceEngine 초기화
            print("🔧 InferenceEngine 초기화...")
            engine = InferenceEngine(model_path, json_path)
            
            wave_cols = self.data_ctrl.wave_cols
            mode = self.data_ctrl.mode
            name_col = self.data_ctrl.name_col
            
            print(f"📊 추론 실행 중... ({len(df)}개 데이터)")
            
            # 5. 추론 실행
            if engine.model_family == "svm":
                print("   SVM 모드")
                if engine.speed_col and engine.speed_col in df.columns:
                    speed = pd.to_numeric(df[engine.speed_col], errors="coerce").fillna(0).values
                else:
                    speed = np.zeros(len(df), dtype=np.float32)
                
                feats = []
                for i in range(len(df)):
                    wave = extract_waveform(df.iloc[i], wave_cols, mode, apply_preprocess=True)
                    if engine.feature_extractor is not None:
                        feat = engine.feature_extractor(wave, sampling_rate=engine.sampling_rate)
                    else:
                        feat = extract_all_features(wave, sampling_rate=engine.sampling_rate)
                    feats.append(feat)
                
                feat_df = pd.DataFrame(np.asarray(feats), columns=engine.feature_cols)
                X = feat_df.values
                predictions, probs = engine.predict(X)
                
            else:  # PyTorch
                print("   PyTorch 모드")
                if engine.feature_extractor is not None:
                    feats = []
                    for i in range(len(df)):
                        wave = extract_waveform(df.iloc[i], wave_cols, mode, apply_preprocess=True)
                        feat = engine.feature_extractor(wave, sampling_rate=engine.sampling_rate)
                        feats.append(feat)
                    
                    feat_df = pd.DataFrame(np.asarray(feats), columns=engine.feature_cols)
                    X = feat_df.values
                else:
                    waves = []
                    for i in range(len(df)):
                        wave = extract_waveform(df.iloc[i], wave_cols, mode, apply_preprocess=True)
                        waves.append(wave)
                    X = np.asarray(waves, dtype=np.float32)
                
                predictions, probs = engine.predict(X)
            
            # 6. 결과 계산
            predictions = np.asarray(predictions)
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            
            if predictions.ndim == 2 and predictions.shape[0] == 2:
                print(f"   ⚠️ predictions가 2차원 - 첫 번째 행만 사용")
                predictions = predictions[0]
            
            total_count = len(predictions)
            crack_count = int(np.sum(predictions == 1))
            normal_count = total_count - crack_count
            defect_rate = (crack_count / total_count * 100) if total_count > 0 else 0
            
            print(f"\n📊 판정 결과:")
            print(f"   - 전체: {total_count}개")
            print(f"   - 정상: {normal_count}개")
            print(f"   - 불량: {crack_count}개")
            print(f"   - 불량률: {defect_rate:.2f}%")
            
            # 7. CSV 저장 (원본 형식 유지)
            self.save_results_to_csv(df, predictions, probs, name_col)
            
            # 8. 현재 판정 결과 UI 업데이트
            if self.textBrowser_13:
                self.textBrowser_13.setText(str(total_count))
            if self.textBrowser_15:
                self.textBrowser_15.setText(str(normal_count))
            if self.textBrowser_17:
                self.textBrowser_17.setText(str(crack_count))
            if self.textBrowser_34:
                self.textBrowser_34.setText(f"{defect_rate:.2f}%")
            
            # 9. 현재 판정 도넛 그래프
            self.create_donut_chart(normal_count, crack_count)
            
            # 10. 불량 제품 리스트 추출
            self.extract_defect_list(df, predictions, probs, name_col)
            
            # 11. 누적 결과 로드 및 표시
            self.load_and_display_daily_results()
            
            # 12. 완료 메시지
            QtWidgets.QMessageBox.information(
                self.main_window,
                "판정 완료",
                f"판정이 완료되었습니다.\n\n"
                f"전체: {total_count}개\n"
                f"정상: {normal_count}개\n"
                f"불량: {crack_count}개\n"
                f"불량률: {defect_rate:.2f}%\n\n"
                f"결과가 Results 폴더에 저장되었습니다."
            )
            
            if self.logger:
                self.logger.log_info(f"판정 완료: 불량 {crack_count}/{total_count} ({defect_rate:.2f}%)")
            
        except Exception as e:
            print(f"❌ 판정 실패: {e}")
            import traceback
            traceback.print_exc()
            
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "판정 실패",
                f"판정 중 오류가 발생했습니다:\n{e}"
            )
            
            if self.logger:
                self.logger.log_error(f"판정 실패: {e}")
    
    def save_results_to_csv(self, df, predictions, probs, name_col):
        """
        원본 CSV에 판정 컬럼 추가 (판정시간, 판정, 확률, 판정근거)
        파일명: result_YYYYMMDD.csv
        """
        # Results 폴더 생성
        results_dir = os.path.join(os.getcwd(), "Results")
        os.makedirs(results_dir, exist_ok=True)
        
        # 파일명: result_YYYYMMDD.csv
        today = datetime.now().strftime("%Y%m%d")
        csv_path = os.path.join(results_dir, f"result_{today}.csv")
        
        # 원본 데이터프레임 복사
        result_df = df.copy()
        
        # 판정 결과 컬럼 생성
        judgment_time = []
        judgment_result = []
        judgment_prob = []
        judgment_reason = []
        
        for i in range(len(df)):
            # 판정시간
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            judgment_time.append(timestamp)
            
            # 판정 (균열/정상)
            is_defect = predictions[i] == 1
            judgment_result.append("균열" if is_defect else "정상")
            
            # 확률
            if probs is not None:
                if isinstance(probs, np.ndarray):
                    if probs.ndim == 2:
                        prob_value = float(probs[i, 1]) if probs.shape[1] > 1 else float(probs[i, 0])
                    else:
                        prob_value = float(probs[i])
                else:
                    prob_value = 1.0 if is_defect else 0.0
            else:
                prob_value = 1.0 if is_defect else 0.0
            
            judgment_prob.append(prob_value)
            
            # 판정근거
            threshold = 0.5
            if is_defect:
                reason = f"균열 가능성 높음 (확률: {prob_value*100:.1f}%, 기준: {threshold*100:.1f}%)"
            else:
                reason = f"정상 (확률: {prob_value*100:.1f}%, 기준: {threshold*100:.1f}%)"
            judgment_reason.append(reason)
        
        # 컬럼 추가
        result_df['판정시간'] = judgment_time
        result_df['판정'] = judgment_result
        result_df['확률'] = judgment_prob
        result_df['판정근거'] = judgment_reason
        
        # 기존 파일이 있으면 합치기
        if os.path.exists(csv_path):
            print(f"📁 기존 CSV 발견: {csv_path}")
            try:
                existing_df = pd.read_csv(csv_path, encoding="utf-8-sig")
                combined_df = pd.concat([existing_df, result_df], ignore_index=True)
                print(f"   - 기존 데이터: {len(existing_df)}행")
                print(f"   - 추가 데이터: {len(result_df)}행")
                print(f"   - 합계: {len(combined_df)}행")
            except Exception as e:
                print(f"⚠️ 기존 CSV 로드 실패: {e}")
                combined_df = result_df
        else:
            print(f"📝 새 CSV 생성: {csv_path}")
            combined_df = result_df
        
        # CSV 저장
        combined_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ CSV 저장 완료: {csv_path} ({len(combined_df)}행)")
        
        if self.logger:
            self.logger.log_file(f"판정 결과 저장: result_{today}.csv ({len(result_df)}행 추가)")
    
    def load_and_display_daily_results(self):
        """오늘 누적 결과 로드 및 표시"""
        results_dir = os.path.join(os.getcwd(), "Results")
        today = datetime.now().strftime("%Y%m%d")
        csv_path = os.path.join(results_dir, f"result_{today}.csv")
        
        if not os.path.exists(csv_path):
            print("📭 오늘 누적 데이터 없음")
            # 위젯 초기화
            if self.textBrowser_32:
                self.textBrowser_32.setText("0")
            if self.textBrowser_30:
                self.textBrowser_30.setText("0")
            if self.textBrowser_28:
                self.textBrowser_28.setText("0")
            if self.textBrowser_37:
                self.textBrowser_37.setText("0.00%")
            return
        
        try:
            # CSV 로드
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            
            # 판정 컬럼 확인
            if '판정' not in df.columns:
                print("❌ 판정 컬럼 없음")
                return
            
            print(f"📊 누적 데이터 로드: {len(df)}행")
            
            # 판정 결과 집계
            total_count = len(df)
            defect_count = len(df[df["판정"] == "균열"])
            normal_count = total_count - defect_count
            defect_rate = (defect_count / total_count * 100) if total_count > 0 else 0
            
            print(f"   - 전체: {total_count}개")
            print(f"   - 정상: {normal_count}개")
            print(f"   - 불량: {defect_count}개")
            print(f"   - 불량률: {defect_rate:.2f}%")
            
            # UI 업데이트
            if self.textBrowser_32:
                self.textBrowser_32.setText(str(total_count))
            if self.textBrowser_30:
                self.textBrowser_30.setText(str(normal_count))
            if self.textBrowser_28:
                self.textBrowser_28.setText(str(defect_count))
            if self.textBrowser_37:
                self.textBrowser_37.setText(f"{defect_rate:.2f}%")
            
            # 누적 도넛 차트
            self.create_daily_donut_chart(normal_count, defect_count)
            
            if self.logger:
                self.logger.log_data(f"누적 결과 로드: {total_count}개 (불량 {defect_count}개)")
            
        except Exception as e:
            print(f"❌ 누적 데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def extract_defect_list(self, df, predictions, probs, name_col):
        """불량 제품 리스트 추출"""
        self.defect_list = []
        
        for i in range(len(df)):
            if predictions[i] == 1:  # 불량
                row = df.iloc[i]
                
                # 제품명
                if name_col and name_col in df.columns:
                    product_name = str(row[name_col])
                else:
                    product_name = f"제품_{i+1}"
                
                # 점수
                if probs is not None:
                    if isinstance(probs, np.ndarray):
                        if probs.ndim == 2:
                            score = float(probs[i, 1] * 100) if probs.shape[1] > 1 else float(probs[i, 0] * 100)
                        else:
                            score = float(probs[i] * 100)
                    else:
                        score = 0.0
                else:
                    score = 100.0
                
                self.defect_list.append((product_name, score))
        
        print(f"🔴 불량 제품 리스트: {len(self.defect_list)}개")
        
        # 페이지 초기화 및 표시
        self.current_page = 0
        self.display_defect_page()
    
    def display_defect_page(self):
        """현재 페이지의 불량 제품 표시"""
        total_pages = (len(self.defect_list) + self.items_per_page - 1) // self.items_per_page
        
        if total_pages == 0:
            # 불량 제품 없음
            for i in range(8):
                if self.textEdit_products[i]:
                    self.textEdit_products[i].clear()
                if self.textEdit_scores[i]:
                    self.textEdit_scores[i].clear()
            
            if self.stackedWidget_defect:
                self.stackedWidget_defect.setCurrentIndex(0)
            
            print("✅ 불량 제품 없음")
            return
        
        # 현재 페이지 데이터
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.defect_list))
        
        page_items = self.defect_list[start_idx:end_idx]
        
        print(f"📄 페이지 {self.current_page + 1}/{total_pages} 표시 ({len(page_items)}개)")
        
        # 8개 슬롯 채우기
        for i in range(8):
            if i < len(page_items):
                product_name, score = page_items[i]
                
                if self.textEdit_products[i]:
                    self.textEdit_products[i].setPlainText(product_name)
                
                if self.textEdit_scores[i]:
                    self.textEdit_scores[i].setPlainText(f"{score:.2f}%")
            else:
                if self.textEdit_products[i]:
                    self.textEdit_products[i].clear()
                
                if self.textEdit_scores[i]:
                    self.textEdit_scores[i].clear()
        
        # stackedWidget 페이지 설정
        if self.stackedWidget_defect:
            page_idx = min(self.current_page, self.stackedWidget_defect.count() - 1)
            self.stackedWidget_defect.setCurrentIndex(page_idx)
        
        # 버튼 활성화/비활성화
        if self.pushButton_prev:
            self.pushButton_prev.setEnabled(self.current_page > 0)
        
        if self.pushButton_after:
            self.pushButton_after.setEnabled(self.current_page < total_pages - 1)
    
    def next_page(self):
        """다음 페이지"""
        total_pages = (len(self.defect_list) + self.items_per_page - 1) // self.items_per_page
        
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.display_defect_page()
            print(f"➡️ 다음 페이지: {self.current_page + 1}")
    
    def prev_page(self):
        """이전 페이지"""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_defect_page()
            print(f"⬅️ 이전 페이지: {self.current_page + 1}")
    
    def create_donut_chart(self, normal_count: int, crack_count: int):
        """현재 판정 도넛 그래프 (label_chart_placeholder)"""
        if not self.label_chart_placeholder:
            return
        
        total = normal_count + crack_count
        if total == 0:
            return
        
        crack_percent = (crack_count / total) * 100
        
        fig = Figure(figsize=(5, 5), facecolor='white')
        ax = fig.add_subplot(111)
        
        sizes = [normal_count, crack_count]
        colors = ['#4472C4', '#E74C3C']
        
        wedges = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.6)
        )[0]
        
        centre_circle = plt.Circle((0,0), 0.6, fc='white')
        ax.add_artist(centre_circle)
        
        ax.text(0, 0, f'{crack_percent:.1f}%', 
                ha='center', va='center', 
                fontsize=30, weight='bold', color='#E74C3C')
        
        ax.axis('equal')
        
        import io
        from PyQt5.QtGui import QPixmap
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        
        self.label_chart_placeholder.setPixmap(pixmap)
        self.label_chart_placeholder.setScaledContents(True)
        
        buf.close()
        plt.close(fig)
        
        print(f"📊 현재 판정 도넛 그래프 생성 완료")
    
    def create_daily_donut_chart(self, normal_count: int, crack_count: int):
        """누적 결과 도넛 그래프 (label_judge_chart_placeholder_7)"""
        if not self.label_judge_chart_placeholder_7:
            print("⚠️ label_judge_chart_placeholder_7 위젯 없음 - main.py에서 연결 필요")
            return
        
        total = normal_count + crack_count
        if total == 0:
            print("⚠️ 누적 데이터 없음 (total=0)")
            return
        
        try:
            crack_percent = (crack_count / total) * 100
            
            print(f"📊 누적 도넛 차트 생성 중... (정상: {normal_count}, 불량: {crack_count})")
            
            fig = Figure(figsize=(5, 5), facecolor='white')
            ax = fig.add_subplot(111)
            
            sizes = [normal_count, crack_count]
            colors = ['#4472C4', '#E74C3C']
            
            wedges = ax.pie(
                sizes,
                colors=colors,
                startangle=90,
                wedgeprops=dict(width=0.6)
            )[0]
            
            centre_circle = plt.Circle((0,0), 0.6, fc='white')
            ax.add_artist(centre_circle)
            
            ax.text(0, 0, f'{crack_percent:.1f}%', 
                    ha='center', va='center', 
                    fontsize=30, weight='bold', color='#E74C3C')
            
            ax.axis('equal')
            
            import io
            from PyQt5.QtGui import QPixmap
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            
            self.label_judge_chart_placeholder_7.setPixmap(pixmap)
            self.label_judge_chart_placeholder_7.setScaledContents(True)
            
            buf.close()
            plt.close(fig)
            
            print(f"✅ 누적 결과 도넛 그래프 생성 완료")
        
        except Exception as e:
            print(f"❌ 누적 도넛 차트 생성 실패: {e}")
            import traceback
            traceback.print_exc()