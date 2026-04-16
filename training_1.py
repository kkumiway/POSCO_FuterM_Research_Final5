# -*- coding: utf-8 -*-
"""training_1.py - SVM 모델 학습 컨트롤러 (Robust CSV 로더 포함)"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List
from PyQt5 import QtCore, QtWidgets
import pickle

from scipy.signal import hilbert, find_peaks, stft
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis, entropy
from scipy.integrate import trapezoid

# sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost 미설치. XGB 모델 제외됨. (pip install xgboost)")

# matplotlib
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ============================================
# MultiModelEnsemble - LR/SVM/RF/XGB/MLP × N seeds
# ============================================
class MultiModelEnsemble:
    """
    LR / SVM / RF / XGB / MLP × N seeds 앙상블
    judge_inference_integrated.py와 동일한 predict_proba 인터페이스 유지
    """
    MODEL_NAMES = ["LR", "SVM", "RF", "XGB", "MLP"]

    def __init__(self, ensemble_by_model: dict, model_names: list, scaler=None):
        """
        Args:
            ensemble_by_model: {'LR': [pipe1, pipe2, ...], 'SVM': [...], ...}
                               각 pipe는 scaler가 이미 적용된 모델 (fit된 상태)
            model_names: 실제 사용된 모델명 리스트
            scaler: 외부 스케일러 (fit_transform 완료). pipe 내부에 scaler 없을 때 사용
        """
        self.ensemble_by_model = ensemble_by_model
        self.model_names = model_names
        self.scaler = scaler

    def _scale(self, X):
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def predict_proba_per_model(self, X):
        """모델별 확률값 반환 (dict)"""
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
        """전체 앙상블 확률값 반환 (N×2 array) - 추론 엔진 호환"""
        model_probs = self.predict_proba_per_model(X)
        if not model_probs:
            n = len(X)
            return np.stack([np.ones(n) * 0.5, np.ones(n) * 0.5], axis=1)
        ensemble_probs = np.mean(list(model_probs.values()), axis=0)
        return np.stack([1 - ensemble_probs, ensemble_probs], axis=1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# 하위 호환성 유지 - 기존 .pt 파일(SVM 단일 앙상블) 로드용
class SeedEnsemble:
    """기존 SVM 단일 앙상블 래퍼 (하위 호환성 유지)"""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probs = np.mean(
            [m.predict_proba(X)[:, 1] for m in self.models], axis=0
        )
        return np.stack([1 - probs, probs], axis=1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

print("✅ MultiModelEnsemble / SeedEnsemble 정의 완료")


# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        import matplotlib.font_manager as fm
        import platform
        import warnings
        
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
        
        system = platform.system()
        
        if system == 'Windows':
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
                            fm.fontManager.addfont(font_path)
                            plt.rcParams['font.family'] = font_name
                            print(f"한글 폰트 설정: {font_name}")
                            font_found = True
                            break
                    if font_found:
                        break
                if font_found:
                    break
            
            if not font_found:
                plt.rcParams['font.family'] = 'Malgun Gothic'
        
        elif system == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
        else:
            plt.rcParams['font.family'] = 'NanumGothic'
        
        plt.rcParams['axes.unicode_minus'] = False
        # fm._rebuild()  # deprecated
        
    except Exception as e:
        print(f"한글 폰트 설정 실패: {e}")

setup_korean_font()


# ============================================
# Robust CSV 로더 (judge_dataload.py에서 가져옴)
# ============================================
def _make_unique_columns(cols):
    """중복 컬럼명 처리"""
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
    """PL-Link 형식 CSV 로드"""
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
    rows = [line.rstrip("\n").rstrip("\r").split("\t") for line in lines]
    raw = pd.DataFrame(rows)
    raw = raw.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)
    raw = raw.dropna(axis=1, how="all")
    
    header_keywords = ["인덱스", "SW", "측정", "시간", "속도", "Name", "Unit", "Result", "Signal", "신호", "균열", "판정"]
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


def read_csv_robust(file_path: str) -> pd.DataFrame:
    """
    여러 인코딩을 시도하는 Robust CSV 로더
    judge_dataload.py의 read_measurement_csv_smart()와 동일
    """
    # encodings_probe = ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "cp949", "euc-kr", "utf-8"]
    encodings_probe = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None
    
    # 1단계: 탭 구분 형식 시도
    for enc in encodings_probe:
        try:
            with open(file_path, "r", encoding=enc) as f:
                head = f.read(4096)
            if "\t" in head:
                df = load_pllink_csv_safe(file_path, encoding=enc)
                df.columns = _make_unique_columns([str(c).strip() for c in df.columns])
                print(f"CSV 로드 성공: {enc} (탭 구분)")
                return df
        except Exception as e:
            last_err = e
    
    # 2단계: 일반 CSV 형식 시도
    for enc in encodings_probe:
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=sep)
                if df is not None and df.shape[1] >= 1:
                    df.columns = _make_unique_columns([str(c).strip() for c in df.columns])
                    df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False).dropna(how="all")
                    print(f"CSV 로드 성공: {enc} (구분자: {repr(sep)})")
                    return df
            except Exception as e:
                last_err = e
    
    raise ValueError(f"CSV 로드 실패: {file_path}\n원인: {last_err}")


# ============================================
# 신호 전처리
# ============================================
def preprocess_signal(signal, gain=1.0, clip_max=None, clip_min=None):
    """신호 전처리 - 2.5× 증폭/클리핑 제거, float32 변환만 수행"""
    return np.asarray(signal, dtype=np.float32)


# ============================================
# 피처 추출
# ============================================
def extract_all_features(signal, sampling_rate=1000000):
    """전체 피처 추출"""
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


# 피처 도메인 분류
FEATURE_GROUPS = {
    "time": [
        "peak_amp", "backwall_amp", "TOF", "std", 
        "skew", "kurtosis", "crest_factor", "hilbert_auc"
    ],
    "frequency": [
        "dominant_freq", "band_energy_ratio", "spectral_entropy",
        "spectral_centroid", "spectral_spread", "spectral_flatness",
        "peak_count", "spectral_kurtosis"
    ],
    "time_frequency": [
        "dom_freq_t", "energy_t", "entropy_t", "peak_trend_t",
        "freq_shift_t", "spec_kurtosis_t", "high_band_activity"
    ]
}


# 🆕 피처 한글 매핑 (UI 표시용 - 한글과 영문이 자연스럽게 매칭)
FEATURE_KOREAN_NAMES_FULL = {
    # Time domain (8개)
    "peak_amp": "Peak 진폭 (Peak Amplitude)",
    "backwall_amp": "후면벽 진폭 (Backwall Amplitude)",
    "TOF": "Peak 발생시간 (Peak Occurrence Time)",
    "std": "신호 표준편차 (Signal Standard Deviation)",
    "skew": "포락선 왜도 (Envelope Skewness)",
    "kurtosis": "포락선 첨도 (Envelope Kurtosis)",
    "crest_factor": "Peak 감쇠율 (Peak Attenuation Rate)",
    "hilbert_auc": "신호 포락선 면적 (Signal Envelope AUC)",
    
    # Frequency domain (8개)
    "dominant_freq": "메인 주파수 (Main Frequency)",
    "band_energy_ratio": "특정 주파수 비율 (Specific Frequency Ratio)",
    "spectral_entropy": "스펙트럼 엔트로피 (Spectral Entropy)",
    "spectral_centroid": "주파수 중심값 (Frequency Centroid)",
    "spectral_spread": "주파수 확산도 (Frequency Spread)",
    "spectral_flatness": "스펙트럼 평탄도 (Spectral Flatness)",
    "peak_count": "주파수 Peak 개수 (Frequency Peak Count)",
    "spectral_kurtosis": "스펙트럼 첨도 (Spectral Kurtosis)",
    
    # Time-Frequency domain (7개)
    "dom_freq_t": "시간별 주파수 (Frequency over Time)",
    "energy_t": "에너지 지속시간 (Energy Duration)",
    "entropy_t": "평균 에너지 (Average Energy)",
    "peak_trend_t": "시간별 Peak 밀도 (Peak Density over Time)",
    "freq_shift_t": "주파수 이동량 (Frequency Shift)",
    "spec_kurtosis_t": "시간별 첨도 (Kurtosis over Time)",
    "high_band_activity": "고주파 에너지 비율 (High Frequency Energy Ratio)"
}


# 🆕 피처 한글 매핑 (피처 중요도용 - 한글만)
FEATURE_KOREAN_NAMES = {
    # Time domain
    "peak_amp": "Peak 진폭",
    "backwall_amp": "후면벽 진폭",
    "TOF": "Peak 발생시간",
    "std": "신호 표준편차",
    "skew": "포락선 왜도",
    "kurtosis": "포락선 첨도",
    "crest_factor": "Peak 감쇠율",
    "hilbert_auc": "신호 포락선 면적",
    
    # Frequency domain
    "dominant_freq": "메인 주파수",
    "band_energy_ratio": "특정 주파수 비율",
    "spectral_entropy": "스펙트럼 엔트로피",
    "spectral_centroid": "주파수 중심값",
    "spectral_spread": "주파수 확산도",
    "spectral_flatness": "스펙트럼 평탄도",
    "peak_count": "주파수 Peak 개수",
    "spectral_kurtosis": "스펙트럼 첨도",
    
    # Time-Frequency domain
    "dom_freq_t": "시간별 주파수",
    "energy_t": "에너지 지속시간",
    "entropy_t": "평균 에너지",
    "peak_trend_t": "시간별 Peak 밀도",
    "freq_shift_t": "주파수 이동량",
    "spec_kurtosis_t": "시간별 첨도",
    "high_band_activity": "고주파 에너지 비율"
}




ALL_FEATURES = (
    FEATURE_GROUPS["time"] + 
    FEATURE_GROUPS["frequency"] + 
    FEATURE_GROUPS["time_frequency"]
)


# ============================================
# 파형 컬럼 자동 감지 (judge_dataload.py에서 가져옴)
# ============================================
def pick_first_existing(df, candidates):
    """후보 컬럼 중 첫 번째 존재하는 컬럼 반환"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_wave_start_idx(df: pd.DataFrame) -> int:
    """파형 데이터 시작 인덱스 찾기"""
    cols = list(df.columns)
    
    # 1. "wave" 컬럼 찾기
    for i, c in enumerate(cols):
        if str(c).strip().lower() == "wave":
            return i
    
    # 2. "signal" 또는 "신호" 다음 컬럼
    for i, c in enumerate(cols):
        c_str = str(c).strip().lower()
        if ("signal" in c_str) or ("신호" in c_str):
            return min(i + 1, len(cols) - 1)
    
    # 3. "Unnamed" 시작
    for i, c in enumerate(cols):
        c_str = str(c).strip().lower()
        if c_str.startswith("unnamed"):
            return i
    
    # 4. 숫자로 시작하는 컬럼
    for i, c in enumerate(cols):
        s = str(c).strip()
        try:
            float(s)
            return i
        except Exception:
            pass
    
    return 0


def detect_columns_and_wavecols(df: pd.DataFrame):
    """라벨, 이름, 파형 컬럼 자동 감지"""
    label_col = pick_first_existing(df, ["균열유무", "균열여부", "판정", "label", "Label", "crack", "Crack"])
    name_col = pick_first_existing(df, ["Name", "name", "제품명", "이름", "부분명"])
    
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


def extract_waveform(df_row, wave_cols, mode):
    """파형 데이터 추출"""
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
# 학습 워커 (멀티스레딩)
# ============================================
class TrainingWorker(QtCore.QThread):
    """백그라운드 학습 스레드"""
    progress = QtCore.pyqtSignal(int, str)  # (진행률, 메시지)
    finished = QtCore.pyqtSignal(dict)  # 학습 결과
    error = QtCore.pyqtSignal(str)  # 에러 메시지
    
    def __init__(self, train_data, val_data, config):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.is_stopped = False
    
    def stop(self):
        """학습 중지"""
        self.is_stopped = True
    
    def run(self):
        """학습 실행 - LR / SVM / RF / XGB / MLP × N seeds 앙상블"""
        try:
            X_train, y_train = self.train_data
            X_val, y_val = self.val_data

            kernel       = self.config['kernel']        # SVM에만 적용
            scaler_type  = self.config['scaler']
            n_seeds      = self.config['n_seeds']
            threshold    = self.config['threshold']
            seeds        = [42 + i * 100 for i in range(n_seeds)]

            # ── 스케일러 선택 및 fit ──────────────────────────────
            if scaler_type == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif scaler_type == 'StandardScaler':
                scaler = StandardScaler()
            elif scaler_type == 'RobustScaler':
                scaler = RobustScaler()
            else:
                scaler = MinMaxScaler()

            self.progress.emit(5, "데이터 스케일링 중...")
            X_train_s = scaler.fit_transform(X_train)
            X_val_s   = scaler.transform(X_val)

            # ── 사용할 모델 목록 결정 ────────────────────────────
            MODEL_NAMES = ["LR", "SVM", "RF", "XGB", "MLP"]
            active_models = list(MODEL_NAMES)
            if not XGBOOST_AVAILABLE and "XGB" in active_models:
                active_models.remove("XGB")
                print("⚠️ XGBoost 없음 → XGB 제외")

            ensemble_by_model = {name: [] for name in active_models}

            # ── N seeds × 5 모델 학습 ────────────────────────────
            total_steps = n_seeds * len(active_models)
            step = 0

            for seed in seeds:
                if self.is_stopped:
                    return

                model_set = {
                    "LR": LogisticRegression(
                        penalty="l1", solver="liblinear",
                        class_weight="balanced", max_iter=1000,
                        random_state=seed
                    ),
                    "SVM": SVC(
                        C=self.config.get('C', 1.0),
                        kernel=kernel,          # cb_kernel 값 그대로 사용
                        gamma='scale',
                        class_weight="balanced",
                        probability=True,
                        random_state=seed
                    ),
                    "RF": RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=seed
                    ),
                    "MLP": MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=300,
                        early_stopping=True,
                        random_state=seed
                    ),
                }
                if XGBOOST_AVAILABLE:
                    model_set["XGB"] = XGBClassifier(
                        eval_metric="logloss",
                        use_label_encoder=False,
                        random_state=seed
                    )

                for name in active_models:
                    if self.is_stopped:
                        return
                    step += 1
                    pct = 10 + int(step / total_steps * 65)
                    self.progress.emit(pct, f"[{name}] Seed {seed} 학습 중... ({step}/{total_steps})")
                    model_set[name].fit(X_train_s, y_train)
                    ensemble_by_model[name].append(model_set[name])

            # ── 앙상블 예측 ──────────────────────────────────────
            self.progress.emit(78, "앙상블 예측 중...")

            # 모델별 확률값 (스케일된 X 직접 사용)
            model_probs_train = {}
            model_probs_val   = {}
            for name in active_models:
                p_tr = np.mean([m.predict_proba(X_train_s)[:, 1]
                                for m in ensemble_by_model[name]], axis=0)
                p_va = np.mean([m.predict_proba(X_val_s)[:, 1]
                                for m in ensemble_by_model[name]], axis=0)
                model_probs_train[name] = p_tr
                model_probs_val[name]   = p_va

            train_probs_mean = np.mean(list(model_probs_train.values()), axis=0)
            val_probs_mean   = np.mean(list(model_probs_val.values()),   axis=0)

            train_preds = (train_probs_mean >= threshold).astype(int)
            val_preds   = (val_probs_mean   >= threshold).astype(int)

            # ── 평가 ─────────────────────────────────────────────
            self.progress.emit(85, "성능 평가 중...")

            train_metrics = {
                'accuracy':  accuracy_score(y_train, train_preds),
                'precision': precision_score(y_train, train_preds, zero_division=0),
                'recall':    recall_score(y_train, train_preds, zero_division=0),
                'f1':        f1_score(y_train, train_preds, zero_division=0),
            }
            val_metrics = {
                'accuracy':  accuracy_score(y_val, val_preds),
                'precision': precision_score(y_val, val_preds, zero_division=0),
                'recall':    recall_score(y_val, val_preds, zero_division=0),
                'f1':        f1_score(y_val, val_preds, zero_division=0),
            }

            # 모델별 개별 성능 (val)
            model_val_metrics = {}
            for name in active_models:
                mp = (model_probs_val[name] >= threshold).astype(int)
                model_val_metrics[name] = {
                    'accuracy': accuracy_score(y_val, mp),
                    'f1':       f1_score(y_val, mp, zero_division=0),
                }

            cm_train = confusion_matrix(y_train, train_preds)
            cm_val   = confusion_matrix(y_val,   val_preds)

            fpr, tpr, _ = roc_curve(y_val, val_probs_mean)
            roc_auc = auc(fpr, tpr)

            # ── CV 점수 (RF 대표 - 가장 안정적) ─────────────────
            self.progress.emit(88, "교차 검증 중 (RF)...")
            cv_scores = []
            if "RF" in active_models:
                rf_ref = RandomForestClassifier(n_estimators=100,
                                                class_weight="balanced",
                                                random_state=42)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(rf_ref, X_train_s, y_train,
                                         cv=cv, scoring='f1')
                cv_scores = scores.tolist()
            else:
                cv_scores = [val_metrics['f1']]

            # ── 피처 중요도 (RF 내장 importance 사용) ────────────
            self.progress.emit(91, "피처 중요도 계산 중...")

            feature_names_korean = [
                FEATURE_KOREAN_NAMES.get(f, f)
                for f in self.config['selected_features']
            ]

            if "RF" in active_models and ensemble_by_model["RF"]:
                rf_model = ensemble_by_model["RF"][0]
                importances     = rf_model.feature_importances_
                # RF는 std 계산 가능
                imp_std = np.std(
                    [tree.feature_importances_
                     for tree in rf_model.estimators_], axis=0
                )
                importance_raw = np.array(
                    [t.feature_importances_ for t in rf_model.estimators_]
                ).T   # (n_features, n_trees)
            else:
                # fallback: SVM permutation importance
                ref_model = ensemble_by_model[active_models[0]][0]
                perm = permutation_importance(
                    ref_model, X_val_s, y_val, n_repeats=10, random_state=42
                )
                importances    = perm.importances_mean
                imp_std        = perm.importances_std
                importance_raw = perm.importances

            feature_importance = pd.DataFrame({
                'feature':     feature_names_korean,
                'feature_eng': self.config['selected_features'],
                'importance':  importances,
                'std':         imp_std,
            }).sort_values('importance', ascending=False)

            # ── 결과 패키징 ───────────────────────────────────────
            self.progress.emit(96, "결과 정리 중...")

            results = {
                # 앙상블 객체 생성에 필요한 정보
                'ensemble_by_model': ensemble_by_model,
                'active_models':     active_models,
                'scaler':            scaler,
                # 기존 키 유지 (display_results 호환)
                'models':            [m for name in active_models
                                       for m in ensemble_by_model[name]],
                'train_metrics':     train_metrics,
                'val_metrics':       val_metrics,
                'model_val_metrics': model_val_metrics,
                'cv_scores':         cv_scores,
                'cm_train':          cm_train,
                'cm_val':            cm_val,
                'roc_curve':         (fpr, tpr, roc_auc),
                'feature_importance': feature_importance,
                'importance_raw':     importance_raw,
                'feature_names':      self.config['selected_features'],
                'val_probs':          val_probs_mean,
                'model_probs_val':    model_probs_val,
                'val_preds':          val_preds,
                'val_labels':         y_val,
                'threshold':          threshold,
            }

            self.progress.emit(100, "학습 완료!")
            self.finished.emit(results)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ============================================
# 학습 컨트롤러
# ============================================
class TrainingController(QtCore.QObject):
    """학습 탭 컨트롤러"""
    
    def __init__(
        self,
        main_window,
        # 🆕 groupBox_8 프로젝트 정보 표시용 위젯
        textBrowser_project_2=None,
        textBrowser_project_date_2=None,
        textBrowser_process_2=None,
        textBrowser_product_2=None,
        # 🆕 groupBox_9 모델 정보 표시용 위젯
        textBrowser_model_2=None,
        textBrowser_traindata_2=None,
        textBrowser_detail_2=None,
        textBrowser_validdata_2=None,
        # 파일 선택
        btn_model_load=None,
        le_model_path=None,
        btn_train_csv=None,
        le_train_csv=None,
        btn_val_csv=None,
        le_val_csv=None,
        btn_out_dir=None,
        le_out_dir=None,
        # 파라미터
        cb_kernel=None,
        cb_scaler=None,
        spin_seed=None,
        dspin_threshold=None,
        # 피처 선택
        list_time=None,
        list_freq=None,
        list_time_freq=None,
        # 학습 제어
        btn_start=None,
        btn_stop=None,
        btn_save=None,  # 저장 버튼
        progress_bar=None,
        # 결과 표시
        label_cm=None,
        label_roc=None,
        table_result=None,
        text_shap=None,
        text_importance=None,
        text_report=None,
        logger=None,
        # TrainingProjectController 추가
        train_proj_ctrl=None,
        # 디렉토리 경로
        models_dir=None,
        projects_dir=None
    ):
        super().__init__(main_window)
        
        self.main_window = main_window
        
        # TrainingProjectController 참조
        self.train_proj_ctrl = train_proj_ctrl
        
        # 디렉토리 경로
        self.models_dir = models_dir
        self.projects_dir = projects_dir
        
        # 🆕 groupBox_8 프로젝트 정보 표시용 위젯
        self.textBrowser_project_2 = textBrowser_project_2
        self.textBrowser_project_date_2 = textBrowser_project_date_2
        self.textBrowser_process_2 = textBrowser_process_2
        self.textBrowser_product_2 = textBrowser_product_2
        
        # 🆕 groupBox_9 모델 정보 표시용 위젯
        self.textBrowser_model_2 = textBrowser_model_2
        self.textBrowser_traindata_2 = textBrowser_traindata_2
        self.textBrowser_detail_2 = textBrowser_detail_2
        self.textBrowser_validdata_2 = textBrowser_validdata_2
        
        # UI 요소
        self.btn_model_load = btn_model_load or getattr(main_window, 'btn_train_traincsv_load_ml_2', None)
        self.le_model_path = le_model_path or getattr(main_window, 'le_train_model_name_ml', None)
        self.btn_train_csv = btn_train_csv or getattr(main_window, 'btn_train_traincsv_browse_ml', None)
        self.le_train_csv = le_train_csv or getattr(main_window, 'le_train_traincsv_ml', None)
        self.btn_val_csv = btn_val_csv or getattr(main_window, 'btn_train_validcsv_browse_ml', None)
        self.le_val_csv = le_val_csv or getattr(main_window, 'le_train_validcsv_ml', None)
        self.btn_out_dir = btn_out_dir or getattr(main_window, 'btn_train_outdir_browse_ml', None)
        self.le_out_dir = le_out_dir or getattr(main_window, 'le_train_outdir_ml', None)
        
        self.cb_kernel = cb_kernel or getattr(main_window, 'cb_train_cv_ml', None)
        self.cb_scaler = cb_scaler or getattr(main_window, 'cb_train_scaler_ml', None)
        self.spin_seed = spin_seed or getattr(main_window, 'spin_train_seed_ml', None)
        self.dspin_threshold = dspin_threshold or getattr(main_window, 'dspin_train_thr_ml', None)
        
        self.list_time = list_time or getattr(main_window, 'listWidget', None)
        self.list_freq = list_freq or getattr(main_window, 'listWidget_2', None)
        self.list_time_freq = list_time_freq or getattr(main_window, 'listWidget_3', None)
        
        self.btn_start = btn_start or getattr(main_window, 'btn_train_start_ml', None)
        self.btn_stop = btn_stop or getattr(main_window, 'btn_train_stop_ml', None)
        self.btn_save = btn_save or getattr(main_window, 'pushButton_5', None)  # 저장 버튼
        self.progress_bar = progress_bar or getattr(main_window, 'progress_train_ml_2', None)  # 수정: _2 추가
        
        self.label_cm = label_cm or getattr(main_window, 'label_train_plot_loss_ml_2', None)
        self.label_roc = label_roc or getattr(main_window, 'label_train_plot_val_ml_2', None)
        self.table_result = table_result or getattr(main_window, 'tableWidget_result', None)
        self.text_shap = text_shap or getattr(main_window, 'text_shap_ml_1', None)
        self.text_importance = text_importance or getattr(main_window, 'text_featureimportance_1', None)
        self.text_report = text_report or getattr(main_window, 'text_train_report_1', None)
        
        # SHAP과 Feature Importance를 위한 위젯 (이미지 표시용)
        self.widget_shap = getattr(main_window, 'widget_shap_ml_1', None)
        self.widget_importance = getattr(main_window, 'widget_Featureimportance_ml_1', None)
        
        # 데이터
        self.model_json = None
        self.train_df = None
        self.val_df = None
        self.training_worker = None
        self.logger = logger
        
        # 시그널 연결
        self._connect_signals()
        
        # 초기화
        self._init_ui()
    
    def _connect_signals(self):
        """시그널 연결"""
        if self.btn_model_load:
            self.btn_model_load.clicked.connect(self.load_model)
        if self.btn_train_csv:
            self.btn_train_csv.clicked.connect(self.load_train_csv)
        if self.btn_val_csv:
            self.btn_val_csv.clicked.connect(self.load_val_csv)
        if self.btn_out_dir:
            self.btn_out_dir.clicked.connect(self.select_output_dir)
        if self.btn_start:
            self.btn_start.clicked.connect(self.start_training)
        if self.btn_stop:
            self.btn_stop.clicked.connect(self.stop_training)
        if self.btn_save:
            self.btn_save.clicked.connect(self.save_model_and_project)  # 저장 버튼 연결
    
    def _init_ui(self):
        """UI 초기화"""
        # 커널 선택
        if self.cb_kernel:
            self.cb_kernel.clear()
            self.cb_kernel.addItems(['rbf', 'linear', 'poly', 'sigmoid'])
            self.cb_kernel.setCurrentText('rbf')
        
        # 스케일러 선택
        if self.cb_scaler:
            self.cb_scaler.clear()
            self.cb_scaler.addItems(['MinMaxScaler', 'StandardScaler', 'RobustScaler'])
            self.cb_scaler.setCurrentText('MinMaxScaler')
        
        # 시드 개수
        if self.spin_seed:
            self.spin_seed.setValue(5)
            self.spin_seed.setMinimum(1)
            self.spin_seed.setMaximum(20)
        
        # Threshold
        if self.dspin_threshold:
            self.dspin_threshold.setValue(0.5)
            self.dspin_threshold.setMinimum(0.0)
            self.dspin_threshold.setMaximum(1.0)
            self.dspin_threshold.setSingleStep(0.01)
        
        # 진행률
        if self.progress_bar:
            self.progress_bar.setValue(0)
        
        # 버튼 상태
        if self.btn_stop:
            self.btn_stop.setEnabled(False)
        
        # 🆕 초기화 시 기본 feature list 로드
        self._load_default_feature_lists()
        print("TrainingController 초기화 완료 - 기본 feature list 로드됨")
        
        # 디버깅: progress_bar 연결 확인
        if self.progress_bar:
            print(f"progress_bar 연결됨: {self.progress_bar.objectName()}")
        else:
            print("경고: progress_bar가 연결되지 않음!")
    
    
    def set_project_info(self, model_name: str, train_csv: str, valid_csv: str, algorithm: str):
        """
        프로젝트 정보를 TrainingController에 설정
        TrainingProjectController에서 프로젝트 로드 시 호출됨
        
        Args:
            model_name: 모델 파일명
            train_csv: 학습 데이터 파일명
            valid_csv: 검증 데이터 파일명
            algorithm: 알고리즘 이름
        """
        print(f"\nTrainingController에 프로젝트 정보 설정:")
        
        # 모델 이름
        if self.le_model_path and model_name:
            self.le_model_path.setText(model_name)
            print(f"   - 모델: {model_name}")
        
        # 학습 데이터
        if self.le_train_csv and train_csv:
            self.le_train_csv.setText(train_csv)
            print(f"   - 학습 데이터: {train_csv}")
        
        # 검증 데이터
        if self.le_val_csv and valid_csv:
            self.le_val_csv.setText(valid_csv)
            print(f"   - 검증 데이터: {valid_csv}")
        
        # 알고리즘 (출력 디렉토리에 표시)
        if self.le_out_dir and algorithm:
            self.le_out_dir.setText(algorithm)
            print(f"   - 알고리즘: {algorithm}")
        
        print(f"프로젝트 정보 설정 완료!")
    
    def update_project_display(self, project_info: dict):
        """
        groupBox_8에 프로젝트 정보 표시
        
        Args:
            project_info: 프로젝트 정보 딕셔너리
        """
        print(f"\n학습 탭 프로젝트 정보 표시 (TrainingController):")
        print(f"   받은 project_info: {project_info}")
        print(f"   위젯 상태:")
        print(f"   - textBrowser_project_2: {self.textBrowser_project_2 is not None}")
        print(f"   - textBrowser_project_date_2: {self.textBrowser_project_date_2 is not None}")
        print(f"   - textBrowser_process_2: {self.textBrowser_process_2 is not None}")
        print(f"   - textBrowser_product_2: {self.textBrowser_product_2 is not None}")
        
        # 프로젝트명
        if self.textBrowser_project_2:
            project_name = project_info.get("project_filename", "")
            self._set_text_centered(self.textBrowser_project_2, project_name)
            print(f"   프로젝트: {project_name}")
        else:
            print(f"   textBrowser_project_2가 None")
        
        # 생성날짜
        if self.textBrowser_project_date_2:
            created_date = project_info.get("created_date", "")
            self._set_text_centered(self.textBrowser_project_date_2, created_date)
            print(f"   생성날짜: {created_date}")
        else:
            print(f"   textBrowser_project_date_2가 None")
        
        # 대상공정
        if self.textBrowser_process_2:
            target_process = project_info.get("target_process", "")
            self._set_text_centered(self.textBrowser_process_2, target_process)
            print(f"   대상공정: {target_process}")
        else:
            print(f"   textBrowser_process_2가 None")
        
        # 대상제품
        if self.textBrowser_product_2:
            target_product = project_info.get("target_product", "")
            self._set_text_centered(self.textBrowser_product_2, target_product)
            print(f"   대상제품: {target_product}")
        else:
            print(f"   textBrowser_product_2가 None")
        
        print(f"\n   groupBox_9 모델 정보 업데이트:")
        
        # 분석파일 (모델 파일명)
        if self.textBrowser_model_2:
            model_filename = project_info.get("model_filename", "")
            self._set_text_centered(self.textBrowser_model_2, model_filename)
            print(f"   분석파일: {model_filename}")
        else:
            print(f"   textBrowser_model_2가 None")
        
        # 학습데이터
        if self.textBrowser_traindata_2:
            train_data = project_info.get("train_data_filename", "")
            self._set_text_centered(self.textBrowser_traindata_2, train_data)
            print(f"   학습데이터: {train_data}")
        else:
            print(f"   textBrowser_traindata_2가 None")
        
        # 배포날짜 (detail_2)
        if self.textBrowser_detail_2:
            model_date = project_info.get("model_date", "")
            self._set_text_centered(self.textBrowser_detail_2, model_date)
            print(f"   배포날짜: {model_date}")
        else:
            print(f"   textBrowser_detail_2가 None")
        
        # 검증데이터
        if self.textBrowser_validdata_2:
            valid_data = project_info.get("valid_data_filename", "")
            self._set_text_centered(self.textBrowser_validdata_2, valid_data)
            print(f"   검증데이터: {valid_data}")
        else:
            print(f"   textBrowser_validdata_2가 None")
        
        print(f"학습 탭 프로젝트 정보 표시 완료!")
    
    def _set_text_centered(self, text_browser, text: str):
        """
        QTextBrowser에 텍스트 설정 (왼쪽 정렬)
        수직은 중앙, 수평은 왼쪽 정렬
        """
        if text_browser:
            # HTML로 수직 중앙, 수평 왼쪽 정렬
            html = f'''
            <div style="
                display: flex;
                align-items: center;
                height: 100%;
                padding-left: 10px;
            ">
                {text}
            </div>
            '''
            text_browser.setHtml(html)
    def load_model(self):
        """모델 불러오기"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window,
            "모델 선택",
            "",
            "PyTorch Model (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # .pt 파일 경로 표시
        if self.le_model_path:
            self.le_model_path.setText(file_path)
        
        # 🆕 groupBox_9 업데이트 - 분석파일
        if self.textBrowser_model_2:
            model_filename = os.path.basename(file_path)
            self._set_text_centered(self.textBrowser_model_2, model_filename)
            print(f"   groupBox_9 업데이트: 분석파일 = {model_filename}")
        
        # JSON 파일 로드
        json_path = file_path.rsplit('.', 1)[0] + '.json'
        
        if not os.path.exists(json_path):
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "JSON 파일 없음",
                f"모델과 동일 이름의 JSON 파일이 필요합니다:\n{json_path}"
            )
            return
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.model_json = json.load(f)
            
            # 피처 목록 로드
            self._load_feature_lists()
            
            # 🆕 groupBox_9 업데이트 - 배포날짜 (모델 파일 수정 날짜)
            if self.textBrowser_detail_2:
                try:
                    file_mtime = os.path.getmtime(file_path)
                    model_date = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
                    self._set_text_centered(self.textBrowser_detail_2, model_date)
                    print(f"   groupBox_9 업데이트: 배포날짜 = {model_date}")
                except Exception as e:
                    print(f"   배포날짜 추출 실패: {e}")
            
            print(f"모델 로드: {os.path.basename(file_path)}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "JSON 로드 실패",
                f"JSON 파일을 읽을 수 없습니다:\n{e}"
            )
    
    def load_from_project(self):
        """
        프로젝트에서 모델 정보를 로드하고 피처 리스트 표시
        TrainingProjectController에서 모델이 로드되면 자동 호출됨
        """
        if not self.train_proj_ctrl:
            print("TrainingProjectController가 연결되지 않음")
            return
        
        # 프로젝트에서 모델 경로 가져오기
        if not self.train_proj_ctrl.current_model_path:
            print("프로젝트에 로드된 모델이 없음")
            return
        
        model_path = self.train_proj_ctrl.current_model_path
        json_path = model_path.rsplit('.', 1)[0] + '.json'
        
        if not os.path.exists(json_path):
            print(f"JSON 파일 없음: {json_path}")
            return
        
        try:
            # JSON 로드
            with open(json_path, 'r', encoding='utf-8') as f:
                self.model_json = json.load(f)
            
            # 모델 경로 표시
            if self.le_model_path:
                self.le_model_path.setText(model_path)
            
            # 피처 리스트 로드
            self._load_feature_lists()
            
            print(f"프로젝트에서 모델 정보 로드 완료: {os.path.basename(model_path)}")
            print(f"   피처 리스트 표시 완료")
            
            if self.logger:
                self.logger.log_success(f"프로젝트에서 모델 정보 로드 완료")
                self.logger.log_info(f"   모델: {os.path.basename(model_path)}")
                self.logger.log_info(f"   피처 리스트 표시 완료")
        
        except Exception as e:
            print(f"프로젝트에서 모델 정보 로드 실패: {e}")
            if self.logger:
                self.logger.log_error(f"프로젝트에서 모델 정보 로드 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_default_feature_lists(self):
        """
        초기화 시 기본 feature list 로드
        모든 feature를 체크된 상태로 표시
        🆕 한글 (영어) 형태로 표시, Time → Time Segment
        """
        print("\n기본 신호 특성 리스트 로드 중...")
        
        # Time domain features
        if self.list_time:
            self.list_time.clear()
            for feature in FEATURE_GROUPS['time']:
                # 🆕 한글 (영어) 형태
                display_name = FEATURE_KOREAN_NAMES_FULL.get(feature, feature)
                item = QtWidgets.QListWidgetItem(display_name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)  # 기본적으로 모두 선택
                item.setData(QtCore.Qt.UserRole, feature)  # 영어 변수명 저장
                self.list_time.addItem(item)
            print(f"   시간 영역 (Time domain): {len(FEATURE_GROUPS['time'])}개")
        
        # Frequency domain features
        if self.list_freq:
            self.list_freq.clear()
            for feature in FEATURE_GROUPS['frequency']:
                # 🆕 한글 (영어) 형태
                display_name = FEATURE_KOREAN_NAMES_FULL.get(feature, feature)
                item = QtWidgets.QListWidgetItem(display_name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)  # 기본적으로 모두 선택
                item.setData(QtCore.Qt.UserRole, feature)  # 영어 변수명 저장
                self.list_freq.addItem(item)
            print(f"   주파수 영역 (Frequency domain): {len(FEATURE_GROUPS['frequency'])}개")
        
        # Time-Frequency domain features
        if self.list_time_freq:
            self.list_time_freq.clear()
            for feature in FEATURE_GROUPS['time_frequency']:
                # 🆕 한글 (영어) 형태
                display_name = FEATURE_KOREAN_NAMES_FULL.get(feature, feature)
                item = QtWidgets.QListWidgetItem(display_name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)  # 기본적으로 모두 선택
                item.setData(QtCore.Qt.UserRole, feature)  # 영어 변수명 저장
                self.list_time_freq.addItem(item)
            print(f"   시간-주파수 영역 (Time-Frequency domain): {len(FEATURE_GROUPS['time_frequency'])}개")
        
        print("✅ 기본 신호 특성 리스트 로드 완료 (한글 표시)")
    
    def _load_feature_lists(self):
        """
        JSON에서 피처 목록 로드하여 리스트에 표시
        🆕 한글 (영어) 형태로 표시, Time → Time Segment
        """
        if not self.model_json:
            return
        
        # JSON에서 피처 컬럼 가져오기
        feature_cols = self.model_json.get('feature_extraction', {}).get('feature_cols', ALL_FEATURES)
        
        # 도메인별 분류
        time_features = [f for f in FEATURE_GROUPS['time'] if f in feature_cols]
        freq_features = [f for f in FEATURE_GROUPS['frequency'] if f in feature_cols]
        time_freq_features = [f for f in FEATURE_GROUPS['time_frequency'] if f in feature_cols]
        
        # 리스트에 추가 (체크 가능) - 기본적으로 모두 선택
        if self.list_time:
            self.list_time.clear()
            for feature in FEATURE_GROUPS['time']:
                # 🆕 한글 (영어) 형태
                display_name = FEATURE_KOREAN_NAMES_FULL.get(feature, feature)
                item = QtWidgets.QListWidgetItem(display_name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)  # 기본적으로 모두 선택
                item.setData(QtCore.Qt.UserRole, feature)  # 영어 변수명 저장
                self.list_time.addItem(item)
        
        if self.list_freq:
            self.list_freq.clear()
            for feature in FEATURE_GROUPS['frequency']:
                # 🆕 한글 (영어) 형태
                display_name = FEATURE_KOREAN_NAMES_FULL.get(feature, feature)
                item = QtWidgets.QListWidgetItem(display_name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)  # 기본적으로 모두 선택
                item.setData(QtCore.Qt.UserRole, feature)  # 영어 변수명 저장
                self.list_freq.addItem(item)
        
        if self.list_time_freq:
            self.list_time_freq.clear()
            for feature in FEATURE_GROUPS['time_frequency']:
                # 🆕 한글 (영어) 형태
                display_name = FEATURE_KOREAN_NAMES_FULL.get(feature, feature)
                item = QtWidgets.QListWidgetItem(display_name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)  # 기본적으로 모두 선택
                item.setData(QtCore.Qt.UserRole, feature)  # 영어 변수명 저장
                self.list_time_freq.addItem(item)
        
        print(f"✅ 피처 목록 로드: {len(feature_cols)}개 (한글 표시)")
    
    def _get_selected_features(self):
        """
        선택된 피처 목록 반환
        🆕 UserRole에 저장된 영어 변수명 반환
        """
        selected = []
        
        if self.list_time:
            for i in range(self.list_time.count()):
                item = self.list_time.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    # UserRole에 저장된 영어 변수명 사용
                    feature_name = item.data(QtCore.Qt.UserRole)
                    if feature_name:
                        selected.append(feature_name)
                    else:
                        # fallback
                        selected.append(item.text())
        
        if self.list_freq:
            for i in range(self.list_freq.count()):
                item = self.list_freq.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    feature_name = item.data(QtCore.Qt.UserRole)
                    if feature_name:
                        selected.append(feature_name)
                    else:
                        selected.append(item.text())
        
        if self.list_time_freq:
            for i in range(self.list_time_freq.count()):
                item = self.list_time_freq.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    feature_name = item.data(QtCore.Qt.UserRole)
                    if feature_name:
                        selected.append(feature_name)
                    else:
                        selected.append(item.text())
        
        return selected
    
    def load_train_csv(self):
        """학습 CSV 불러오기"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window,
            "학습 데이터 선택",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Robust CSV 로더 사용
            self.train_df = read_csv_robust(file_path)
            
            if self.le_train_csv:
                self.le_train_csv.setText(file_path)
            
            print(f"학습 데이터 로드: {len(self.train_df)}행")
            
            # 로그 추가
            if self.logger:
                self.logger.log_data(f"학습 데이터 로드: {os.path.basename(file_path)} ({len(self.train_df)}행)")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "CSV 로드 실패",
                f"CSV 파일을 읽을 수 없습니다:\n{e}"
            )
            import traceback
            traceback.print_exc()
    
    def load_val_csv(self):
        """검증 CSV 불러오기"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window,
            "검증 데이터 선택",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Robust CSV 로더 사용
            self.val_df = read_csv_robust(file_path)
            
            if self.le_val_csv:
                self.le_val_csv.setText(file_path)
            
            print(f"검증 데이터 로드: {len(self.val_df)}행")
            
            # 로그 추가
            if self.logger:
                self.logger.log_data(f"검증 데이터 로드: {os.path.basename(file_path)} ({len(self.val_df)}행)")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "CSV 로드 실패",
                f"CSV 파일을 읽을 수 없습니다:\n{e}"
            )
            import traceback
            traceback.print_exc()
    
    def select_output_dir(self):
        """저장 경로 선택"""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.main_window,
            "저장 경로 선택",
            ""
        )
        
        if not dir_path:
            return
        
        if self.le_out_dir:
            self.le_out_dir.setText(dir_path)
        
        print(f"저장 경로: {dir_path}")
    
    def start_training(self):
        """학습 시작"""
        # TrainingProjectController에서 모델 정보 가져오기
        if self.train_proj_ctrl and not self.model_json:
            # 프로젝트에서 모델이 로드되었는지 확인
            if self.train_proj_ctrl.current_model_path:
                model_path = self.train_proj_ctrl.current_model_path
                json_path = model_path.rsplit('.', 1)[0] + '.json'
                
                if os.path.exists(json_path):
                    print("프로젝트에서 모델 JSON 로드")
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            self.model_json = json.load(f)
                        self._load_feature_lists()
                        print(f"모델 JSON 로드 완료: {os.path.basename(json_path)}")
                    except Exception as e:
                        print(f"JSON 로드 실패: {e}")
        
        # 검증
        if not self.model_json:
            QtWidgets.QMessageBox.warning(self.main_window, "모델 없음", "먼저 모델을 로드하세요.")
            return
        
        # TrainingProjectController에서 데이터 가져오기
        if self.train_proj_ctrl:
            if self.train_df is None and self.train_proj_ctrl.current_train_data is not None:
                self.train_df = self.train_proj_ctrl.current_train_data
                print(f"프로젝트에서 학습 데이터 가져옴: {len(self.train_df)}행")
            
            if self.val_df is None and self.train_proj_ctrl.current_valid_data is not None:
                self.val_df = self.train_proj_ctrl.current_valid_data
                print(f"프로젝트에서 검증 데이터 가져옴: {len(self.val_df)}행")
        
        if self.train_df is None:
            QtWidgets.QMessageBox.warning(self.main_window, "학습 데이터 없음", "학습 CSV를 로드하세요.")
            return
        
        if self.val_df is None:
            QtWidgets.QMessageBox.warning(self.main_window, "검증 데이터 없음", "검증 CSV를 로드하세요.")
            return
        
        # 선택된 피처
        selected_features = self._get_selected_features()
        
        if not selected_features:
            QtWidgets.QMessageBox.warning(self.main_window, "피처 없음", "최소 1개 이상의 피처를 선택하세요.")
            return
        
        try:
            # 🆕 진행률 0% 시작
            if self.progress_bar:
                self.progress_bar.setValue(0)
            
            # 데이터 준비 (0-10%: 학습 데이터, 10-20%: 검증 데이터)
            print("학습 데이터 피처 추출 중... (0-10%)")
            X_train, y_train = self._prepare_data(self.train_df, selected_features, progress_start=0, progress_end=10)
            
            print("검증 데이터 피처 추출 중... (10-20%)")
            X_val, y_val = self._prepare_data(self.val_df, selected_features, progress_start=10, progress_end=20)
            
            # 설정
            config = {
                'kernel': self.cb_kernel.currentText() if self.cb_kernel else 'rbf',
                'scaler': self.cb_scaler.currentText() if self.cb_scaler else 'MinMaxScaler',
                'n_seeds': self.spin_seed.value() if self.spin_seed else 5,
                'threshold': self.dspin_threshold.value() if self.dspin_threshold else 0.5,
                'C': self.model_json.get('training_config', {}).get('C', 1.0),
                'polynomial': False,  # 필요시 추가
                'selected_features': selected_features
            }
            
            # 워커 생성
            self.training_worker = TrainingWorker((X_train, y_train), (X_val, y_val), config)
            self.training_worker.progress.connect(self._on_progress)
            self.training_worker.finished.connect(self._on_finished)
            self.training_worker.error.connect(self._on_error)
            
            # UI 상태
            if self.btn_start:
                self.btn_start.setEnabled(False)
            if self.btn_stop:
                self.btn_stop.setEnabled(True)
            
            # 학습 시작
            self.training_worker.start()
            print("학습 시작")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "학습 시작 실패",
                f"학습을 시작할 수 없습니다:\n{e}"
            )
            import traceback
            traceback.print_exc()
    
    def _prepare_data(self, df, selected_features, progress_start=0, progress_end=100):
        """
        데이터 준비 (피처 추출) - 파형 컬럼 자동 감지
        
        Args:
            df: 데이터프레임
            selected_features: 선택된 피처 목록
            progress_start: 시작 진행률 (%)
            progress_end: 종료 진행률 (%)
        """
        # 파형 컬럼 자동 감지
        name_col, label_col, wave_cols, mode = detect_columns_and_wavecols(df)
        
        print(f"자동 감지 결과:")
        print(f"  - 라벨 컬럼: {label_col}")
        print(f"  - 이름 컬럼: {name_col}")
        print(f"  - 파형 모드: {mode}")
        print(f"  - 파형 컬럼 수: {len(wave_cols)}")
        
        # 파형 컬럼 검증
        if not wave_cols:
            raise ValueError("파형 데이터를 찾을 수 없습니다. CSV 파일의 구조를 확인하세요.")
        
        # 라벨 컬럼 검증
        if not label_col or label_col not in df.columns:
            raise ValueError(f"라벨 컬럼을 찾을 수 없습니다. CSV에 다음 중 하나가 필요합니다: 균열유무, 균열여부, 판정, label")
        
        # 라벨 추출
        y_raw = df[label_col].astype(str).str.strip()
        
        # 라벨 매핑 (여러 형식 지원)
        label_map = {
            '정상': 0, '0': 0, 'normal': 0, 'Normal': 0, 'NORMAL': 0,
            '균열': 1, '1': 1, 'crack': 1, 'Crack': 1, 'CRACK': 1, 'defect': 1
        }
        
        y = y_raw.map(label_map).values
        
        # 라벨 검증
        if np.any(np.isnan(y)):
            unique_values = y_raw.unique()
            raise ValueError(f"라벨 값을 인식할 수 없습니다. 발견된 값: {unique_values}\n"
                           f"지원되는 값: 정상/균열, 0/1, normal/crack")
        
        y = y.astype(int)
        
        # 피처 추출
        features_list = []
        
        print(f"피처 추출 중... (총 {len(df)}개 샘플, {progress_start}%-{progress_end}%)")
        
        for i in range(len(df)):
            # 🆕 진행률 업데이트 (progress_start부터 progress_end까지)
            if self.progress_bar and i % 10 == 0:
                progress = progress_start + int((i / len(df)) * (progress_end - progress_start))
                self.progress_bar.setValue(progress)
            
            if i % 100 == 0 and i > 0:
                print(f"  진행: {i}/{len(df)} ({i/len(df)*100:.1f}%)")
            
            # 파형 추출 (자동 감지된 방식 사용)
            wave = extract_waveform(df.iloc[i], wave_cols, mode)
            
            # 전처리
            wave = preprocess_signal(wave)
            
            # 전체 피처 추출
            all_feats = extract_all_features(wave)
            
            # 선택된 피처만 추출
            feat_dict = dict(zip(ALL_FEATURES, all_feats))
            selected_feats = [feat_dict[f] for f in selected_features]
            
            features_list.append(selected_feats)
        
        X = np.array(features_list, dtype=np.float32)
        
        # 🆕 진행률을 progress_end로 설정
        if self.progress_bar:
            self.progress_bar.setValue(progress_end)
        
        print(f"데이터 준비 완료: X={X.shape}, y={y.shape}")
        print(f"  - 정상: {np.sum(y==0)}개, 균열: {np.sum(y==1)}개")
        
        return X, y
    
    def stop_training(self):
        """학습 중지"""
        if self.training_worker:
            self.training_worker.stop()
            print("학습 중지 요청")
    
    def save_model_and_project(self):
        """
        학습한 모델과 프로젝트 저장
        - .pt 파일: Models 폴더에 저장 (덮어쓰기)
        - .json 파일: projects 폴더에 저장 (덮어쓰기)
        """
        try:
            # 1. 학습 결과가 있는지 확인
            if not hasattr(self, 'trained_model') or self.trained_model is None:
                QtWidgets.QMessageBox.warning(
                    self.main_window,
                    "저장 불가",
                    "저장할 학습 결과가 없습니다.\n먼저 모델을 학습해주세요."
                )
                return
            
            # 2. Models 폴더 확인
            if not self.models_dir:
                QtWidgets.QMessageBox.warning(
                    self.main_window,
                    "저장 불가",
                    "Models 폴더 경로가 설정되지 않았습니다."
                )
                return
            
            # 3. 프로젝트 정보 확인 (TrainingProjectController에서)
            if not self.train_proj_ctrl:
                QtWidgets.QMessageBox.warning(
                    self.main_window,
                    "저장 불가",
                    "프로젝트가 선택되지 않았습니다."
                )
                return
            
            project_info = self.train_proj_ctrl.get_current_project_info()
            if not project_info:
                QtWidgets.QMessageBox.warning(
                    self.main_window,
                    "저장 불가",
                    "프로젝트 정보를 가져올 수 없습니다."
                )
                return
            
            # 4. 모델 이름 확인
            original_model_name = project_info.get('model_filename', '')
            if not original_model_name:
                QtWidgets.QMessageBox.warning(
                    self.main_window,
                    "저장 불가",
                    "원본 모델 이름을 찾을 수 없습니다."
                )
                return
            
            # 5. Models 폴더에 저장 경로 설정 (덮어쓰기)
            os.makedirs(self.models_dir, exist_ok=True)
            model_path = os.path.join(self.models_dir, original_model_name)
            
            print(f"\n모델 저장 중...")
            print(f"  모델 이름: {original_model_name}")
            print(f"  저장 경로: {model_path}")
            
            # 6. 선택된 피처 추출 (bundle에 포함하기 위해)
            selected_features = []
            if self.list_time:
                for i in range(self.list_time.count()):
                    item = self.list_time.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        selected_features.append(item.text())
            if self.list_freq:
                for i in range(self.list_freq.count()):
                    item = self.list_freq.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        selected_features.append(item.text())
            if self.list_time_freq:
                for i in range(self.list_time_freq.count()):
                    item = self.list_time_freq.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        selected_features.append(item.text())
            
            # 7. 모델 저장 (.pt) - 덮어쓰기
            bundle = {
                'ensemble':      self.trained_model,          # MultiModelEnsemble
                'scaler':        self.trained_scaler,
                'threshold':     self.trained_threshold,
                'model_family':  'multi_model_seed_ensemble',
                'active_models': self.last_results.get('active_models', []),
                'feature_cols':  selected_features,
                'sampling_rate': 1000000,
                'wave_mode':     'cols',
                'amp_cols':      [],
                'speed_col':     '속도'
            }
            torch.save(bundle, model_path)
            print(f"모델 저장 완료: {original_model_name}")
            
            # 8. projects 폴더에 JSON 저장 경로 설정
            if not self.projects_dir:
                self.projects_dir = os.path.join(os.path.dirname(self.models_dir), "projects")
            os.makedirs(self.projects_dir, exist_ok=True)
            
            # JSON 파일명 (.pt -> .json)
            json_name = os.path.splitext(original_model_name)[0] + '.json'
            json_path = os.path.join(self.projects_dir, json_name)
            
            # 9. JSON 메타데이터 생성
            metadata = {
                'model_name': original_model_name,
                'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'project_name': project_info.get('project_name', ''),
                'target_process': project_info.get('target_process', ''),
                'target_product': project_info.get('target_product', ''),
                'train_data': os.path.basename(self.le_train_csv.text()) if self.le_train_csv and self.le_train_csv.text() else None,
                'val_data': os.path.basename(self.le_val_csv.text()) if self.le_val_csv and self.le_val_csv.text() else None,
                'algorithm': self.cb_kernel.currentText() if self.cb_kernel else 'rbf',
                'scaler': self.cb_scaler.currentText() if self.cb_scaler else 'StandardScaler',
                'seed': self.spin_seed.value() if self.spin_seed else 42,
                'threshold': self.dspin_threshold.value() if self.dspin_threshold else 0.5,
                'features': selected_features,
                'performance': {
                    'val_accuracy': float(self.last_results['val_metrics']['accuracy']) if hasattr(self, 'last_results') else None,
                    'val_f1': float(self.last_results['val_metrics']['f1']) if hasattr(self, 'last_results') else None,
                    'val_precision': float(self.last_results['val_metrics']['precision']) if hasattr(self, 'last_results') else None,
                    'val_recall': float(self.last_results['val_metrics']['recall']) if hasattr(self, 'last_results') else None,
                } if hasattr(self, 'last_results') else None
            }
            
            # 9. JSON 저장 (덮어쓰기)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"메타데이터 저장 완료: {json_name}")
            print(f"  저장 위치: {json_path}")
            
            # 10. 성공 메시지
            QtWidgets.QMessageBox.information(
                self.main_window,
                "저장 완료",
                f"모델과 프로젝트가 성공적으로 저장되었습니다!\n\n"
                f"모델: {original_model_name}\n"
                f"위치: Models 폴더\n\n"
                f"프로젝트: {json_name}\n"
                f"위치: projects 폴더"
            )
            
        except Exception as e:
            print(f"모델 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "저장 실패",
                f"모델 저장 중 오류가 발생했습니다:\n{str(e)}"
            )
    
    def _on_progress(self, value, message):
        """
        진행률 업데이트
        TrainingWorker의 0-100%를 전체 프로세스의 20-100%로 매핑
        """
        # 🆕 0-100%를 20-100%로 매핑
        mapped_value = 20 + int(value * 0.8)
        
        if self.progress_bar:
            self.progress_bar.setValue(mapped_value)
        print(f"[{mapped_value}%] {message}")
    
    def _on_finished(self, results):
        """학습 완료"""
        print("학습 완료!")
        
        # 🆕 진행률 100% 설정
        if self.progress_bar:
            self.progress_bar.setValue(100)
        
        # UI 상태
        if self.btn_start:
            self.btn_start.setEnabled(True)
        if self.btn_stop:
            self.btn_stop.setEnabled(False)
        
        # 로그 추가
        if self.logger:
            val_acc = results['val_metrics']['accuracy']
            val_f1 = results['val_metrics']['f1']
            self.logger.log_success(f"학습 완료! 검증 정확도: {val_acc*100:.2f}%, F1: {val_f1*100:.2f}%")
            self.logger.separator()
        
        # 결과 표시
        self._display_results(results)
        
        # 학습 결과 저장 (save_model_and_project에서 사용)
        ensemble_by_model = results.get('ensemble_by_model', {})
        active_models     = results.get('active_models', [])
        self.trained_model    = MultiModelEnsemble(ensemble_by_model, active_models,
                                                   scaler=results.get('scaler'))
        self.trained_scaler   = results.get('scaler')
        self.trained_threshold = results.get('threshold')
        self.last_results     = results

        print(f"✅ MultiModelEnsemble 생성 완료 (모델: {active_models}, "
              f"seeds/model: {len(list(ensemble_by_model.values())[0]) if ensemble_by_model else 0})")
        
        # 모델 저장
        self._save_model(results)
        
        QtWidgets.QMessageBox.information(
            self.main_window,
            "학습 완료",
            f"학습이 완료되었습니다!\n\n"
            f"검증 정확도: {results['val_metrics']['accuracy']*100:.2f}%\n"
            f"F1-Score: {results['val_metrics']['f1']*100:.2f}%"
        )
    
    def _on_error(self, error_msg):
        """에러 처리"""
        print(f"학습 실패: {error_msg}")
        
        # 로그 추가
        if self.logger:
            self.logger.log_error(f"학습 실패: {error_msg}")
        
        # UI 상태
        if self.btn_start:
            self.btn_start.setEnabled(True)
        if self.btn_stop:
            self.btn_stop.setEnabled(False)
        
        QtWidgets.QMessageBox.critical(
            self.main_window,
            "학습 실패",
            f"학습 중 오류가 발생했습니다:\n{error_msg}"
        )
    
    def _display_results(self, results):
        """결과 표시"""
        # 1. Confusion Matrix
        self._display_confusion_matrix(results['cm_val'])
        
        # 2. ROC Curve
        self._display_roc_curve(results['roc_curve'])
        
        # 3. 균열 분류 테이블
        self._display_crack_table(results)
        
        # 4. 피처 중요도
        self._display_feature_importance(results['feature_importance'])
        
        # 5. 리포트
        self._display_report(results)
    
    def _display_confusion_matrix(self, cm):
        """Confusion Matrix 표시"""
        if not self.label_cm:
            return
        
        # 기존 위젯 제거
        layout = self.label_cm.layout()
        if layout:
            while layout.count():
                
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            layout = QtWidgets.QVBoxLayout(self.label_cm)
            layout.setContentsMargins(0, 0, 0, 0)
            self.label_cm.setLayout(layout)
        
        # 그래프 생성 (UI 위젯 크기에 맞춤)
        fig = Figure(figsize=(3.5, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['정상', '균열'],
               yticklabels=['정상', '균열'],
               title='Confusion Matrix',
               ylabel='실제',
               xlabel='예측')
        
        # 숫자 표시
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        # 레이아웃 조정
        fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.12)
        
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored
        )
        layout.addWidget(canvas)
    
    def _display_roc_curve(self, roc_data):
        """ROC Curve 표시"""
        if not self.label_roc:
            return
        
        fpr, tpr, roc_auc = roc_data
        
        # 기존 위젯 제거
        layout = self.label_roc.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            layout = QtWidgets.QVBoxLayout(self.label_roc)
            layout.setContentsMargins(0, 0, 0, 0)
            self.label_roc.setLayout(layout)
        
        # 그래프 생성 (UI 위젯 크기에 맞춤)
        fig = Figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=8)
        ax.set_ylabel('True Positive Rate', fontsize=8)
        ax.set_title('ROC Curve', fontsize=9)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # 레이아웃 조정
        fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.12)
        
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored
        )
        layout.addWidget(canvas)
    
    def _display_crack_table(self, results):
        """균열 분류 테이블 표시"""
        if not self.table_result:
            return
        
        # 균열로 분류된 샘플
        val_preds = results['val_preds']
        val_labels = results['val_labels']
        
        crack_indices = np.where(val_preds == 1)[0]
        
        if len(crack_indices) == 0:
            self.table_result.setRowCount(1)
            self.table_result.setColumnCount(1)
            self.table_result.setHorizontalHeaderLabels(['메시지'])
            self.table_result.setItem(0, 0, QtWidgets.QTableWidgetItem('균열로 분류된 데이터가 없습니다.'))
            return
        
        # 테이블 설정
        self.table_result.setRowCount(len(crack_indices))
        self.table_result.setColumnCount(3)
        self.table_result.setHorizontalHeaderLabels(['제품 이름', '판정 결과', '실제 값'])
        
        # 데이터 채우기
        for row_idx, idx in enumerate(crack_indices):
            # 제품 이름 (검증 데이터에서 가져오기)
            if self.val_df is not None and '부분명' in self.val_df.columns:
                name = str(self.val_df.iloc[idx]['부분명'])
            else:
                name = f'Sample_{idx}'
            
            pred = '균열'
            actual = '균열' if val_labels[idx] == 1 else '정상'
            
            self.table_result.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(name))
            self.table_result.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(pred))
            self.table_result.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(actual))
        
        self.table_result.resizeColumnsToContents()
    
    def _display_feature_importance(self, importance_df):
        """피처 중요도 표시 (UI에 이미지 표시)"""
        # Widget이 있으면 이미지로 표시
        if self.widget_importance:
            try:
                # 기존 위젯 제거
                layout = self.widget_importance.layout()
                if layout:
                    while layout.count():
                        item = layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                else:
                    layout = QtWidgets.QVBoxLayout(self.widget_importance)
                    layout.setContentsMargins(0, 0, 0, 0)  # 여백 제거
                    self.widget_importance.setLayout(layout)
                
                # 그래프 생성 (UI 위젯 크기에 맞춤)
                fig = Figure(figsize=(5.5, 3), dpi=120)
                ax = fig.add_subplot(111)
                
                # 상위 12개만 표시
                top_n = min(10, len(importance_df))
                data = importance_df.head(top_n).iloc[::-1]
                
                # Horizontal bar chart
                y_pos = np.arange(len(data))
                ax.barh(y_pos, data['importance'], 
                       xerr=data['std'], 
                       color='steelblue', 
                       alpha=0.8,
                       edgecolor='black')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(data['feature'], fontsize=7)  # 🆕 한글만 표시
                ax.set_xlabel('중요도 (Importance)', fontsize=8, fontweight='bold')
                ax.set_title('피처 중요도 (Feature Importance)', fontsize=9, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # 레이아웃 조정
                fig.subplots_adjust(left=0.28, right=0.95, top=0.93, bottom=0.10)
                
                canvas = FigureCanvasQTAgg(fig)
                canvas.setSizePolicy(
                    QtWidgets.QSizePolicy.Ignored,  # UI 크기 따름
                    QtWidgets.QSizePolicy.Ignored
                )
                layout.addWidget(canvas)
                
                if self.text_importance:
                    self.text_importance.setPlainText("✅ 피처 중요도 표시 완료 (한글)")
                
            except Exception as e:
                if self.text_importance:
                    self.text_importance.setPlainText(f"이미지 생성 실패: {str(e)}")
        
        # Widget이 없으면 텍스트로 표시
        elif self.text_importance:
            lines = ["피처 중요도 (Permutation Importance)\n"]
            lines.append("=" * 50 + "\n")
            lines.append("widget_Featureimportance_ml_1 위젯이 없어 텍스트로 표시됩니다.\n\n")
            
            for idx, row in importance_df.head(10).iterrows():
                feature = row['feature']
                importance = row['importance']
                std = row['std']
                lines.append(f"{idx+1:2d}. {feature:20s} {importance:.4f} (±{std:.4f})\n")
            
            self.text_importance.setPlainText(''.join(lines))
    
    def _display_report(self, results):
        """분석 리포트 표시"""
        if not self.text_report:
            return
        
        val_metrics = results['val_metrics']
        train_metrics = results['train_metrics']
        cv_scores = results['cv_scores']
        cm = results['cm_val']  # 🆕 Confusion Matrix 미리 가져오기
        
        # 🆕 Confusion Matrix 값 추출
        TN = cm[0, 0]  # True Negative (정상→정상)
        FP = cm[0, 1]  # False Positive (정상→균열)
        FN = cm[1, 0]  # False Negative (균열→정상)
        TP = cm[1, 1]  # True Positive (균열→균열)
        
        # 리포트 생성
        report = []
        report.append("=" * 20)
        report.append("학습 결과 리포트")
        report.append("=" * 20)
        report.append("")
        
        # 🆕 검증 성능 - 간결한 분수 표시
        report.append("검증 성능:")
        report.append(f"  Accuracy:  {val_metrics['accuracy']*100:6.2f}% = {TP+TN}/{TP+TN+FP+FN}  (TP+TN={TP}+{TN}, Total={TP+TN+FP+FN})")
        report.append(f"  Precision: {val_metrics['precision']*100:6.2f}% = {TP}/{TP+FP}  (TP={TP}, TP+FP={TP+FP})")
        report.append(f"  Recall:    {val_metrics['recall']*100:6.2f}% = {TP}/{TP+FN}  (TP={TP}, TP+FN={TP+FN})")
        report.append(f"  F1-Score:  {val_metrics['f1']*100:6.2f}% = 2*Precision*Recall / (Precision+Recall)")
        report.append("")
        
        report.append("학습 성능:")
        report.append(f"  Accuracy:  {train_metrics['accuracy']*100:6.2f}%")
        report.append(f"  F1-Score:  {train_metrics['f1']*100:6.2f}%")
        report.append("")
        
        report.append("교차 검증 (RF 대표):")
        report.append(f"  평균 F1:   {np.mean(cv_scores)*100:6.2f}%")
        report.append(f"  표준편차:  {np.std(cv_scores)*100:6.2f}%")
        report.append("")
        
        # 모델별 개별 성능
        model_val_metrics = results.get('model_val_metrics', {})
        if model_val_metrics:
            report.append("모델별 검증 성능:")
            for name, m in model_val_metrics.items():
                report.append(f"  {name:4s}  Acc: {m['accuracy']*100:5.2f}%  F1: {m['f1']*100:5.2f}%")
            report.append("")
        
        # Confusion Matrix
        report.append("Confusion Matrix:")
        report.append(f"  TN (정상->정상): {TN:4d}")
        report.append(f"  FP (정상->균열): {FP:4d}")
        report.append(f"  FN (균열->정상): {FN:4d}")
        report.append(f"  TP (균열->균열): {TP:4d}")
        report.append("")
        
        # 🆕 분석 - 이모지 제거
        report.append("분석:")
        if val_metrics['accuracy'] > 0.95:
            report.append("매우 우수한 성능")
        elif val_metrics['accuracy'] > 0.90:
            report.append("우수한 성능")
        else:
            report.append("성능 개선 필요")
        
        if abs(train_metrics['accuracy'] - val_metrics['accuracy']) > 0.05:
            report.append("과적합 가능성 (Train/Val 성능 차이 큼)")
        else:
            report.append("과적합 없음")
        
        if FN > 0:
            report.append(f"균열 놓침(FN): {FN}개 - Threshold 조정 고려")
        
        if FP > 0:
            report.append(f"오검출(FP): {FP}개")
        
        report.append("")
        report.append("=" * 60)
        
        self.text_report.setPlainText('\n'.join(report))
        
        # SHAP (Scatter plot 스타일로 시각화)
        if self.widget_shap:
            try:
                # 기존 위젯 제거
                layout = self.widget_shap.layout()
                if layout:
                    while layout.count():
                        item = layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                else:
                    layout = QtWidgets.QVBoxLayout(self.widget_shap)
                    layout.setContentsMargins(0, 0, 0, 0)  # 여백 제거
                    self.widget_shap.setLayout(layout)
                
                importance_df = results['feature_importance']
                importance_raw = results['importance_raw']  # (n_features, n_repeats)
                feature_names = results['feature_names']
                
                # 그래프 생성 (UI 위젯 크기에 맞춤)
                fig = Figure(figsize=(5, 2.5), dpi=100)
                ax = fig.add_subplot(111)
                
                # 상위 12개만 표시
                top_n = min(12, len(importance_df))
                top_features = importance_df.head(top_n)['feature'].tolist()
                
                # feature_names에서 인덱스 찾기
                feature_indices = [feature_names.index(f) for f in top_features if f in feature_names]
                
                # 데이터 준비 (역순으로)
                y_positions = []
                x_values = []
                colors = []
                
                for i, idx in enumerate(reversed(feature_indices)):
                    # 각 feature의 모든 importance 값 (n_repeats개)
                    values = importance_raw[idx, :]
                    n_points = len(values)
                    
                    # Y position (약간 jitter 추가)
                    y_pos = np.ones(n_points) * i + np.random.randn(n_points) * 0.1
                    y_positions.extend(y_pos)
                    x_values.extend(values)
                    
                    # 색상 (importance 값에 따라)
                    colors.extend(values)
                
                # Scatter plot
                scatter = ax.scatter(x_values, y_positions, 
                                    c=colors, 
                                    cmap='coolwarm', 
                                    alpha=0.6, 
                                    s=25,
                                    edgecolors='black',
                                    linewidth=0.5)
                
                # Feature names (역순)
                y_labels = [feature_names[idx] for idx in reversed(feature_indices)]
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels(y_labels, fontsize=7)
                
                ax.set_xlabel('SHAP value', fontsize=8, fontweight='bold')
                ax.set_title('SHAP Summary Plot', fontsize=9, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3)
                
                # Colorbar
                cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
                cbar.set_label('Feature value', fontsize=7)
                
                # 레이아웃 조정
                fig.subplots_adjust(left=0.28, right=0.88, top=0.93, bottom=0.10)
                
                canvas = FigureCanvasQTAgg(fig)
                canvas.setSizePolicy(
                    QtWidgets.QSizePolicy.Ignored,  # UI 크기 따름
                    QtWidgets.QSizePolicy.Ignored
                )
                layout.addWidget(canvas)
                
                if self.text_shap:
                    self.text_shap.setPlainText("SHAP Summary Plot 표시 완료")
                    
            except Exception as e:
                if self.text_shap:
                    self.text_shap.setPlainText(f"SHAP 이미지 생성 실패: {str(e)}")
        
        # Widget이 없으면 텍스트로 표시
        elif self.text_shap:
            shap_text = "SHAP Value 분석\n"
            shap_text += "=" * 50 + "\n"
            shap_text += "widget_shap_ml_1 위젯이 없어 텍스트로 표시됩니다.\n\n"
            shap_text += "상위 10개 중요 피처:\n\n"
            
            importance_df = results['feature_importance']
            for idx, row in importance_df.head(10).iterrows():
                shap_text += f"{idx+1}. {row['feature']:20s} {row['importance']:.4f}\n"
            
            shap_text += "\nSHAP은 모델 예측에 각 피처가 기여한 정도를 나타냅니다."
            
            self.text_shap.setPlainText(shap_text)
    
    def _save_model(self, results):
        """모델 저장"""
        if not self.le_out_dir or not self.le_out_dir.text():
            print("저장 경로가 지정되지 않았습니다.")
            return
        
        out_dir = self.le_out_dir.text()
        
        # 모델 이름
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        active_models = results.get('active_models', ['SVM'])
        model_name = f"Ensemble_{'_'.join(active_models)}_{timestamp}"
        
        # .pt 파일 저장
        model_path = os.path.join(out_dir, f"{model_name}.pt")
        
        bundle = {
            'ensemble':      self.trained_model,          # MultiModelEnsemble
            'scaler':        results['scaler'],
            'threshold':     results['threshold'],
            'model_family':  'multi_model_seed_ensemble',
            'active_models': results.get('active_models', []),
            'feature_cols':  self._get_selected_features(),
            'sampling_rate': 1000000,
            'wave_mode':     'cols',
            'amp_cols':      [],
            'speed_col':     '속도'
        }
        
        torch.save(bundle, model_path)
        
        # JSON 파일 저장
        json_path = os.path.join(out_dir, f"{model_name}.json")
        
        json_data = {
            "model_family":    "multi_model_seed_ensemble",
            "schema_version":  2,
            "created_at":      datetime.now().isoformat(),
            "active_models":   results.get('active_models', []),
            "training_config": {
                "C":             self.model_json.get('training_config', {}).get('C', 1.0),
                "svm_kernel":    self.cb_kernel.currentText() if self.cb_kernel else 'rbf',
                "gamma":         "scale",
                "class_weight":  "balanced",
                "random_seeds":  [42 + i * 100 for i in range(
                                    self.spin_seed.value() if self.spin_seed else 5)],
                "cv_folds":      5
            },
            "preprocessing": {
                "signal": {
                    "gain":     2.5,
                    "clip_max": 32664,
                    "clip_min": -32664
                },
                "scaler": self.cb_scaler.currentText() if self.cb_scaler else 'MinMaxScaler',
            },
            "feature_extraction": {
                "sampling_rate": 1000000,
                "band_min":      200000,
                "band_max":      500000,
                "feature_cols":  self._get_selected_features()
            },
            "performance": {
                "validation": {
                    "accuracy":  float(results['val_metrics']['accuracy']),
                    "precision": float(results['val_metrics']['precision']),
                    "recall":    float(results['val_metrics']['recall']),
                    "f1_score":  float(results['val_metrics']['f1'])
                },
                "train": {
                    "accuracy": float(results['train_metrics']['accuracy']),
                    "f1_score": float(results['train_metrics']['f1'])
                },
                "per_model": {
                    name: {
                        "accuracy": float(m['accuracy']),
                        "f1_score": float(m['f1'])
                    }
                    for name, m in results.get('model_val_metrics', {}).items()
                }
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"모델 저장: {model_path}")
        print(f"JSON 저장: {json_path}")