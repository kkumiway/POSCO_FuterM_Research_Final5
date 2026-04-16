# -*- coding: utf-8 -*-
"""
training_2.py - ResNet18 STFT 학습 컨트롤러
tab_train_dl 탭 전용
"""

from __future__ import annotations
import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datetime import datetime
from typing import Optional, Dict, Any

from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import torchvision.models as tv_models
import torchvision.transforms as T

from scipy.signal import stft as scipy_stft
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 한글 폰트
# ─────────────────────────────────────────────
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def _setup_korean_font():
    try:
        import platform
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif platform.system() == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
        else:
            plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

_setup_korean_font()


# ─────────────────────────────────────────────
# CSV 로더 (training_1.py와 동일)
# ─────────────────────────────────────────────
RENAME_MAP = {
    "kinds": "부호", "code": "부호",
    "균열여부": "균열유무", "defect": "균열유무",
    "target": "균열유무", "y": "균열유무",
}

LABEL_MAP = {
    "정상": 0, "0": 0, "normal": 0, "Normal": 0, "NORMAL": 0,
    "균열": 1, "1": 1, "crack": 1, "Crack": 1, "CRACK": 1, "defect": 1,
}

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8", "utf-16"]:
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] >= 2:
                    df.columns = [str(c).strip() for c in df.columns]
                    df.rename(columns={k: v for k, v in RENAME_MAP.items()
                                       if k in df.columns}, inplace=True)
                    return df
            except Exception:
                pass
    raise ValueError(f"CSV 로드 실패: {path}")

def _extract_labels(df: pd.DataFrame):
    label_candidates = ["균열유무", "균열여부", "판정", "label", "target", "y"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError("라벨 컬럼을 찾을 수 없습니다.")
    y_raw = df[label_col].astype(str).str.strip()
    y = y_raw.map(LABEL_MAP)
    if y.isna().any():
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
    return y.astype(int).values

def _get_amp_cols(df: pd.DataFrame):
    """training_1.py의 detect_columns_and_wavecols와 동일한 로직으로 파형 컬럼 탐지"""
    cols = list(df.columns)

    # 1. "wave" 컬럼
    for i, c in enumerate(cols):
        if str(c).strip().lower() == "wave":
            return cols[i:i+1], "wave"

    # 2. 라벨 컬럼 위치 찾기 (파형은 라벨 앞까지)
    label_candidates = ["균열유무", "균열여부", "판정", "label", "Label", "crack", "Crack"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    label_idx = cols.index(label_col) if label_col else len(cols)

    # 3. 파형 시작 컬럼 찾기
    wave_start = 0
    for i, c in enumerate(cols):
        if str(c).strip().lower() == "wave":
            wave_start = i; break
    else:
        for i, c in enumerate(cols):
            c_str = str(c).strip().lower()
            if "signal" in c_str or "신호" in c_str:
                wave_start = min(i + 1, len(cols) - 1); break
        else:
            for i, c in enumerate(cols):
                if str(c).strip().lower().startswith("unnamed"):
                    wave_start = i; break
            else:
                for i, c in enumerate(cols):
                    try:
                        float(str(c).strip()); wave_start = i; break
                    except Exception:
                        pass

    if wave_start >= label_idx:
        wave_start = label_idx

    wave_cols = cols[wave_start:label_idx]

    # 4. 실제로 숫자로 변환 가능한 컬럼만 남기기
    valid = []
    for c in wave_cols:
        try:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > len(df) * 0.5:   # 50% 이상 유효한 숫자
                valid.append(c)
        except Exception:
            pass

    mode = "wave" if (len(valid) == 1 and valid[0] == "wave") else "cols"
    return valid, mode


# ─────────────────────────────────────────────
# SpecAugment (원본 코드 동일)
# ─────────────────────────────────────────────
class TimeFreqMask:
    def __init__(self, time_masks=2, freq_masks=2,
                 time_mask_ratio=0.15, freq_mask_ratio=0.15, p=0.7):
        self.time_masks      = time_masks
        self.freq_masks      = freq_masks
        self.time_mask_ratio = time_mask_ratio
        self.freq_mask_ratio = freq_mask_ratio
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        C, H, W = x.shape
        for _ in range(self.freq_masks):
            m = int(H * self.freq_mask_ratio)
            if m <= 0: continue
            f = torch.randint(0, max(1, m + 1), (1,)).item()
            if f == 0: continue
            f0 = torch.randint(0, max(1, H - f + 1), (1,)).item()
            x[:, f0:f0 + f, :] = 0.0
        for _ in range(self.time_masks):
            m = int(W * self.time_mask_ratio)
            if m <= 0: continue
            t = torch.randint(0, max(1, m + 1), (1,)).item()
            if t == 0: continue
            t0 = torch.randint(0, max(1, W - t + 1), (1,)).item()
            x[:, :, t0:t0 + t] = 0.0
        return x


def _get_train_augmentation():
    """원본 코드와 동일: TimeFreqMask + RandomErasing"""
    return T.Compose([
        TimeFreqMask(time_masks=2, freq_masks=2,
                     time_mask_ratio=0.15, freq_mask_ratio=0.15, p=0.7),
        T.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
    ])


# ─────────────────────────────────────────────
# STFT Spectrogram Dataset (원본 코드 동일)
# ─────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, signals, labels, config: dict, augment=False):
        self.signals  = signals
        self.labels   = labels
        self.config   = config
        self.augment  = augment
        self.resize   = T.Resize((config["img_size"], config["img_size"]), antialias=True)
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])
        # 학습 시 augmentation
        self.transform = _get_train_augmentation() if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = np.asarray(self.signals[idx], dtype=np.float32)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        # STFT (원본 signal_to_spectrogram + normalize_spectrogram 동일)
        _, _, Zxx = scipy_stft(
            sig,
            fs=self.config["sampling_rate"],
            window=self.config["window"],
            nperseg=self.config["nperseg"],
            noverlap=self.config["noverlap"],
            nfft=self.config["nfft"],
        )
        mag = 20 * np.log10(np.abs(Zxx) + 1e-10)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        s_min, s_max = mag.min(), mag.max()
        if s_max - s_min < 1e-8:
            mag = np.zeros_like(mag)
        else:
            mag = (mag - s_min) / (s_max - s_min)

        t = torch.tensor(mag, dtype=torch.float32).unsqueeze(0)
        t = self.resize(t)
        if self.transform:
            t = self.transform(t)
        t = self.normalize(t)
        return t, torch.tensor(int(self.labels[idx]), dtype=torch.long)


# ─────────────────────────────────────────────
# ResNet18 빌드 (원본 build_model 동일)
# ─────────────────────────────────────────────
def _patch_conv1(model):
    old = model.conv1
    if old.in_channels == 1:
        return model
    new_conv = nn.Conv2d(1, old.out_channels,
                         kernel_size=old.kernel_size,
                         stride=old.stride,
                         padding=old.padding,
                         bias=(old.bias is not None))
    with torch.no_grad():
        new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        if old.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old.bias)
    model.conv1 = new_conv
    return model


def _build_resnet18(num_classes=2, freeze_backbone=True):
    """원본 build_model과 동일"""
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model = _patch_conv1(model)
    if freeze_backbone:
        for _, param in model.named_parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(in_features, 128), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(128, num_classes),
    )
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def _set_trainable_backbone_layers(model, trainable_layers):
    """원본 set_trainable_backbone_layers 동일"""
    for lname in ["layer1", "layer2", "layer3", "layer4"]:
        module = getattr(model, lname)
        req = (lname in trainable_layers)
        for p in module.parameters():
            p.requires_grad = req


def _compute_trainable_layers(epoch, unfreeze_epoch):
    """원본 compute_trainable_layers 동일"""
    if epoch < unfreeze_epoch:
        return []
    if epoch < unfreeze_epoch + 10:
        return ["layer4"]
    if epoch < unfreeze_epoch + 20:
        return ["layer4", "layer3"]
    return ["layer4", "layer3", "layer2"]


def _freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def _make_optimizer_and_scheduler(model, cfg):
    """원본 make_optimizer_and_scheduler 동일 — 레이어별 차등 LR"""
    base_lr = cfg["lr"]
    bb_lr   = base_lr * cfg["backbone_lr_mult"]
    stem_params = []
    if hasattr(model, "conv1"): stem_params += list(model.conv1.parameters())
    if hasattr(model, "bn1"):   stem_params += list(model.bn1.parameters())
    param_groups = [
        {"name": "head",   "params": model.fc.parameters(),     "lr": base_lr},
        {"name": "layer4", "params": model.layer4.parameters(), "lr": bb_lr},
        {"name": "layer3", "params": model.layer3.parameters(), "lr": bb_lr},
        {"name": "layer2", "params": model.layer2.parameters(), "lr": bb_lr},
        {"name": "layer1", "params": model.layer1.parameters(), "lr": bb_lr},
        {"name": "stem",   "params": stem_params,               "lr": bb_lr},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["num_epochs"], eta_min=1e-6)
    return optimizer, scheduler


# ─────────────────────────────────────────────
# 학습 워커 (QThread)
# ─────────────────────────────────────────────
class ResNetTrainingWorker(QtCore.QThread):
    progress   = QtCore.pyqtSignal(int, str)
    epoch_done = QtCore.pyqtSignal(int, float, float, float, float)  # epoch, tr_loss, tr_acc, va_loss, va_acc
    finished   = QtCore.pyqtSignal(dict)
    error      = QtCore.pyqtSignal(str)

    def __init__(self, train_signals, train_labels,
                       val_signals,   val_labels,
                       config: dict):
        super().__init__()
        self.train_signals = train_signals
        self.train_labels  = train_labels
        self.val_signals   = val_signals
        self.val_labels    = val_labels
        self.config        = config
        self.is_stopped    = False

    def stop(self):
        self.is_stopped = True

    def run(self):
        try:
            cfg = self.config
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ── 시드 고정 (원본 seed_everything 동일)
            import random
            seed = cfg.get("random_seed", 42)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

            self.progress.emit(1, f"Device: {device} | Seed: {seed}")

            # ── Dataset / DataLoader
            train_ds = SpectrogramDataset(
                self.train_signals, self.train_labels, cfg, augment=True)
            val_ds   = SpectrogramDataset(
                self.val_signals,   self.val_labels,   cfg, augment=False)

            # WeightedSampler
            counts = np.bincount(self.train_labels, minlength=2)
            w = 1.0 / np.maximum(counts, 1)
            sample_w = [w[l] for l in self.train_labels]
            sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

            train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                      sampler=sampler, num_workers=0, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                                      shuffle=False, num_workers=0, pin_memory=True)

            # ── 모델 (원본 build_model 동일)
            model = _build_resnet18(num_classes=2, freeze_backbone=True).to(device)
            _freeze_bn(model)

            criterion = nn.CrossEntropyLoss()

            # ── Optimizer: 레이어별 차등 LR (원본 make_optimizer_and_scheduler 동일)
            optimizer, scheduler = _make_optimizer_and_scheduler(model, cfg)

            best_f1    = -1.0
            best_state = None
            best_epoch = 0
            patience_cnt = 0

            tr_losses, va_losses = [], []
            tr_accs,   va_accs   = [], []

            total_epochs = cfg["num_epochs"]
            unfreeze_ep  = cfg.get("unfreeze_epoch", 10)

            for epoch in range(total_epochs):
                if self.is_stopped:
                    self.progress.emit(0, "학습 중지됨")
                    return

                # ── 점진적 unfreeze (원본 compute_trainable_layers 동일)
                trainable = _compute_trainable_layers(epoch, unfreeze_ep)
                _set_trainable_backbone_layers(model, trainable)
                _freeze_bn(model)

                # ── Train
                model.train()
                _freeze_bn(model)
                tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    out  = model(xb)
                    loss = criterion(out, yb)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    optimizer.step()  # 원본과 동일 (gradient clipping 없음)
                    tr_loss_sum += loss.item() * xb.size(0)
                    tr_correct  += (out.argmax(1) == yb).sum().item()
                    tr_total    += xb.size(0)

                tr_loss = tr_loss_sum / max(1, tr_total)
                tr_acc  = tr_correct  / max(1, tr_total)

                # ── Validation
                model.eval()
                _freeze_bn(model)
                va_loss_sum, va_correct, va_total = 0.0, 0, 0
                all_preds, all_labels, all_probs = [], [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        out  = model(xb)
                        loss = criterion(out, yb)
                        va_loss_sum += loss.item() * xb.size(0)
                        prob = torch.softmax(out, dim=1)[:, 1]
                        pred = (prob >= cfg["threshold"]).long()
                        va_correct  += (pred == yb).sum().item()
                        va_total    += xb.size(0)
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(yb.cpu().numpy())
                        all_probs.extend(prob.cpu().numpy())

                va_loss = va_loss_sum / max(1, va_total)
                va_acc  = va_correct  / max(1, va_total)
                val_f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)

                scheduler.step()

                tr_losses.append(tr_loss); va_losses.append(va_loss)
                tr_accs.append(tr_acc);    va_accs.append(va_acc)

                pct = int((epoch + 1) / total_epochs * 100)
                self.progress.emit(pct,
                    f"Epoch {epoch+1}/{total_epochs} | "
                    f"Loss {tr_loss:.4f}/{va_loss:.4f} | "
                    f"Acc {tr_acc:.3f}/{va_acc:.3f} | F1 {val_f1:.4f}")
                self.epoch_done.emit(epoch + 1, tr_loss, tr_acc, va_loss, va_acc)

                # Early stopping
                if val_f1 > best_f1 + 1e-6:
                    best_f1    = val_f1
                    best_epoch = epoch + 1
                    patience_cnt = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= cfg["patience"]:
                        self.progress.emit(pct,
                            f"Early stop (best: ep {best_epoch}, F1={best_f1:.4f})")
                        break

            # ── Best 모델로 최종 평가
            if best_state:
                model.load_state_dict(best_state)
                model.to(device)
            model.eval()
            _freeze_bn(model)

            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    prob = torch.softmax(model(xb), dim=1)[:, 1]
                    pred = (prob >= cfg["threshold"]).long()
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(yb.numpy())
                    all_probs.extend(prob.cpu().numpy())

            all_preds  = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs  = np.array(all_probs)

            cm  = confusion_matrix(all_labels, all_preds)
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)

            self.finished.emit({
                "best_state":   best_state,
                "best_epoch":   best_epoch,
                "best_f1":      float(best_f1),
                "accuracy":     float(accuracy_score(all_labels, all_preds)),
                "precision":    float(precision_score(all_labels, all_preds, zero_division=0)),
                "recall":       float(recall_score(all_labels, all_preds, zero_division=0)),
                "f1":           float(f1_score(all_labels, all_preds, zero_division=0)),
                "roc_auc":      float(roc_auc),
                "cm":           cm,
                "roc_curve":    (fpr, tpr, roc_auc),
                "tr_losses":    tr_losses,
                "va_losses":    va_losses,
                "tr_accs":      tr_accs,
                "va_accs":      va_accs,
                "val_preds":    all_preds,
                "val_labels":   all_labels,
                "val_probs":    all_probs,
                "config":       cfg,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ─────────────────────────────────────────────
# DL Training Controller
# ─────────────────────────────────────────────
class DLTrainingController(QtCore.QObject):

    def __init__(
        self,
        main_window,
        # ── 프로젝트 정보 표시 (groupBox_11 / groupBox_10)
        textBrowser_project_6=None,
        textBrowser_project_date_6=None,
        textBrowser_process_6=None,
        textBrowser_product_6=None,
        textBrowser_model_6=None,
        textBrowser_traindata_6=None,
        textBrowser_validdata_6=None,
        textBrowser_detail_6=None,
        # ── 학습 제어
        btn_start=None,          # btn_train_start_ml_2
        progress_bar=None,       # progress_train_ml_3
        # ── 실시간 그래프 (Loss / Val-Acc)
        label_loss=None,         # label_train_plot_loss_dl
        label_val=None,          # label_train_plot_val_dl
        # ── 학습 로그 (epoch-by-epoch)
        text_log=None,           # text_train_log_dl
        # ── 최종 결과 위젯
        widget_cm=None,          # label_train_plot_loss_ml_2  → Confusion Matrix
        widget_roc=None,         # label_train_plot_val_ml_2   → ROC Curve
        widget_grad=None,        # widget_Featureimportance_ml_1 → GradCAM / 클래스별 확률 분포
        text_report=None,        # text_train_report_1
        # ── 디렉토리
        models_dir=None,
        projects_dir=None,
        # ── 연동 컨트롤러
        train_proj_ctrl=None,
        logger=None,
    ):
        super().__init__(main_window)
        self.main_window = main_window

        # 프로젝트 표시 위젯
        self.tb_project      = textBrowser_project_6
        self.tb_date         = textBrowser_project_date_6
        self.tb_process      = textBrowser_process_6
        self.tb_product      = textBrowser_product_6
        self.tb_model        = textBrowser_model_6
        self.tb_traindata    = textBrowser_traindata_6
        self.tb_validdata    = textBrowser_validdata_6
        self.tb_detail       = textBrowser_detail_6

        # 제어
        self.btn_start       = btn_start
        self.progress_bar    = progress_bar

        # 실시간 그래프
        self.label_loss      = label_loss
        self.label_val       = label_val

        # 로그
        self.text_log        = text_log

        # 결과 위젯
        self.widget_cm       = widget_cm
        self.widget_roc      = widget_roc
        self.widget_grad     = widget_grad
        self.text_report     = text_report

        # 디렉토리
        self.models_dir      = models_dir or os.path.join(os.getcwd(), "Models")
        self.projects_dir    = projects_dir or os.path.join(os.getcwd(), "projects")

        # 연동
        self.train_proj_ctrl = train_proj_ctrl
        self.logger          = logger

        # 상태
        self.worker          = None
        self.train_df        = None
        self.val_df          = None
        self._tr_losses      = []
        self._va_losses      = []
        self._tr_accs        = []
        self._va_accs        = []

        self._connect_signals()

    # ──────────────────────────────────────────
    def _connect_signals(self):
        if self.btn_start:
            self.btn_start.clicked.connect(self.start_training)

    # ──────────────────────────────────────────
    def update_project_display(self, project_info: dict):
        """TrainingProjectController.project_loaded 시그널로 호출됨"""
        def _set(widget, key):
            if widget:
                val = project_info.get(key, "")
                widget.setHtml(
                    f'<div style="padding-left:10px;padding-top:8px;">{val}</div>')

        _set(self.tb_project,   "project_filename")
        _set(self.tb_date,      "created_date")
        _set(self.tb_process,   "target_process")
        _set(self.tb_product,   "target_product")
        _set(self.tb_model,     "model_filename")
        _set(self.tb_traindata, "train_data_filename")
        _set(self.tb_validdata, "valid_data_filename")
        _set(self.tb_detail,    "model_date")

        # 프로젝트에서 CSV 경로 자동 로드
        train_path = project_info.get("train_data_path", "")
        val_path   = project_info.get("valid_data_path", "")
        if train_path and os.path.exists(train_path):
            try:
                self.train_df = _read_csv_robust(train_path)
                self._log(f"학습 데이터 자동 로드: {os.path.basename(train_path)} "
                          f"({len(self.train_df)}행)")
            except Exception as e:
                self._log(f"학습 CSV 자동 로드 실패: {e}")
        if val_path and os.path.exists(val_path):
            try:
                self.val_df = _read_csv_robust(val_path)
                self._log(f"검증 데이터 자동 로드: {os.path.basename(val_path)} "
                          f"({len(self.val_df)}행)")
            except Exception as e:
                self._log(f"검증 CSV 자동 로드 실패: {e}")

    # ──────────────────────────────────────────
    def start_training(self):
        # ── 데이터 확인
        if self.train_df is None and self.train_proj_ctrl:
            if self.train_proj_ctrl.current_train_data is not None:
                self.train_df = self.train_proj_ctrl.current_train_data
        if self.val_df is None and self.train_proj_ctrl:
            if self.train_proj_ctrl.current_valid_data is not None:
                self.val_df = self.train_proj_ctrl.current_valid_data

        if self.train_df is None:
            QtWidgets.QMessageBox.warning(
                self.main_window, "데이터 없음",
                "프로젝트에서 학습 CSV를 먼저 로드하세요.")
            return
        if self.val_df is None:
            QtWidgets.QMessageBox.warning(
                self.main_window, "데이터 없음",
                "프로젝트에서 검증 CSV를 먼저 로드하세요.")
            return

        # ── 신호 / 라벨 추출
        try:
            amp_tr, _ = _get_amp_cols(self.train_df)
            amp_va, _ = _get_amp_cols(self.val_df)
            if not amp_tr:
                raise ValueError("학습 CSV에서 파형 컬럼을 찾을 수 없습니다.")

            train_signals = self.train_df[amp_tr].values.astype(np.float32)
            train_signals = np.nan_to_num(train_signals, nan=0.0, posinf=0.0, neginf=0.0)
            train_labels  = _extract_labels(self.train_df)
            val_signals   = self.val_df[amp_va].values.astype(np.float32)
            val_signals   = np.nan_to_num(val_signals, nan=0.0, posinf=0.0, neginf=0.0)
            val_labels    = _extract_labels(self.val_df)

            n_tr = np.bincount(train_labels, minlength=2)
            n_va = np.bincount(val_labels,   minlength=2)
            self._log(f"학습: {len(train_labels)}건 (정상 {n_tr[0]}, 균열 {n_tr[1]})")
            self._log(f"검증: {len(val_labels)}건 (정상 {n_va[0]}, 균열 {n_va[1]})")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main_window, "데이터 오류", str(e))
            return

        # ── 학습 설정 (기본값 고정, 추후 UI에 파라미터 추가 가능)
        config = {
            "sampling_rate":   1_000_000,
            "nperseg":         128,
            "noverlap":        96,
            "nfft":            256,
            "window":          "hann",
            "img_size":        224,
            "num_epochs":      50,
            "batch_size":      16,
            "lr":              1e-4,
            "weight_decay":    1e-4,
            "patience":        20,
            "threshold":       0.5,
            "unfreeze_epoch":  10,
            "backbone_lr_mult": 0.1,   # 원본 동일
            "random_seed":       42,    # 원본 seed_everything(42) 동일
            "waveform_length": train_signals.shape[1],  # 🎯 실제 신호 길이 저장
        }

        # ── UI 초기화
        self._tr_losses.clear(); self._va_losses.clear()
        self._tr_accs.clear();   self._va_accs.clear()
        if self.progress_bar:
            self.progress_bar.setValue(0)
        if self.btn_start:
            self.btn_start.setEnabled(False)
        self._log("=" * 40)
        self._log("ResNet18 STFT 학습 시작")
        self._log(f"Epochs: {config['num_epochs']}, Batch: {config['batch_size']}, "
                  f"LR: {config['lr']}")
        self._log("=" * 40)

        # ── 워커 시작
        self.worker = ResNetTrainingWorker(
            train_signals, train_labels,
            val_signals,   val_labels,
            config
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.epoch_done.connect(self._on_epoch_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    # ──────────────────────────────────────────
    def _on_progress(self, pct: int, msg: str):
        if self.progress_bar:
            self.progress_bar.setValue(pct)
        self._log(msg)

    def _on_epoch_done(self, epoch, tr_loss, tr_acc, va_loss, va_acc):
        self._tr_losses.append(tr_loss); self._va_losses.append(va_loss)
        self._tr_accs.append(tr_acc);    self._va_accs.append(va_acc)
        self._update_live_graphs()

    def _on_finished(self, results: dict):
        if self.progress_bar:
            self.progress_bar.setValue(100)
        if self.btn_start:
            self.btn_start.setEnabled(True)

        self._log("=" * 40)
        self._log(f"학습 완료! Best Epoch: {results['best_epoch']}")
        self._log(f"Accuracy : {results['accuracy']*100:.2f}%")
        self._log(f"Precision: {results['precision']*100:.2f}%")
        self._log(f"Recall   : {results['recall']*100:.2f}%")
        self._log(f"F1-Score : {results['f1']*100:.2f}%")
        self._log(f"ROC-AUC  : {results['roc_auc']:.4f}")
        self._log("=" * 40)

        self._display_confusion_matrix(results["cm"])
        self._display_roc_curve(results["roc_curve"])
        self._display_prob_distribution(results)
        self._display_report(results)
        self._save_model(results)

    def _on_error(self, msg: str):
        if self.btn_start:
            self.btn_start.setEnabled(True)
        self._log(f"오류 발생: {msg}")
        QtWidgets.QMessageBox.critical(
            self.main_window, "학습 실패", msg)

    # ──────────────────────────────────────────
    # 실시간 그래프 (Loss + Val-Acc 동시)
    # ──────────────────────────────────────────
    def _update_live_graphs(self):
        epochs = list(range(1, len(self._tr_losses) + 1))
        if not epochs:
            return

        # Loss 그래프
        if self.label_loss:
            self._draw_to_label(
                self.label_loss,
                lambda ax: (
                    ax.plot(epochs, self._tr_losses, label="Train Loss", color="#2563eb"),
                    ax.plot(epochs, self._va_losses, label="Val Loss",   color="#dc2626"),
                    ax.set_title("Loss", fontsize=9),
                    ax.set_xlabel("Epoch"), ax.legend(fontsize=7), ax.grid(alpha=0.3)
                )
            )

        # Val Accuracy 그래프
        if self.label_val:
            self._draw_to_label(
                self.label_val,
                lambda ax: (
                    ax.plot(epochs, self._tr_accs, label="Train Acc", color="#059669"),
                    ax.plot(epochs, self._va_accs, label="Val Acc",   color="#d97706"),
                    ax.set_title("Accuracy", fontsize=9),
                    ax.set_xlabel("Epoch"), ax.legend(fontsize=7), ax.grid(alpha=0.3)
                )
            )

    # ──────────────────────────────────────────
    # 최종 결과 표시
    # ──────────────────────────────────────────
    def _display_confusion_matrix(self, cm):
        if not self.widget_cm:
            return
        def _plot(ax):
            im = ax.imshow(cm, cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=[0,1], yticks=[0,1],
                   xticklabels=["정상","균열"],
                   yticklabels=["정상","균열"],
                   title="Confusion Matrix",
                   ylabel="실제", xlabel="예측")
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i,j]),
                            ha="center", va="center",
                            color="white" if cm[i,j] > thresh else "black")
        self._draw_to_widget(self.widget_cm, _plot, figsize=(3.5, 3.2))

    def _display_roc_curve(self, roc_data):
        if not self.widget_roc:
            return
        fpr, tpr, roc_auc = roc_data
        def _plot(ax):
            ax.plot(fpr, tpr, color="darkorange", lw=2,
                    label=f"ROC (AUC={roc_auc:.3f})")
            ax.plot([0,1],[0,1], color="navy", lw=1.5, linestyle="--")
            ax.set(xlim=[0,1], ylim=[0,1.05],
                   xlabel="FPR", ylabel="TPR", title="ROC Curve")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        self._draw_to_widget(self.widget_roc, _plot, figsize=(3.5, 3.2))

    def _display_prob_distribution(self, results):
        """균열 확률 분포 (정상 vs 균열 히스토그램) — widget_Featureimportance_ml_1"""
        if not self.widget_grad:
            return
        probs  = results["val_probs"]
        labels = results["val_labels"]
        def _plot(ax):
            ax.hist(probs[labels==0], bins=20, alpha=0.6,
                    color="#2563eb", label="정상")
            ax.hist(probs[labels==1], bins=20, alpha=0.6,
                    color="#dc2626", label="균열")
            ax.axvline(results["config"]["threshold"],
                       color="black", linestyle="--", lw=1.5,
                       label=f"Threshold={results['config']['threshold']}")
            ax.set(title="균열 확률 분포",
                   xlabel="Crack Probability", ylabel="Count")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        self._draw_to_widget(self.widget_grad, _plot, figsize=(4.5, 3.2))

    def _display_report(self, results):
        if not self.text_report:
            return
        cm = results["cm"]
        TN, FP = int(cm[0,0]), int(cm[0,1])
        FN, TP = int(cm[1,0]), int(cm[1,1])
        total  = TN + FP + FN + TP
        lines  = [
            "=" * 30,
            "ResNet18 학습 결과 리포트",
            "=" * 30,
            "",
            f"Best Epoch : {results['best_epoch']}",
            f"Best Val F1: {results['best_f1']*100:.2f}%",
            "",
            "[ 검증 성능 ]",
            f"  Accuracy : {results['accuracy']*100:.2f}%  ({TP+TN}/{total})",
            f"  Precision: {results['precision']*100:.2f}%  ({TP}/{TP+FP})",
            f"  Recall   : {results['recall']*100:.2f}%  ({TP}/{TP+FN})",
            f"  F1-Score : {results['f1']*100:.2f}%",
            f"  ROC-AUC  : {results['roc_auc']:.4f}",
            "",
            "[ Confusion Matrix ]",
            f"  TN (정상→정상): {TN}",
            f"  FP (정상→균열): {FP}",
            f"  FN (균열→정상): {FN}",
            f"  TP (균열→균열): {TP}",
            "",
            "[ 분석 ]",
        ]
        if results["accuracy"] > 0.95:
            lines.append("  매우 우수한 성능")
        elif results["accuracy"] > 0.90:
            lines.append("  우수한 성능")
        else:
            lines.append("  성능 개선 필요")

        if FN > 0:
            lines.append(f"  균열 놓침(FN) {FN}건 — Threshold 낮추기 고려")
        if FP > 0:
            lines.append(f"  오검출(FP) {FP}건")
        if abs(max(self._tr_accs or [0]) - results["accuracy"]) > 0.05:
            lines.append("  과적합 가능성 있음 (Train/Val 차이 큼)")
        else:
            lines.append("  과적합 없음")

        lines += ["", "=" * 30]
        self.text_report.setPlainText("\n".join(lines))

    # ──────────────────────────────────────────
    # 모델 저장
    # ──────────────────────────────────────────
    def _save_model(self, results):
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"ResNet18_STFT_{ts}"
            pt_path    = os.path.join(self.models_dir, f"{model_name}.pth")
            json_path  = os.path.join(self.models_dir, f"{model_name}.json")

            # .pth — raw state_dict
            torch.save(results["best_state"], pt_path)

            # .json
            cfg = results["config"]
            project_info = {}
            if self.train_proj_ctrl:
                project_info = self.train_proj_ctrl.get_current_project_info() or {}

            json_data = {
                "model_family": "pytorch",
                "backbone":     "resnet18",
                "schema_version": 1,
                "created_at":   datetime.now().isoformat(),
                "model_config": {
                    "use_single_channel": True,
                    "num_classes": 2,
                },
                "inference_config": {
                    "threshold":     cfg["threshold"],
                    "sampling_rate": cfg["sampling_rate"],
                    "waveform_length": cfg.get("waveform_length", 1000),  # 🎯 신호 길이 저장
                },
                "stft_config": {
                    "nperseg":  cfg["nperseg"],
                    "noverlap": cfg["noverlap"],
                    "nfft":     cfg["nfft"],
                    "window":   cfg["window"],
                    "img_size": cfg["img_size"],
                },
                "application_info": {
                    "대상 재질": project_info.get("target_process", ""),
                    "대상 부호": project_info.get("target_product", ""),
                    "설명":      f"ResNet18 STFT 균열 탐지 (Best F1: {results['best_f1']*100:.1f}%)",
                    "베포 날짜": datetime.now().strftime("%Y-%m-%d"),
                    "만든 사람": "",
                },
                "performance": {
                    "accuracy":  float(results["accuracy"]),
                    "precision": float(results["precision"]),
                    "recall":    float(results["recall"]),
                    "f1_score":  float(results["f1"]),
                    "roc_auc":   float(results["roc_auc"]),
                    "best_epoch": results["best_epoch"],
                },
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            self._log(f"모델 저장 완료: {model_name}.pth")
            self._log(f"JSON  저장 완료: {model_name}.json")

            if self.logger:
                self.logger.log_model(f"ResNet18 모델 저장: {model_name}.pth")

            QtWidgets.QMessageBox.information(
                self.main_window, "저장 완료",
                f"모델이 저장되었습니다.\n\n"
                f"경로: {self.models_dir}\n"
                f"파일: {model_name}.pth\n"
                f"Acc : {results['accuracy']*100:.2f}%  "
                f"F1: {results['f1']*100:.2f}%")

        except Exception as e:
            self._log(f"모델 저장 실패: {e}")
            import traceback; traceback.print_exc()

    # ──────────────────────────────────────────
    # 공통 그래프 헬퍼
    # ──────────────────────────────────────────
    def _draw_to_label(self, label_widget, plot_fn, figsize=(3.5, 2.8)):
        """QLabel 안에 matplotlib 그래프 표시"""
        layout = label_widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            layout = QtWidgets.QVBoxLayout(label_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            label_widget.setLayout(layout)

        fig = Figure(figsize=figsize, dpi=90)
        ax  = fig.add_subplot(111)
        plot_fn(ax)
        fig.tight_layout(pad=0.5)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        layout.addWidget(canvas)

    def _draw_to_widget(self, widget, plot_fn, figsize=(4, 3.5)):
        """QWidget 안에 matplotlib 그래프 표시"""
        layout = widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            layout = QtWidgets.QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

        fig = Figure(figsize=figsize, dpi=100)
        ax  = fig.add_subplot(111)
        plot_fn(ax)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        layout.addWidget(canvas)

    def _log(self, msg: str):
        if self.text_log:
            self.text_log.append(msg)
            sb = self.text_log.verticalScrollBar()
            sb.setValue(sb.maximum())
        print(msg)