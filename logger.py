# -*- coding: utf-8 -*-
"""logger.py - 이벤트 로그 관리"""

from datetime import datetime
from typing import Optional, Union
from PyQt5 import QtWidgets


class EventLogger:
    """
    이벤트 로그 관리 클래스
    QTextEdit 또는 QPlainTextEdit 위젯에 타임스탬프와 함께 로그 메시지 표시
    """
    
    def __init__(self, text_widget: Optional[Union[QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit]] = None):
        """
        Args:
            text_widget: 로그를 표시할 QTextEdit 또는 QPlainTextEdit 위젯 (None이면 로그 안함)
        """
        self.text_widget = text_widget
    
    def _get_timestamp(self) -> str:
        """현재 시간을 포맷팅"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _append_log(self, icon: str, message: str, color: str = "#000"):
        """
        로그 메시지 추가
        
        Args:
            icon: 아이콘 (이모지)
            message: 로그 메시지
            color: 텍스트 색상 (기본값: 검은색)
        """
        if not self.text_widget:
            return
        
        timestamp = self._get_timestamp()
        # 모든 색상을 검은색으로 통일
        log_line = f'<span style="color:#000;">[{timestamp}]</span> {icon} <span style="color:#000;">{message}</span>'
        
        # QTextEdit와 QPlainTextEdit 구분
        if isinstance(self.text_widget, QtWidgets.QPlainTextEdit):
            # QPlainTextEdit: appendHtml 사용
            self.text_widget.appendHtml(log_line)
        else:
            # QTextEdit: append 사용
            self.text_widget.append(log_line)
        
        # 자동 스크롤 (맨 아래로)
        scrollbar = self.text_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def log_info(self, message: str):
        """
        일반 정보 로그
        
        Args:
            message: 로그 메시지
        """
        self._append_log("", message, "#000")
    
    def log_success(self, message: str):
        """
        성공 로그 (초록색)
        
        Args:
            message: 로그 메시지
        """
        self._append_log("", message, "#000")
    
    def log_warning(self, message: str):
        """
        경고 로그 (주황색)
        
        Args:
            message: 로그 메시지
        """
        self._append_log("", message, "#000")
    
    def log_error(self, message: str):
        """
        에러 로그 (빨간색)
        
        Args:
            message: 로그 메시지
        """
        self._append_log("", message, "#000")
    
    def log_start(self, message: str):
        """
        시작 로그 (파란색)
        
        Args:
            message: 로그 메시지
        """
        self._append_log("", message, "#000")
    
    def log_data(self, message: str):
        """
        데이터 관련 로그 (보라색)
        
        Args:
            message: 로그 메시지
        """
        self._append_log("", message, "#000")
    
    def log_file(self, message: str):
        """
        파일 관련 로그 (회색)
        
        Args:
            message: 로그 메시지
        """
        self._append_log("", message, "#000")
    
    def log_model(self, message: str):
        """
        모델 관련 로그 (청록색)
        
        Args:
            message: 로그 메시지
        """
        self._append_log("", message, "#000")
    
    def clear(self):
        """로그 지우기"""
        if self.text_widget:
            self.text_widget.clear()
    
    def separator(self):
        """구분선 추가"""
        if self.text_widget:
            line = '<span style="color:#000;">═══════════════════════════════════════════════════════</span>'
            if isinstance(self.text_widget, QtWidgets.QPlainTextEdit):
                self.text_widget.appendHtml(line)
            else:
                self.text_widget.append(line)