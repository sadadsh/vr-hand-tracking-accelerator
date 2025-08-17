"""
Datasets package for VR Hand Tracking Accelerator

This package contains utilities for downloading, preprocessing, and managing
gesture datasets for realtime VR hand tracking applications.
"""

from .gesture_preprocessor import GesturePreprocessor, ProcessingConfig

__all__ = ['GesturePreprocessor', 'ProcessingConfig']
