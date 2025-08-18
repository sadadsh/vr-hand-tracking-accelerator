"""
VR Hand Tracking Accelerator - Annotations Package

This package contains utilities for downloading, preprocessing, and managing
gesture datasets for real-time VR hand tracking applications.

Author: Sadad Haidari
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Sadad Haidari"
__license__ = "MIT"

from .generate_annotations import HaGRIDAnnotationGenerator

__all__ = [
    'HaGRIDAnnotationGenerator',
    '__version__',
    '__author__',
    '__license__'
]
