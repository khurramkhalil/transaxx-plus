"""
Runtime STL Monitoring for TransAxx Approximate DNNs

This module provides Signal Temporal Logic (STL) based runtime monitoring
for approximate Deep Neural Networks. It enables formal verification of
temporal properties during inference and provides adaptive approximation
control based on robustness semantics.

Key Components:
- STLFormulas: Predefined temporal formulas for approximate computing
- SignalCollector: Extract and track signals from model inference
- RuntimeMonitor: STL monitoring engine using rtamt
- AdaptiveController: Robustness-guided approximation adaptation

Example Usage:
    >>> from runtime_monitor import RuntimeMonitor, AdaptiveController
    >>> monitor = RuntimeMonitor()
    >>> controller = AdaptiveController(model, monitor)
    >>> result = controller.monitor_and_adapt(batch_metrics)
"""

__version__ = "1.0.0"
__author__ = "TransAxx Team"

from .monitor import RuntimeMonitor
from .controller import AdaptiveController
from .signal_collector import SignalCollector
from .stl_formulas import STLFormulaLibrary
from .config import MonitorConfig, MultiplierConfig

__all__ = [
    'RuntimeMonitor',
    'AdaptiveController',
    'SignalCollector',
    'STLFormulaLibrary',
    'MonitorConfig',
    'MultiplierConfig'
]
