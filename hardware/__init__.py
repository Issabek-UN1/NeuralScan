"""
Hardware communication package for NeuralScan.
"""

from .serial_controller import ArduinoController, MockSerialController, get_controller

__all__ = ["ArduinoController", "MockSerialController", "get_controller"]
