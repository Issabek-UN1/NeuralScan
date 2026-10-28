"""
UART Serial Driver & Turntable Controller for Arduino Mega 2560 + L298P
Enforces settling delays to dampen physical microvibrations.
"""

import time
import logging
from typing import Optional
import serial
import serial.tools.list_ports
import config

logger = logging.getLogger("NeuralScan.Hardware")


class ArduinoController:
    """
    Serial controller for Arduino Mega 2560 managing stepper motor and laser.
    Implements strict mechanical settling time rules.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = config.BAUD_RATE, timeout: float = config.SERIAL_TIMEOUT_S):
        self.port_name = port or config.SERIAL_PORT
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self.is_connected = False
        self.current_step = 0

    def connect(self) -> bool:
        """
        Establishes UART communication and waits for bootloader banner.
        """
        # If specified port doesn't exist, try auto-discovery
        available_ports = [p.device for p in serial.tools.list_ports.comports()]
        if self.port_name not in available_ports and len(available_ports) > 0:
            logger.info(f"Configured port {self.port_name} not found. Attempting discovery across: {available_ports}")
            self.port_name = available_ports[0]

        try:
            logger.info(f"Opening Serial port {self.port_name} at {self.baudrate} baud...")
            self.serial = serial.Serial(self.port_name, self.baudrate, timeout=self.timeout)
            
            # Wait for Arduino reboot reset (DTR toggle)
            time.sleep(1.8)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

            # Verify communication with Ping
            if self.ping():
                self.is_connected = True
                logger.info(f"Arduino Mega successfully handshake on {self.port_name}")
                return True
            else:
                logger.warning(f"Port {self.port_name} opened but handshake failed.")
                return False

        except serial.SerialException as e:
            logger.error(f"Failed to connect to Arduino on {self.port_name}: {e}")
            self.is_connected = False
            return False

    def send_command(self, cmd: str, wait_for_ack: bool = True) -> str:
        """
        Transmits ASCII command and optionally waits for 'OK:' or 'PONG:' response.
        """
        if not self.serial or not self.serial.is_open:
            raise ConnectionError("Serial port is not connected.")

        self.serial.write(cmd.encode("ascii"))
        self.serial.flush()

        if not wait_for_ack:
            return ""

        response = self.serial.readline().decode("ascii", errors="replace").strip()
        logger.debug(f"TX: '{cmd}' -> RX: '{response}'")
        return response

    def step(self, forward: bool = True) -> bool:
        """
        Executes a 7.5-degree step.
        MANDATORY: Enforces settling pause (config.SETTLING_TIME_S: 80-120 ms)
        to eliminate mechanical resonance before camera capture.
        """
        cmd = "S" if forward else "s"
        response = self.send_command(cmd)
        
        # Physical settling time: wait for turntable vibrations to dampen
        time.sleep(config.SETTLING_TIME_S)
        
        if "OK: STEP" in response:
            self.current_step = (self.current_step + (1 if forward else -1)) % config.TOTAL_STEPS
            return True
        return False

    def rotate_360(self) -> bool:
        """
        Triggers full 360-degree rotation (48 steps).
        """
        logger.info("Executing full 360-degree rotation...")
        response = self.send_command("R")
        time.sleep(config.SETTLING_TIME_S)
        return "OK: ROTATE" in response

    def return_to_zero(self) -> bool:
        """
        Returns turntable smoothly back to the initial zero (Home) position.
        """
        logger.info("Returning turntable to initial zero position (Home)...")
        response = self.send_command("H")
        self.current_step = 0
        time.sleep(config.SETTLING_TIME_S)
        return ("OK: HOME_ZERO" in response) or ("OK" in response)

    def laser_on(self) -> bool:
        """
        Turns the laser sheet ON.
        """
        response = self.send_command("L")
        return "OK: LASER ON" in response

    def laser_off(self) -> bool:
        """
        Turns the laser sheet OFF.
        """
        response = self.send_command("l")
        return "OK: LASER OFF" in response

    def ping(self) -> bool:
        """
        Heartbeat ping to verify MCU alive state.
        """
        try:
            response = self.send_command("P")
            return "PONG" in response
        except Exception:
            return False

    def emergency_cutoff(self) -> bool:
        """
        Immediately cuts power to stepper coils and laser module.
        """
        if self.serial and self.serial.is_open:
            response = self.send_command("E")
            return "OK" in response
        return False

    def close(self):
        """
        Closes serial connection safely after turning off coils and laser.
        """
        if self.serial and self.serial.is_open:
            try:
                self.emergency_cutoff()
            except Exception:
                pass
            self.serial.close()
            self.is_connected = False
            logger.info("Serial port closed.")


class MockSerialController:
    """
    High-fidelity simulation controller used when physical Arduino is not connected.
    Simulates step sequence, delays, and state machine.
    """

    def __init__(self, port: str = "MOCK_PORT", baudrate: int = config.BAUD_RATE, timeout: float = 1.0):
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_connected = True
        self.current_step = 0
        self.laser_active = False
        logger.info("MockSerialController initialized (Simulation Mode).")

    def connect(self) -> bool:
        self.is_connected = True
        logger.info("[MOCK] Connected to simulated Arduino Mega.")
        return True

    def step(self, forward: bool = True) -> bool:
        time.sleep(config.SETTLING_TIME_S)
        self.current_step = (self.current_step + (1 if forward else -1)) % config.TOTAL_STEPS
        logger.debug(f"[MOCK] Stepped to index {self.current_step} (Theta = {self.current_step * config.STEP_ANGLE_DEG:.1f} deg)")
        return True

    def rotate_360(self) -> bool:
        time.sleep(0.5)
        self.current_step = 0
        logger.info("[MOCK] Completed full 360 rotation.")
        return True

    def return_to_zero(self) -> bool:
        self.current_step = 0
        logger.info("[MOCK] Turntable returned to initial zero position (Home).")
        return True

    def laser_on(self) -> bool:
        self.laser_active = True
        logger.debug("[MOCK] Laser ON")
        return True

    def laser_off(self) -> bool:
        self.laser_active = False
        logger.debug("[MOCK] Laser OFF")
        return True

    def ping(self) -> bool:
        return True

    def emergency_cutoff(self) -> bool:
        self.laser_active = False
        logger.info("[MOCK] Emergency cutoff executed.")
        return True

    def close(self):
        self.is_connected = False
        logger.info("[MOCK] Serial connection closed.")


def get_controller(force_mock: bool = False) -> ArduinoController | MockSerialController:
    """
    Factory method: returns physical ArduinoController if hardware is connected,
    or gracefully falls back to MockSerialController.
    """
    if force_mock:
        return MockSerialController()

    controller = ArduinoController()
    if controller.connect():
        return controller

    logger.warning("Physical Arduino not detected. Automatically falling back to MockSerialController.")
    return MockSerialController()
