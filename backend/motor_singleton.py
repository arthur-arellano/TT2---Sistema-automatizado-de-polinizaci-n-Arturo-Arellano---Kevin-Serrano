"""
Motor Singleton - Gestión de instancia única del controlador de motor
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MotorManagerSingleton:
    """
    Singleton que maneja la instancia del controlador de motor.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not MotorManagerSingleton._initialized:
            self.motor = None
            self._simulation_mode = False
            MotorManagerSingleton._initialized = True
            self._initialize()
    
    def _initialize(self):
        """Inicializa el controlador de motor con fallback a simulación"""
        try:
            from motor_controller import MotorController
            self.motor = MotorController()
            self._simulation_mode = False
            logger.info("✅ Motor inicializado correctamente (hardware real)")
        except Exception as e:
            # Activamos modo simulación
            logger.error(f"⚠️ FALLO HARDWARE MOTOR: {e}")
            logger.warning("⚠️ Iniciando en MODO SIMULACIÓN (Backend funcionará sin motor)")
            self.motor = None
            self._simulation_mode = True
            # No hacemos raise RuntimeError
    
    def is_simulation_mode(self) -> bool:
        return self._simulation_mode
    
    def is_hardware_available(self) -> bool:
        return self.motor is not None
    
    def get_status(self) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.get_status()
        raise RuntimeError("Motor no inicializado")
    
    def polinizar_selectivo(
        self, 
        plantas: List[int],
        sensor_data: List[dict] = None,
        ignorar_condiciones: bool = True,
        duracion_vibracion: float = None
    ) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.polinizar_selectivo(
                plantas, 
                sensor_data, 
                ignorar_condiciones
            )
        raise RuntimeError("Motor no inicializado")
    
    def test_senal_externa(self, duracion: float = 2.0) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.test_senal_externa(duracion)
        raise RuntimeError("Motor no inicializado")
    
    def stop(self) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.stop()
        raise RuntimeError("Motor no inicializado")
    
    def ir_a_home(self) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.ir_a_home()
        raise RuntimeError("Motor no inicializado")
    
    def reset_completo(self) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.reset_completo()
        raise RuntimeError("Motor no inicializado")
    
    def verificar_condiciones_ambientales(
        self, 
        sensor_data: List[dict],
        usar_promedios: bool = True
    ) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.verificar_condiciones_ambientales(sensor_data)
        raise RuntimeError("Motor no inicializado")
    
    def get_condiciones_optimas(self) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.get_condiciones_optimas()
        raise RuntimeError("Motor no inicializado")
    
    def activate(self, duration: float = 2.0, steps: int = 200) -> Dict[str, Any]:
        if self.motor is not None:
            return self.motor.activate(duration, steps)
        raise RuntimeError("Motor no inicializado")


# Instancia global del singleton
motor_manager = MotorManagerSingleton()
