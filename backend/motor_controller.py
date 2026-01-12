"""
Motor Controller para Raspberry Pi 5 con lgpio
Sistema de Polinización Automatizada de Tomates Cherry

PINES GPIO:
- Motor Horizontal: PUL=17, DIR=27
- Motor Vertical: PUL=22, DIR=23
- Señal Externa (Relay): GPIO24

"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURACIÓN DE HARDWARE
# ============================================

# Pines GPIO (Raspberry Pi 5 - lgpio)
PIN_PUL_H = 17    # Pulsos motor horizontal
PIN_DIR_H = 27    # Dirección motor horizontal
PIN_PUL_V = 22    # Pulsos motor vertical
PIN_DIR_V = 23    # Dirección motor vertical
PIN_SIGNAL = 24   # Señal externa (relay/vibrador)

# Parámetros del motor
PASOS_POR_REV = 200       # Pasos por revolución (1.8° por paso)
MICROSTEPPING = 1         # Sin microstepping
MM_POR_REV_H = 40.0       # mm por revolución horizontal
MM_POR_REV_V = 40.0       # mm por revolución vertical

# Velocidad (microsegundos entre pulsos)
DELAY_H = 1000            # Motor horizontal
DELAY_V = 1000            # Motor vertical

# Distancias (mm)
DISTANCIAS_PLANTAS = {
    1: 200.0,    # mm desde home a planta 1
    2: 740.0,    # mm desde planta 1 a planta 2
    3: 600.0,    # mm desde planta 2 a planta 3
    4: 758.0,    # mm desde planta 3 a planta 4
}
DISTANCIA_VERTICAL = 25.0  # mm de acercamiento

# Tiempos
DELAY_ESTABILIZACION = 2.0   # segundos antes de vibrar
DURACION_VIBRACION = 5.0    # segundos de vibración
DELAY_VERTICAL = 2.0         # segundos de espera después de mover Y 

# ============================================
# CONDICIONES AMBIENTALES
# Basado en literatura científica
# ============================================
CONDICIONES_OPTIMAS = {
    "temperatura": {
        "min": 18.0,        # °C - [1] Arshad 2024: "18-29°C during the day"
        "max": 30.0,        # °C - [2] Šalagovič 2024: "Optimal range 15-30°C"
        "critico": 32.0,    # °C - [4] Miller 2021: "Viability drops 66% at 32/26°C"
        "severo": 35.0,     # °C - Extrapolación prudente
        "optimo_min": 21.0, # °C - [1] Arshad 2024: "Peak growth at 21.32°C"
        "optimo_max": 27.0, # °C - Conservador (vs 29°C en Arshad 2024)
    },
    "humedad": {
        "min": 40.0,        # % - Estrés hídrico severo
        "max": 80.0,        # % - [1] Arshad 2024: "Fungal diseases above this"
        "optimo_min": 60.0, # % - Estándar industrial
        "optimo_max": 70.0, # % - [1] Arshad 2024: "High humidity enhances yield"
    },
    "luz": {
        "min": 1000,        # lux - No polinizar en penumbra
        "max": 100000,      # lux - Máximo (luz solar directa intensa)
        "optimo_min": 10000,# lux - [1] Arshad 2024: "7,751-12,022 lux medido"
        "optimo_max": 50000,# lux - Rango típico invernadero
    },
    "conductividad": {
        "min": 200,         # µS/cm - Subfertilización severa
        "max": 2000,        # µS/cm - [1] Arshad 2024 usa 2,800-3,000 µS/cm
        "optimo_min": 500,  # µS/cm - Cultivo conservador
        "optimo_max": 1500, # µS/cm - Cultivo conservador
    }
}

# Configuración de promedios
VENTANA_PROMEDIO_MINUTOS = 60  # Minutos para calcular promedio
LECTURAS_PARA_PROMEDIO = 4   # Número de lecturas a promediar

# ============================================
# HISTORIAL DE SENSORES (para promedios)
# ============================================

class SensorHistory:
    """Mantiene historial de lecturas para calcular promedios"""
    
    def __init__(self, max_age_minutes: int = 30):
        self.max_age = timedelta(minutes=max_age_minutes)
        self.readings: Dict[str, deque] = {
            "temperatura": deque(maxlen=100),
            "humedad": deque(maxlen=100),
            "conductividad": deque(maxlen=100),
            "luz": deque(maxlen=100),
        }
    
    def add_reading(self, param: str, value: float, timestamp: datetime = None):
        """Agregar una lectura al historial"""
        if timestamp is None:
            timestamp = datetime.now()
        if param in self.readings:
            self.readings[param].append((timestamp, value))
    
    def add_all_readings(self, sensor_data: dict):
        """Agregar todas las lecturas de un sensor"""
        now = datetime.now()
        if "temperature" in sensor_data:
            self.add_reading("temperatura", sensor_data["temperature"], now)
        if "moisture" in sensor_data:
            self.add_reading("humedad", sensor_data["moisture"], now)
        if "conductivity" in sensor_data:
            self.add_reading("conductividad", sensor_data["conductivity"], now)
        if "light" in sensor_data:
            self.add_reading("luz", sensor_data["light"], now)
    
    def get_average(self, param: str, minutes: int = 15) -> Optional[float]:
        """Obtener promedio de los últimos X minutos"""
        if param not in self.readings or not self.readings[param]:
            return None
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        values = [v for t, v in self.readings[param] if t >= cutoff]
        
        if not values:
            return None
        
        return sum(values) / len(values)
    
    def get_latest(self, param: str) -> Optional[float]:
        """Obtener la última lectura"""
        if param not in self.readings or not self.readings[param]:
            return None
        return self.readings[param][-1][1]
    
    def cleanup_old(self):
        """Eliminar lecturas antiguas"""
        cutoff = datetime.now() - self.max_age
        for param in self.readings:
            while self.readings[param] and self.readings[param][0][0] < cutoff:
                self.readings[param].popleft()


# Instancia global del historial
sensor_history = SensorHistory(max_age_minutes=30)


# ============================================
# CLASE PRINCIPAL DEL CONTROLADOR
# ============================================

class MotorController:
    """
    Controlador de motores NEMA 17 para Raspberry Pi 5.
    Incluye verificación ambiental basada en literatura científica.
    """
    
    def __init__(self):
        self.handle = None
        self.is_active = False
        self.is_moving = False
        self.total_activations = 0
        self.last_activation = None
        self.posicion_horizontal_mm = 0.0
        self.posicion_vertical_mm = 0.0
        self.hardware_connected = False
        self.stop_requested = False
        
        # Historial de verificaciones
        self.ultima_verificacion = None
        self.verificacion_cache = None
        
        self._init_gpio()
    
    def _init_gpio(self):
        """Inicializar GPIO con lgpio"""
        try:
            import lgpio
            self.lgpio = lgpio
            
            self.handle = lgpio.gpiochip_open(4)
            
            # Configurar pines como salida
            for pin in [PIN_PUL_H, PIN_DIR_H, PIN_PUL_V, PIN_DIR_V, PIN_SIGNAL]:
                lgpio.gpio_claim_output(self.handle, pin)
                lgpio.gpio_write(self.handle, pin, 0)
            
            self.hardware_connected = True
            logger.info("✅ GPIO configurado correctamente (lgpio - Pi 5)")
            logger.info(f"   Horizontal: PUL={PIN_PUL_H}, DIR={PIN_DIR_H}")
            logger.info(f"   Vertical: PUL={PIN_PUL_V}, DIR={PIN_DIR_V}")
            logger.info(f"   Señal: GPIO{PIN_SIGNAL}")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando GPIO: {e}")
            self.hardware_connected = False
            raise RuntimeError(f"No se pudo inicializar GPIO: {e}")
    
    # ============================================
    # VERIFICACIÓN AMBIENTAL
    # ============================================
    
    def verificar_condiciones_ambientales(
        self, 
        sensor_data: List[dict],
        usar_promedios: bool = True
    ) -> dict:
        """
        Verifica si las condiciones ambientales permiten polinización.
        
        Args:
            sensor_data: Lista de diccionarios con datos de sensores
            usar_promedios: Si True, usa promedios de 15 min (recomendado)
        
        Returns:
            {
                "puede_polinizar": bool,
                "nivel_confianza": str,  # "optimo", "aceptable", "marginal", "no_recomendado"
                "condiciones": {...},
                "problemas": [...],
                "advertencias": [...],
                "recomendaciones": [...]
            }
        """
        
        # Actualizar historial con nuevos datos
        for sensor in sensor_data:
            sensor_history.add_all_readings(sensor)
        
        # Calcular valores a usar
        if usar_promedios:
            # Promedios de 15 minutos (excepto luz que es instantáneo)
            temp = sensor_history.get_average("temperatura", VENTANA_PROMEDIO_MINUTOS)
            humedad = sensor_history.get_average("humedad", VENTANA_PROMEDIO_MINUTOS)
            conductividad = sensor_history.get_average("conductividad", VENTANA_PROMEDIO_MINUTOS)
            luz = sensor_history.get_latest("luz")  # Luz = valor instantáneo
        else:
            # Valores instantáneos (promedio de sensores actuales)
            temp = self._promedio_sensores(sensor_data, "temperature")
            humedad = self._promedio_sensores(sensor_data, "moisture")
            conductividad = self._promedio_sensores(sensor_data, "conductivity")
            luz = self._promedio_sensores(sensor_data, "light")
        
        # Si no hay datos suficientes, usar instantáneos
        if temp is None:
            temp = self._promedio_sensores(sensor_data, "temperature")
        if humedad is None:
            humedad = self._promedio_sensores(sensor_data, "moisture")
        if conductividad is None:
            conductividad = self._promedio_sensores(sensor_data, "conductivity")
        if luz is None:
            luz = self._promedio_sensores(sensor_data, "light")
        
        # Inicializar resultado
        resultado = {
            "puede_polinizar": True,
            "nivel_confianza": "optimo",
            "condiciones": {},
            "problemas": [],
            "advertencias": [],
            "recomendaciones": [],
            "timestamp": datetime.now().isoformat(),
            "metodo": "promedios_15min" if usar_promedios else "instantaneo"
        }
        
        # ========== VERIFICAR TEMPERATURA ==========
        cond_temp = CONDICIONES_OPTIMAS["temperatura"]
        if temp is not None:
            temp_status = self._evaluar_parametro(
                temp, 
                cond_temp["min"], 
                cond_temp["max"],
                cond_temp["optimo_min"],
                cond_temp["optimo_max"]
            )
            
            resultado["condiciones"]["temperatura"] = {
                "valor": round(temp, 1),
                "unidad": "°C",
                "estado": temp_status,
                "rango_permitido": f"{cond_temp['min']}-{cond_temp['max']}°C",
                "rango_optimo": f"{cond_temp['optimo_min']}-{cond_temp['optimo_max']}°C"
            }
            
            # Bloqueo crítico por temperatura alta
            if temp >= cond_temp["severo"]:
                resultado["puede_polinizar"] = False
                resultado["nivel_confianza"] = "critico"
                resultado["problemas"].append(
                    f" ⚠️ TEMPERATURA CRÍTICA: {temp:.1f}°C (≥{cond_temp['severo']}°C). "
                    f"No se recomienda la polinización, temperatura mayor a 35°C."
                )
            elif temp >= cond_temp["critico"]:
                resultado["puede_polinizar"] = False
                resultado["nivel_confianza"] = "no_recomendado"
                resultado["problemas"].append(
                    f"⚠️ TEMPERATURA ALTA: {temp:.1f}°C (≥{cond_temp['critico']}°C). "
                    f"Se recomienda esperar a que baje la temperatura"
                )
            elif temp < cond_temp["min"]:
                resultado["puede_polinizar"] = False
                resultado["nivel_confianza"] = "no_recomendado"
                resultado["problemas"].append(
                    f"❄️ TEMPERATURA BAJA: {temp:.1f}°C (<{cond_temp['min']}°C). "
                    f"Polen con baja actividad."
                )
            elif temp > cond_temp["max"]:
                resultado["nivel_confianza"] = "marginal"
                resultado["advertencias"].append(
                    f" ⚠️ Temperatura elevada: {temp:.1f}°C. "
                    f"Temperatura en su rango máximo recomendable."
                )
        else:
            resultado["condiciones"]["temperatura"] = {"valor": None, "estado": "sin_datos"}
            resultado["advertencias"].append("Sin datos de temperatura")
        
        # ========== VERIFICAR HUMEDAD DE SUELO ==========
        cond_hum = CONDICIONES_OPTIMAS["humedad"]
        if humedad is not None:
            hum_status = self._evaluar_parametro(
                humedad,
                cond_hum["min"],
                cond_hum["max"],
                cond_hum["optimo_min"],
                cond_hum["optimo_max"]
            )
            
            resultado["condiciones"]["humedad"] = {
                "valor": round(humedad, 1),
                "unidad": "%",
                "estado": hum_status,
                "rango_permitido": f"{cond_hum['min']}-{cond_hum['max']}%",
                "rango_optimo": f"{cond_hum['optimo_min']}-{cond_hum['optimo_max']}%"
            }
            
            if humedad < cond_hum["min"]:
                resultado["nivel_confianza"] = min(resultado["nivel_confianza"], "marginal")
                resultado["advertencias"].append(
                    f" ⚠️ HUMEDAD BAJA: {humedad:.1f}% (<{cond_hum['min']}%). "
                    f"Se recomienda regar las plantas, rango mínimo aceptable de humedad."
                )
                resultado["recomendaciones"].append("Regar las plantas antes de polinizar")
            elif humedad > cond_hum["max"]:
                resultado["nivel_confianza"] = min(resultado["nivel_confianza"], "marginal")
                resultado["advertencias"].append(
                    f" ⚠️ HUMEDAD ALTA: {humedad:.1f}% (>{cond_hum['max']}%). "
                    f"Se recomienda drenar el sustrato, rango de humedad máximo detectado."
                )
        else:
            resultado["condiciones"]["humedad"] = {"valor": None, "estado": "sin_datos"}
        
        # ========== VERIFICAR LUZ ==========
        cond_luz = CONDICIONES_OPTIMAS["luz"]
        if luz is not None:
            luz_status = self._evaluar_parametro(
                luz,
                cond_luz["min"],
                cond_luz["max"],
                cond_luz["optimo_min"],
                cond_luz["optimo_max"]
            )
            
            resultado["condiciones"]["luz"] = {
                "valor": int(luz),
                "unidad": "lux",
                "estado": luz_status,
                "rango_permitido": f"{cond_luz['min']:,}-{cond_luz['max']:,} lux",
                "rango_optimo": f"{cond_luz['optimo_min']:,}-{cond_luz['optimo_max']:,} lux"
            }
            
            if luz < cond_luz["min"]:
                resultado["puede_polinizar"] = False
                resultado["nivel_confianza"] = "no_recomendado"
                resultado["problemas"].append(
                    f" ⚠️ LUZ INSUFICIENTE: {luz:,} lux (<{cond_luz['min']:,}). "
                    f"Se recomienda colocar a la planta en un lugar con una mayor cantidad de luz."
                )
        else:
            resultado["condiciones"]["luz"] = {"valor": None, "estado": "sin_datos"}
        
        # ========== VERIFICAR CONDUCTIVIDAD ==========
        cond_ec = CONDICIONES_OPTIMAS["conductividad"]
        if conductividad is not None:
            ec_status = self._evaluar_parametro(
                conductividad,
                cond_ec["min"],
                cond_ec["max"],
                cond_ec["optimo_min"],
                cond_ec["optimo_max"]
            )
            
            resultado["condiciones"]["conductividad"] = {
                "valor": int(conductividad),
                "unidad": "µS/cm",
                "estado": ec_status,
                "rango_permitido": f"{cond_ec['min']}-{cond_ec['max']} µS/cm",
                "rango_optimo": f"{cond_ec['optimo_min']}-{cond_ec['optimo_max']} µS/cm"
            }
            
            if conductividad < cond_ec["min"]:
                resultado["advertencias"].append(
                    f"⚡ CONDUCTIVIDAD BAJA: {conductividad} µS/cm. "
                    f"Se recomienda fertilizar el sustrato."
                )
                resultado["recomendaciones"].append("Revisar fertilización")
            elif conductividad > cond_ec["max"]:
                resultado["advertencias"].append(
                    f"⚡ CONDUCTIVIDAD ALTA: {conductividad} µS/cm. "
                    f"Se recomienda lavar el sustrato si es posible."
                )
        else:
            resultado["condiciones"]["conductividad"] = {"valor": None, "estado": "sin_datos"}
        
        # ========== DETERMINAR NIVEL DE CONFIANZA FINAL ==========
        if resultado["puede_polinizar"]:
            estados = [c.get("estado") for c in resultado["condiciones"].values() if c.get("estado")]
            
            if all(e == "optimo" for e in estados):
                resultado["nivel_confianza"] = "optimo"
                resultado["recomendaciones"].append(
                    " ✅ Condiciones recomendadas para polinización"
                )
            elif all(e in ["optimo", "aceptable"] for e in estados):
                resultado["nivel_confianza"] = "aceptable"
                resultado["recomendaciones"].append(
                    " ✅ Condiciones aceptables para polinización"
                )
            elif "fuera_rango" not in estados:
                resultado["nivel_confianza"] = "marginal"
                resultado["recomendaciones"].append(
                    "⚠️ Condiciones no recomendables para polinización"
                )
        
        # Cachear resultado
        self.ultima_verificacion = datetime.now()
        self.verificacion_cache = resultado
        
        return resultado
    
    def _promedio_sensores(self, sensor_data: List[dict], key: str) -> Optional[float]:
        """Calcular promedio de un parámetro de múltiples sensores"""
        values = [s[key] for s in sensor_data if key in s and s[key] is not None]
        if not values:
            return None
        return sum(values) / len(values)
    
    def _evaluar_parametro(
        self, 
        valor: float, 
        min_permitido: float, 
        max_permitido: float,
        optimo_min: float,
        optimo_max: float
    ) -> str:
        """Evaluar estado de un parámetro"""
        if valor < min_permitido or valor > max_permitido:
            return "fuera_rango"
        elif optimo_min <= valor <= optimo_max:
            return "optimo"
        else:
            return "aceptable"
    
    def get_condiciones_optimas(self) -> dict:
        """Obtener los rangos de condiciones óptimas configurados"""
        return CONDICIONES_OPTIMAS.copy()
    
    # ============================================
    # CONTROL DE MOTORES
    # ============================================
    
    def _mm_a_pasos(self, mm: float, es_horizontal: bool = True) -> int:
        """Convertir mm a pasos del motor"""
        mm_por_rev = MM_POR_REV_H if es_horizontal else MM_POR_REV_V
        pasos = int((mm / mm_por_rev) * PASOS_POR_REV * MICROSTEPPING)
        return pasos
    
    def _mover_motor(
        self, 
        pin_pul: int, 
        pin_dir: int, 
        pasos: int, 
        direccion: int = 1,
        delay_us: int = 1000
    ):
        """Mover un motor una cantidad de pasos"""
        if not self.hardware_connected or self.handle is None:
            logger.warning("Hardware no conectado, simulando movimiento")
            time.sleep(pasos * delay_us / 1_000_000)
            return
        
        self.lgpio.gpio_write(self.handle, pin_dir, direccion)
        time.sleep(0.001)
        
        for _ in range(pasos):
            if self.stop_requested:
                logger.warning("⚠️ Stop solicitado, deteniendo motor")
                break
            
            self.lgpio.gpio_write(self.handle, pin_pul, 1)
            time.sleep(delay_us / 1_000_000)
            self.lgpio.gpio_write(self.handle, pin_pul, 0)
            time.sleep(delay_us / 1_000_000)
    
    def mover_horizontal(self, distancia_mm: float, direccion: int = 1):
        """Mover motor horizontal"""
        self.is_moving = True
        pasos = self._mm_a_pasos(distancia_mm, es_horizontal=True)
        
        logger.info(f"🔄 Moviendo horizontal: {distancia_mm}mm = {pasos} pasos, dir={direccion}")
        
        self._mover_motor(PIN_PUL_H, PIN_DIR_H, pasos, direccion, DELAY_H)
        
        if direccion == 1:
            self.posicion_horizontal_mm += distancia_mm
        else:
            self.posicion_horizontal_mm -= distancia_mm
        
        self.is_moving = False
        logger.info(f"✅ Posición horizontal: {self.posicion_horizontal_mm}mm")
    
    def mover_vertical(self, distancia_mm: float, direccion: int = 1):
        """Mover motor vertical"""
        self.is_moving = True
        pasos = self._mm_a_pasos(distancia_mm, es_horizontal=False)
        
        logger.info(f"🔄 Moviendo vertical: {distancia_mm}mm = {pasos} pasos, dir={direccion}")
        
        self._mover_motor(PIN_PUL_V, PIN_DIR_V, pasos, direccion, DELAY_V)
        
        if direccion == 1:
            self.posicion_vertical_mm += distancia_mm
        else:
            self.posicion_vertical_mm -= distancia_mm
        
        self.is_moving = False
        logger.info(f"✅ Posición vertical: {self.posicion_vertical_mm}mm")
    
    def mover_a_planta(self, numero_planta: int):
        """Mover a la posición de una planta específica"""
        if numero_planta < 1 or numero_planta > 4:
            raise ValueError(f"Número de planta inválido: {numero_planta}")
        
        # Calcular distancia total desde home
        distancia_total = sum(DISTANCIAS_PLANTAS[i] for i in range(1, numero_planta + 1))
        
        # Calcular movimiento necesario desde posición actual
        distancia_a_mover = distancia_total - self.posicion_horizontal_mm
        
        logger.info(f"🌱 Moviendo a planta {numero_planta}: {distancia_total}mm (delta: {distancia_a_mover}mm)")
        
        if distancia_a_mover > 0:
            self.mover_horizontal(distancia_a_mover, 1)
        elif distancia_a_mover < 0:
            self.mover_horizontal(abs(distancia_a_mover), 0)
    
    def acercar_vertical(self):
        """Acercar a la planta (bajar)"""
        logger.info(f"⬇️ Bajando vertical: {DISTANCIA_VERTICAL}mm")
        self.mover_vertical(DISTANCIA_VERTICAL, 0)
        self.posicion_vertical_mm = DISTANCIA_VERTICAL  # ← Forzar positivo

    def alejar_vertical(self):
        """Alejar de la planta (subir)"""
        logger.info(f"⬆️ Subiendo vertical: {self.posicion_vertical_mm}mm")
        if self.posicion_vertical_mm > 0:
            self.mover_vertical(self.posicion_vertical_mm, 1)
            self.posicion_vertical_mm = 0.0  # ← Resetear a 0
    
    def activar_senal_externa(self, duracion: float = DURACION_VIBRACION):
        """Activar señal GPIO24 (relay/vibrador)"""
        if not self.hardware_connected or self.handle is None:
            logger.info(f"📳 [SIMULADO] Señal externa por {duracion}s")
            time.sleep(duracion)
            return
        
        logger.info(f"📳 Activando señal GPIO{PIN_SIGNAL} por {duracion}s")
        self.lgpio.gpio_write(self.handle, PIN_SIGNAL, 1)
        time.sleep(duracion)
        self.lgpio.gpio_write(self.handle, PIN_SIGNAL, 0)
        logger.info("📳 Señal desactivada")
    
    def ir_a_home(self):
        """Regresar a posición inicial (0, 0)"""
        logger.info("🏠 Regresando a home...")
        
        # Primero subir (alejar vertical)
        if self.posicion_vertical_mm > 0:
            self.alejar_vertical()
        
        # Luego regresar horizontal
        if self.posicion_horizontal_mm > 0:
            self.mover_horizontal(self.posicion_horizontal_mm, 0)
        
        self.posicion_horizontal_mm = 0.0
        self.posicion_vertical_mm = 0.0
        
        logger.info("🏠 En posición home (0, 0)")
        
        return {
            "status": "success",
            "posicion_horizontal_mm": 0.0,
            "posicion_vertical_mm": 0.0
        }
    
    # ============================================
    # POLINIZACIÓN
    # ============================================
    
    def polinizar_planta(
        self, 
        numero_planta: int, 
        duracion_vibracion: float = DURACION_VIBRACION
    ) -> dict:
        """
        Secuencia completa de polinización para una planta.
        
        Secuencia:
        1. Mover horizontal a la planta
        2. Esperar estabilización (2s)
        3. BAJAR (dirección 0)
        4. Esperar estabilización vertical (2s)
        5. Vibrar (5s)
        6. Esperar antes de subir (2s)
        7. SUBIR (dirección 1)
        8. Siguiente planta o Home
        """
        if self.is_active:
            return {"status": "busy", "message": "Motor ocupado"}
        
        self.is_active = True
        self.stop_requested = False
        
        try:
            logger.info(f"🌸 Iniciando polinización de planta {numero_planta}")
            
            # 1. Mover horizontal a la planta ← RESTAURADO
            self.mover_a_planta(numero_planta)
            
            if self.stop_requested:
                raise InterruptedError("Polinización cancelada")
            
            # 2. Esperar estabilización horizontal ← RESTAURADO
            logger.info(f"⏳ Esperando estabilización horizontal ({DELAY_ESTABILIZACION}s)")
            time.sleep(DELAY_ESTABILIZACION)
            
            # 3. Bajar (acercar)
            self.acercar_vertical()

            if self.stop_requested:
                raise InterruptedError("Polinización cancelada")

            # 4. Esperar estabilización vertical
            logger.info(f"⏳ Esperando estabilización vertical ({DELAY_VERTICAL}s)")
            time.sleep(DELAY_VERTICAL)

            # 5. Vibrar
            self.activar_senal_externa(duracion_vibracion)

            # 6. Esperar antes de subir
            logger.info(f"⏳ Esperando antes de subir ({DELAY_VERTICAL}s)")
            time.sleep(DELAY_VERTICAL)

            # 7. Subir (alejar)
            self.alejar_vertical()
                
            self.total_activations += 1
            self.last_activation = datetime.now()
            
            logger.info(f"✅ Polinización de planta {numero_planta} completada")
            
            return {
                "status": "success",
                "planta": numero_planta,
                "timestamp": self.last_activation.isoformat(),
                "mode": "hardware" if self.hardware_connected else "mock",
                "duracion_vibracion": duracion_vibracion
            }
            
        except InterruptedError as e:
            logger.warning(f"⚠️ {e}")
            return {"status": "cancelled", "message": str(e)}
        except Exception as e:
            logger.error(f"❌ Error en polinización: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.is_active = False
    
    def polinizar_selectivo(
        self, 
        plantas: List[int],
        sensor_data: List[dict] = None,
        ignorar_condiciones: bool = False,
        duracion_vibracion: float = DURACION_VIBRACION
    ) -> dict:
        """
        Polinizar múltiples plantas en secuencia.
        """
        if self.is_active:
            return {"status": "busy", "message": "Motor ocupado"}
        
        # Verificar condiciones ambientales si hay datos de sensores
        verificacion = None
        if sensor_data and not ignorar_condiciones:
            verificacion = self.verificar_condiciones_ambientales(sensor_data)
            
            if not verificacion["puede_polinizar"]:
                return {
                    "status": "condiciones_no_optimas",
                    "message": "Condiciones ambientales no adecuadas",
                    "verificacion": verificacion,
                    "plantas_solicitadas": plantas
                }
        
        self.is_active = True
        self.stop_requested = False
        plantas_completadas = []
        plantas_fallidas = []
        
        try:
            logger.info(f"🌸 Iniciando polinización selectiva: plantas {plantas}")
            
            # Ordenar plantas para optimizar movimiento
            plantas_ordenadas = sorted(plantas)
            
            for planta in plantas_ordenadas:
                if self.stop_requested:
                    logger.warning("⚠️ Stop solicitado, deteniendo secuencia")
                    break
                
                # Liberamos momentáneamente el flag is_active para que polinizar_planta pueda entrar
                self.is_active = False 
                resultado = self.polinizar_planta(planta, duracion_vibracion)
                self.is_active = True  # Bloqueamos de nuevo
                # -----------------------
                
                if resultado["status"] == "success":
                    plantas_completadas.append(planta)
                else:
                    plantas_fallidas.append({"planta": planta, "error": resultado})
            
            # Regresar a home
            if not self.stop_requested:
                self.ir_a_home()
            
            return {
                "status": "success" if not plantas_fallidas else "partial",
                "plantas_completadas": plantas_completadas,
                "plantas_fallidas": plantas_fallidas,
                "total_polinizadas": len(plantas_completadas),
                "verificacion_ambiental": verificacion,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en polinización selectiva: {e}")
            return {
                "status": "error",
                "message": str(e),
                "plantas_completadas": plantas_completadas
            }
        finally:
            self.is_active = False
    
    # ============================================
    # MÉTODOS DE CONTROL Y ESTADO
    # ============================================
    
    def stop(self):
        """Detener motor inmediatamente"""
        self.stop_requested = True
        self.is_active = False
        self.is_moving = False
        
        # Desactivar todas las salidas
        if self.hardware_connected and self.handle:
            for pin in [PIN_PUL_H, PIN_DIR_H, PIN_PUL_V, PIN_DIR_V, PIN_SIGNAL]:
                self.lgpio.gpio_write(self.handle, pin, 0)
        
        logger.warning("🛑 MOTOR DETENIDO")
        return {"status": "stopped", "timestamp": datetime.now().isoformat()}
    
    def activate(self, duration: float = 2.0, steps: int = 200) -> dict:
        """Método de compatibilidad con API anterior"""
        return self.polinizar_planta(1, duration)
    
    def test_senal_externa(self, duracion: float = 2.0) -> dict:
        """Test de señal externa sin mover motores"""
        if self.is_active:
            return {"status": "busy", "message": "Motor ocupado"}
        
        self.is_active = True
        try:
            self.activar_senal_externa(duracion)
            return {
                "status": "success",
                "message": f"Señal GPIO{PIN_SIGNAL} activada por {duracion}s",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.is_active = False
    
    def reset_completo(self) -> dict:
        """Reset completo del sistema"""
        self.stop()
        self.posicion_horizontal_mm = 0.0
        self.posicion_vertical_mm = 0.0
        self.total_activations = 0
        self.last_activation = None
        self.stop_requested = False
        
        logger.info("🔄 Sistema reseteado completamente")
        return {
            "status": "success",
            "message": "Sistema reseteado",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> dict:
        """Obtener estado actual del controlador"""
        return {
            "is_active": self.is_active,
            "is_moving": self.is_moving,
            "hardware_connected": self.hardware_connected,
            "gpio_chip_open": self.handle is not None,
            "posicion_horizontal_mm": self.posicion_horizontal_mm,
            "posicion_vertical_mm": self.posicion_vertical_mm,
            "total_activations": self.total_activations,
            "last_activation": self.last_activation.isoformat() if self.last_activation else None,
            "signal_pin": f"GPIO{PIN_SIGNAL}",
            "condiciones_optimas": CONDICIONES_OPTIMAS,
            "ultima_verificacion": self.ultima_verificacion.isoformat() if self.ultima_verificacion else None
        }
    
    def cleanup(self):
        """Liberar recursos GPIO"""
        if self.handle is not None:
            try:
                for pin in [PIN_PUL_H, PIN_DIR_H, PIN_PUL_V, PIN_DIR_V, PIN_SIGNAL]:
                    self.lgpio.gpio_write(self.handle, pin, 0)
                self.lgpio.gpiochip_close(self.handle)
                logger.info("✅ GPIO liberado correctamente")
            except:
                pass
            self.handle = None
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

