"""
FastAPI Backend - Sistema de Monitoreo de Tomates
Integra: Sensores MiFlora, YOLO Detection, Motor NEMA 17
Con verificación de condiciones ambientales

CARACTERÍSTICAS:
- Grid 2x2 para 4 plantas
- Múltiples fotos por planta
- Análisis YOLO agregado
- Polinización selectiva
- Verificación de condiciones ambientales (sensores MiFlora)
- Señal externa GPIO24
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os
import json
from datetime import datetime, timedelta
import shutil
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos propios
from data_reader import SensorDataReader
from yolo_processor import YOLOProcessor
from motor_singleton import motor_manager

# Inicializar FastAPI
app = FastAPI(
    title=" Monitor de Tomates API",
    description="Sistema integrado con verificación ambiental y polinización selectiva",
    version="3.1.0"
)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="frontend"), name="static")
# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse

class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request, call_next):
        if request.method == 'POST' and '/api/' in str(request.url):
            if 'content-length' in request.headers:
                content_length = int(request.headers['content-length'])
                if content_length > self.max_upload_size:
                    return StarletteJSONResponse(
                        status_code=413,
                        content={
                            'detail': f'Archivo demasiado grande. Máximo: {self.max_upload_size/1024/1024:.1f}MB'
                        }
                    )
        return await call_next(request)

# Aplicar middleware - 10 MB de límite
app.add_middleware(LimitUploadSize, max_upload_size=10 * 1024 * 1024) 

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
CSV_PATH = "/home/arthur/yolo_env/miflora_log.csv"
MODEL_PATH = "models/best.pt"
OUTPUT_DIR = "outputs"
DETECTION_LOG = "data/detections_log.json"
PLANTS_DATA_DIR = "data/plants"

# ==========================================
# INICIALIZACIÓN
# ==========================================
logger.info("=" * 60)
logger.info("🍅 INICIALIZANDO SISTEMA v3.1")
logger.info("=" * 60)

sensor_reader = SensorDataReader(CSV_PATH)
yolo_processor = YOLOProcessor(MODEL_PATH)

# Obtener estado inicial del motor
try:
    motor_status = motor_manager.get_status()
    logger.info(f"✅ Motor HARDWARE inicializado")
    logger.info(f"📍 GPIO conectado: {motor_status.get('hardware_connected', False)}")
    logger.info(f"📍 Pin señal: {motor_status.get('signal_pin', 'GPIO24')}")
except Exception as e:
    logger.warning(f"⚠️ Motor no disponible, funcionando en modo simulación: {e}")
    motor_status = {"mode": "simulation", "hardware_connected": False}

logger.info("=" * 60)

# Crear directorios necesarios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("frontend", exist_ok=True)
for i in range(1, 5):
    os.makedirs(f"{PLANTS_DATA_DIR}/planta_{i}/images", exist_ok=True)

# ==========================================
# ALMACENAMIENTO EN MEMORIA PARA PLANTAS
# ==========================================
plants_data: Dict[int, dict] = {
    1: {"images": [], "analysis": None, "ready": False, "last_update": None},
    2: {"images": [], "analysis": None, "ready": False, "last_update": None},
    3: {"images": [], "analysis": None, "ready": False, "last_update": None},
    4: {"images": [], "analysis": None, "ready": False, "last_update": None},
}

# ==========================================
# MODELOS PYDANTIC
# ==========================================

class SensorData(BaseModel):
    timestamp: str
    sensor_id: str
    temperature: float
    moisture: float
    light: int
    conductivity: int
    battery: int

class MotorCommand(BaseModel):
    duration: float = 2.0
    steps: int = 200

class PollinateRequest(BaseModel):
    plantas: List[int]
    ignorar_condiciones: bool = False

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def load_detection_log():
    """Cargar historial de detecciones"""
    if os.path.exists(DETECTION_LOG):
        try:
            with open(DETECTION_LOG, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_detection_log(detection_data):
    """Guardar detección en el log"""
    log = load_detection_log()
    log.append(detection_data)
    if len(log) > 100:
        log = log[-100:]
    with open(DETECTION_LOG, 'w') as f:
        json.dump(log, f, indent=2)

def check_plant_ready(analysis_result):
    """Verificar si una planta está lista para polinizar"""
    if not analysis_result:
        return False
    flowers = analysis_result.get('flowers', {})
    return flowers.get('Lista_para_polinizar', 0) > 0

def aggregate_plant_analysis(plant_id):
    """Agregar análisis de todas las imágenes de una planta"""
    plant = plants_data.get(plant_id)
    if not plant or not plant["images"]:
        return None
    
    total_flowers = {"Lista_para_polinizar": 0, "No_desarrollada": 0, "Sin_polen": 0}
    total_fruits = {"verde": 0, "mixto": 0, "rojo": 0}
    total_objects = 0
    
    for img_data in plant["images"]:
        if img_data.get("analysis"):
            analysis = img_data["analysis"]
            total_objects += analysis.get("total_objects", 0)
            for key in total_flowers:
                total_flowers[key] += analysis.get("flowers", {}).get(key, 0)
            for key in total_fruits:
                total_fruits[key] += analysis.get("fruits", {}).get(key, 0)
    
    return {
        "total_objects": total_objects,
        "flowers": total_flowers,
        "fruits": total_fruits,
        "images_analyzed": len(plant["images"]),
        "ready_to_pollinate": total_flowers["Lista_para_polinizar"] > 0
    }

def get_sensor_data_for_verification(hours: float = 2.0):
    """
    Obtener datos de sensores para verificación ambiental.
    Calcula el promedio de las últimas X horas.
    
    Args:
        hours: Número de horas hacia atrás para promediar (default: 2 horas)
    
    Returns:
        Lista de diccionarios con datos de sensores o None si no hay datos
    """
    try:
        # Obtener todos los datos recientes
        all_data = sensor_reader.get_latest_data()
        
        if all_data.empty:
            logger.warning("No hay datos de sensores disponibles")
            return None
        
        # Convertir a lista de diccionarios
        sensor_list = []
        for _, row in all_data.iterrows():
            sensor_list.append({
                'temperature': float(row.get('temperature', 0)),
                'moisture': float(row.get('moisture', 0)),
                'conductivity': int(row.get('conductivity', 0)),
                'light': int(row.get('light', 0)),
                'battery': int(row.get('battery', 0)),
                'sensor_id': row.get('sensor_id', 'unknown')
            })
        
        if not sensor_list:
            return None
            
        logger.info(f"📊 Datos de {len(sensor_list)} sensores obtenidos para verificación")
        return sensor_list
        
    except Exception as e:
        logger.error(f"Error obteniendo datos de sensores: {e}")
        return None

# ==========================================
# ENDPOINTS PRINCIPALES
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Servir el frontend"""
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return f.read()
    return """
    <html>
        <head><title>Monitor Tomates v3.1</title></head>
        <body style="font-family: Arial; text-align: center; padding: 50px; background: #e8f5e9;">
            <h1>🍅 Monitor de Tomates v3.1</h1>
            <p>Frontend no encontrado. Crea <code>frontend/index.html</code></p>
            <a href="/docs" style="color: #4caf50;">📊 Ver API Docs</a>
        </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Verificar estado del sistema"""
    motor_status = motor_manager.get_status()
    
    # Verificar sensores
    sensor_data = get_sensor_data_for_verification()
    sensors_ok = sensor_data is not None and len(sensor_data) > 0
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.1.0",
        "components": {
            "sensors": os.path.exists(CSV_PATH),
            "sensors_data_available": sensors_ok,
            "sensors_count": len(sensor_data) if sensor_data else 0,
            "yolo_model": os.path.exists(MODEL_PATH),
            "motor": motor_status.get("hardware_connected", False),
            "motor_mode": "hardware",
            "gpio_connected": motor_status.get("gpio_chip_open", False),
            "signal_pin": motor_status.get("signal_pin", "GPIO24")
        },
        "condiciones_optimas": motor_status.get("condiciones_optimas", {})
    }

# ==========================================
# ENDPOINTS DE PLANTAS
# ==========================================

@app.get("/api/plants")
async def get_all_plants():
    """Obtener estado de todas las plantas"""
    result = {}
    for plant_id in range(1, 5):
        plant = plants_data[plant_id]
        result[plant_id] = {
            "plant_id": plant_id,
            "images_count": len(plant["images"]),
            "analyzed": plant["analysis"] is not None,
            "ready": plant["ready"],
            "last_update": plant["last_update"],
            "analysis_summary": plant["analysis"]
        }
    return result

@app.get("/api/plants/{plant_id}")
async def get_plant(plant_id: int):
    """Obtener información de una planta específica"""
    if plant_id < 1 or plant_id > 4:
        raise HTTPException(status_code=400, detail="plant_id debe ser 1-4")
    
    plant = plants_data[plant_id]
    return {
        "plant_id": plant_id,
        "images": [
            {
                "filename": img["filename"],
                "timestamp": img["timestamp"],
                "analyzed": img.get("analysis") is not None
            }
            for img in plant["images"]
        ],
        "images_count": len(plant["images"]),
        "analyzed": plant["analysis"] is not None,
        "ready": plant["ready"],
        "last_update": plant["last_update"],
        "analysis": plant["analysis"]
    }

@app.post("/api/plants/{plant_id}/images")
async def upload_plant_image(plant_id: int, file: UploadFile = File(...)):
    """Subir una imagen para una planta específica"""
    if plant_id < 1 or plant_id > 4:
        raise HTTPException(status_code=400, detail="plant_id debe ser 1-4")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"planta_{plant_id}_{timestamp}_{file.filename}"
        filepath = f"{PLANTS_DATA_DIR}/planta_{plant_id}/images/{filename}"
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        plants_data[plant_id]["images"].append({
            "filename": filename,
            "filepath": filepath,
            "timestamp": datetime.now().isoformat(),
            "analysis": None
        })
        plants_data[plant_id]["last_update"] = datetime.now().isoformat()
        plants_data[plant_id]["analysis"] = None
        plants_data[plant_id]["ready"] = False
        
        return {
            "status": "success",
            "plant_id": plant_id,
            "filename": filename,
            "images_count": len(plants_data[plant_id]["images"]),
            "message": f"Imagen subida para planta {plant_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/plants/{plant_id}/images")
async def clear_plant_images(plant_id: int):
    """Limpiar todas las imágenes de una planta"""
    if plant_id < 1 or plant_id > 4:
        raise HTTPException(status_code=400, detail="plant_id debe ser 1-4")
    
    try:
        for img in plants_data[plant_id]["images"]:
            filepath = img.get("filepath")
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        
        plants_data[plant_id] = {
            "images": [],
            "analysis": None,
            "ready": False,
            "last_update": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "plant_id": plant_id,
            "message": f"Imágenes de planta {plant_id} eliminadas"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plants/{plant_id}/images/{filename}")
async def get_plant_image(plant_id: int, filename: str):
    """Obtener una imagen de una planta"""
    filepath = f"{PLANTS_DATA_DIR}/planta_{plant_id}/images/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(filepath)

# ==========================================
# ANÁLISIS DE PLANTAS
# ==========================================

@app.post("/api/plants/{plant_id}/analyze")
async def analyze_plant(plant_id: int):
    """Analizar todas las imágenes de una planta"""
    if plant_id < 1 or plant_id > 4:
        raise HTTPException(status_code=400, detail="plant_id debe ser 1-4")
    
    plant = plants_data[plant_id]
    
    if not plant["images"]:
        raise HTTPException(status_code=400, detail=f"Planta {plant_id} no tiene imágenes")
    
    try:
        for img_data in plant["images"]:
            if img_data.get("analysis"):
                continue
            
            filepath = img_data["filepath"]
            if not os.path.exists(filepath):
                continue
            
            output_filename = f"detected_{img_data['filename']}"
            output_path = f"{OUTPUT_DIR}/{output_filename}"
            
            detections = yolo_processor.process_image(filepath, output_path)
            img_data["analysis"] = detections
            img_data["output_path"] = output_path
        
        aggregated = aggregate_plant_analysis(plant_id)
        plant["analysis"] = aggregated
        plant["ready"] = aggregated["ready_to_pollinate"] if aggregated else False
        plant["last_update"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "plant_id": plant_id,
            "images_analyzed": len(plant["images"]),
            "analysis": aggregated,
            "ready_to_pollinate": plant["ready"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-all")
async def analyze_all_plants():
    """Analizar todas las plantas con imágenes"""
    try:
        results = {}
        ready_plants = []
        
        for plant_id in range(1, 5):
            plant = plants_data[plant_id]
            
            if not plant["images"]:
                results[plant_id] = {
                    "status": "no_images",
                    "ready": False,
                    "message": "Sin imágenes"
                }
                continue
            
            for img_data in plant["images"]:
                if img_data.get("analysis"):
                    continue
                
                filepath = img_data["filepath"]
                if not os.path.exists(filepath):
                    continue
                
                output_filename = f"detected_{img_data['filename']}"
                output_path = f"{OUTPUT_DIR}/{output_filename}"
                
                detections = yolo_processor.process_image(filepath, output_path)
                img_data["analysis"] = detections
                img_data["output_path"] = output_path
                
                # ✅ NUEVO: Guardar en el log de detecciones
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "image_name": img_data['filename'],
                    "plant_id": plant_id,
                    "input_path": filepath,
                    "output_path": output_path,
                    "total_objects": detections.get('total_objects', 0),
                    "flowers": detections.get('flowers', {}),
                    "fruits": detections.get('fruits', {}),
                    "should_activate_motor": detections.get('flowers', {}).get('Lista_para_polinizar', 0) > 0,
                    "motor_activated": False,
                    "motor_mode": "hardware"
                }
                save_detection_log(log_entry)
            
            aggregated = aggregate_plant_analysis(plant_id)
            plant["analysis"] = aggregated
            plant["ready"] = aggregated["ready_to_pollinate"] if aggregated else False
            plant["last_update"] = datetime.now().isoformat()
            
            results[plant_id] = {
                "status": "analyzed",
                "images_count": len(plant["images"]),
                "analysis": aggregated,
                "ready": plant["ready"]
            }
            
            if plant["ready"]:
                ready_plants.append(plant_id)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "plants": results,
            "ready_plants": ready_plants,
            "total_ready": len(ready_plants),
            "message": f"{len(ready_plants)} plantas listas para polinizar" if ready_plants else "No hay plantas listas"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ready-plants")
async def get_ready_plants():
    """Obtener lista de plantas listas para polinizar"""
    ready = []
    for plant_id in range(1, 5):
        if plants_data[plant_id]["ready"]:
            ready.append({
                "plant_id": plant_id,
                "analysis": plants_data[plant_id]["analysis"]
            })
    
    return {
        "ready_plants": [p["plant_id"] for p in ready],
        "count": len(ready),
        "details": ready
    }

# ==========================================
# VERIFICACIÓN DE CONDICIONES AMBIENTALES
# ==========================================

@app.get("/api/environmental-check")
async def check_environmental_conditions(usar_promedios: bool = Query(True)):
    """
    Verificar si las condiciones ambientales son adecuadas para polinizar.
    
    """
    try:
        sensor_data = get_sensor_data_for_verification()
        
        if not sensor_data:
            return {
                "status": "no_data",
                "puede_polinizar": False,
                "message": "No hay datos de sensores disponibles",
                "verificacion": None
            }
        
        verificacion = motor_manager.verificar_condiciones_ambientales(
            sensor_data, 
            usar_promedios=usar_promedios
        )
        
        return {
            "status": "success",
            "puede_polinizar": verificacion["puede_polinizar"],
            "nivel_confianza": verificacion["nivel_confianza"],
            "verificacion": verificacion,
            "sensores_usados": len(sensor_data),
            "metodo": "promedios_15min" if usar_promedios else "instantaneo",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# CAMBIO 5: Actualizar get_sensor_data_for_verification
# ============================================
#

def get_sensor_data_for_verification(hours: float = 2.0):
    """
    Obtener datos de sensores para verificación ambiental.
    También actualiza el historial interno para cálculo de promedios.
    """
    try:
        from motor_controller import sensor_history
        
        all_data = sensor_reader.get_latest_data()
        
        if all_data.empty:
            logger.warning("No hay datos de sensores disponibles")
            return None
        
        sensor_list = []
        for _, row in all_data.iterrows():
            sensor_dict = {
                'temperature': float(row.get('temperature', 0)),
                'moisture': float(row.get('moisture', 0)),
                'conductivity': int(row.get('conductivity', 0)),
                'light': int(row.get('light', 0)),
                'battery': int(row.get('battery', 0)),
                'sensor_id': row.get('sensor_id', 'unknown')
            }
            sensor_list.append(sensor_dict)
            
            # Actualizar historial para promedios
            sensor_history.add_all_readings(sensor_dict)
        
        if not sensor_list:
            return None
            
        logger.info(f"📊 Datos de {len(sensor_list)} sensores obtenidos y agregados al historial")
        return sensor_list
        
    except Exception as e:
        logger.error(f"Error obteniendo datos de sensores: {e}")
        return None



@app.get("/api/condiciones-optimas")
async def get_condiciones_optimas():
    """Obtener los rangos de condiciones óptimas configurados"""
    return {
        "condiciones": motor_manager.get_condiciones_optimas(),
        "descripcion": {
            "temperatura": "Temperatura del aire en °C",
            "humedad": "Humedad del suelo en %",
            "conductividad": "Conductividad/fertilidad del suelo en µS/cm",
            "luz": "Intensidad lumínica en lux"
        }
    }

# ==========================================
# POLINIZACIÓN SELECTIVA
# ==========================================

@app.post("/api/pollinate-ready")
async def pollinate_ready_plants(ignorar_condiciones: bool = Query(True)):
    """
    Polinizar SOLO las plantas que están listas.
    
    Query params:
        ignorar_condiciones: Si True, poliniza sin verificar condiciones ambientales
    
    Secuencia: Mover a planta → Delay 2s → Señal 5s → Siguiente → Home
    """
    try:
        ready_plants = [pid for pid in range(1, 5) if plants_data[pid]["ready"]]
        
        if not ready_plants:
            return {
                "status": "no_plants_ready",
                "message": "No hay plantas listas para polinizar. Analiza primero.",
                "plantas_analizadas": sum(1 for p in plants_data.values() if p["analysis"]),
                "plantas_listas": 0
            }
        
        # Obtener datos de sensores para verificación
        sensor_data = None
        verificacion_previa = None
        
        if not ignorar_condiciones:
            sensor_data = get_sensor_data_for_verification()
            
            if sensor_data:
                verificacion_previa = motor_manager.verificar_condiciones_ambientales(sensor_data)
                
                if not verificacion_previa["puede_polinizar"]:
                    return {
                        "status": "condiciones_no_optimas",
                        "message": "Condiciones ambientales no adecuadas para polinización",
                        "plantas_listas": ready_plants,
                        "verificacion": verificacion_previa,
                        "sugerencia": "Usa ignorar_condiciones=true para forzar la polinización"
                    }
        
        logger.info(f"🌸 Iniciando polinización: plantas {ready_plants}")
        
        # Ejecutar polinización selectiva
        result = motor_manager.polinizar_selectivo(
            ready_plants, 
            sensor_data=sensor_data,
            ignorar_condiciones=ignorar_condiciones
        )
        
        # Verificar si está en modo simulación
        is_simulated = motor_manager.is_simulation_mode()
        
        if result["status"] in ["success", "simulated"]:
            # Guardar en log
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "pollinate_selective",
                "plants": ready_plants,
                "verificacion_ambiental": verificacion_previa,
                "result": "success",
                "simulated": is_simulated  # Registrar si fue simulado
            }
            save_detection_log(log_entry)
            
            response = {
                "status": "success",
                "message": f"Polinización completada para plantas: {ready_plants}",
                "plantas_polinizadas": ready_plants,
                "total": len(ready_plants),
                "verificacion_ambiental": verificacion_previa,
                "detalles": result
            }
            
            # Agregar nota si fue simulado
            if is_simulated:
                response["nota"] = "⚠️ Ejecutado en MODO SIMULACIÓN - hardware no conectado"
                response["simulation_mode"] = True
            
            return response
        else:
            return {
                "status": result.get("status", "error"),
                "message": result.get("message", "Error en polinización"),
                "verificacion": result.get("verificacion"),
                "plantas_listas": ready_plants,
                "simulation_mode": is_simulated
            }
            
    except Exception as e:
        logger.error(f"❌ Error en polinización selectiva: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/pollinate-plants")
async def pollinate_specific_plants(request: PollinateRequest):
    """
    Polinizar plantas específicas (selección manual).
    Body: {"plantas": [1, 3], "ignorar_condiciones": false}
    """
    try:
        plantas = request.plantas
        ignorar = request.ignorar_condiciones
        
        for p in plantas:
            if p < 1 or p > 4:
                raise HTTPException(status_code=400, detail=f"Planta {p} inválida. Debe ser 1-4")
        
        if not plantas:
            raise HTTPException(status_code=400, detail="Debe especificar al menos una planta")
        
        # Obtener datos de sensores
        sensor_data = None
        verificacion = None
        
        if not ignorar:
            sensor_data = get_sensor_data_for_verification()
            if sensor_data:
                verificacion = motor_manager.verificar_condiciones_ambientales(sensor_data)
                if not verificacion["puede_polinizar"]:
                    return {
                        "status": "condiciones_no_optimas",
                        "message": "Condiciones ambientales no adecuadas",
                        "verificacion": verificacion
                    }
        
        logger.info(f"🌸 Polinización manual: plantas {plantas}")
        
        result = motor_manager.polinizar_selectivo(
            plantas,
            sensor_data=sensor_data,
            ignorar_condiciones=ignorar
        )
        
        if result["status"] == "success":
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "pollinate_manual",
                "plants": plantas,
                "verificacion_ambiental": verificacion,
                "result": "success"
            }
            save_detection_log(log_entry)
        
        return {
            "status": result["status"],
            "message": f"Polinización completada para plantas: {plantas}" if result["status"] == "success" else result.get("message"),
            "plantas_polinizadas": plantas if result["status"] == "success" else [],
            "verificacion_ambiental": verificacion,
            "detalles": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ENDPOINTS DE SENSORES
# ==========================================

@app.get("/api/sensors", response_model=List[SensorData])
async def get_all_sensors():
    """Obtener últimos datos de todos los sensores"""
    try:
        data = sensor_reader.get_latest_data()
        if data.empty:
            raise HTTPException(status_code=404, detail="No hay datos de sensores disponibles")
        
        result = []
        for _, row in data.iterrows():
            result.append({
                "timestamp": str(row['timestamp']),
                "sensor_id": row['sensor_id'],
                "temperature": float(row['temperature']),
                "moisture": float(row['moisture']),
                "light": int(row['light']),
                "conductivity": int(row['conductivity']),
                "battery": int(row['battery'])
            })
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sensors/{sensor_id}")
async def get_sensor_by_id(sensor_id: str):
    """Obtener datos de un sensor específico"""
    try:
        data = sensor_reader.get_sensor_data(sensor_id)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} no encontrado")
        
        latest = data.iloc[-1]
        return {
            "sensor_id": sensor_id,
            "timestamp": str(latest['timestamp']),
            "temperature": float(latest['temperature']),
            "moisture": float(latest['moisture']),
            "light": int(latest['light']),
            "conductivity": int(latest['conductivity']),
            "battery": int(latest['battery']),
            "total_readings": len(data)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sensors/{sensor_id}/history")
async def get_sensor_history(sensor_id: str, limit: int = 50):
    """Obtener historial de un sensor"""
    try:
        data = sensor_reader.get_sensor_data(sensor_id)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} no encontrado")
        
        history = data.tail(limit)
        
        result = []
        for _, row in history.iterrows():
            result.append({
                "timestamp": str(row['timestamp']),
                "temperature": float(row['temperature']),
                "moisture": float(row['moisture']),
                "light": int(row['light']),
                "conductivity": int(row['conductivity']),
                "battery": int(row['battery'])
            })
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sensors/history/aggregated")
async def get_sensors_history_aggregated(minutes: int = Query(15, ge=5, le=60)):
    """
    Obtener datos agregados (promedios) de los últimos X minutos.
    Útil para la verificación ambiental.
    
    Args:
        minutes: Minutos hacia atrás para calcular promedio (default: 15)
    """
    try:
        from motor_controller import sensor_history
        
        result = {
            "periodo_minutos": minutes,
            "timestamp": datetime.now().isoformat(),
            "promedios": {}
        }
        
        for param in ["temperatura", "humedad", "conductividad", "luz"]:
            avg = sensor_history.get_average(param, minutes)
            latest = sensor_history.get_latest(param)
            result["promedios"][param] = {
                "promedio": round(avg, 2) if avg else None,
                "ultimo_valor": round(latest, 2) if latest else None
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ENDPOINTS DE DETECCIÓN YOLO (LEGACY)
# ==========================================

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detección YOLO simple (legacy - una imagen)"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = f"{OUTPUT_DIR}/input_{timestamp}_{file.filename}"
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        output_path = f"{OUTPUT_DIR}/detected_{timestamp}_{file.filename}"
        detections = yolo_processor.process_image(input_path, output_path)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "image_name": file.filename,
            "input_path": input_path,
            "output_path": output_path,
            "total_objects": detections['total_objects'],
            "flowers": detections['flowers'],
            "fruits": detections['fruits'],
            "should_activate_motor": detections['flowers'].get('Lista_para_polinizar', 0) > 0,
            "motor_activated": False
        }
        
        save_detection_log(result)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en detección: {str(e)}")

@app.get("/api/detections")
async def get_detections_history(limit: int = 20, only_images: bool = False):
    """Obtener historial de detecciones
    
    Args:
        limit: Número máximo de resultados
        only_images: Si True, solo devuelve detecciones con imágenes (filtra acciones de polinización)
    """
    try:
        log = load_detection_log()
        
        if only_images:
            # Filtrar solo entradas que tienen output_path (detecciones con imagen)
            log = [entry for entry in log if entry.get('output_path')]
        
        return log[-limit:] if len(log) > limit else log
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detections/image/{filename}")
async def get_detection_image(filename: str):
    """Obtener imagen procesada"""
    file_path = f"{OUTPUT_DIR}/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(file_path)

# ==========================================
# ENDPOINTS DE MOTOR
# ==========================================

@app.post("/api/motor/activate")
async def activate_motor_endpoint(command: MotorCommand):
    """Activar motor manualmente (legacy)"""
    try:
        result = motor_manager.activate(
            duration=command.duration, 
            steps=command.steps
        )
        
        if result.get("status") == "busy":
            raise HTTPException(status_code=409, detail="Motor ocupado")
        
        return {
            "status": "success",
            "message": "Motor activado",
            "details": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/motor/status")
async def get_motor_status():
    """Obtener estado del motor"""
    try:
        status = motor_manager.get_status()
        
        # Agregar información de modo
        status["simulation_mode"] = motor_manager.is_simulation_mode()
        status["hardware_available"] = motor_manager.is_hardware_available()
        
        return status
    except Exception as e:
        # Incluso si falla, retornar un estado válido
        return {
            "status": "error",
            "error": str(e),
            "simulation_mode": True,
            "hardware_available": False,
            "message": "Error obteniendo estado del motor"
        }

@app.post("/api/motor/stop")
async def stop_motor():
    """Detener motor (emergencia)"""
    try:
        result = motor_manager.stop()
        return {
            "status": "success",
            "message": "Motor detenido",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/motor/home")
async def motor_go_home():
    """Enviar motor a posición inicial"""
    try:
        result = motor_manager.ir_a_home()
        return {
            "status": "success",
            "message": "Motor en posición home",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/motor/reset")
async def reset_motor():
    """Resetear motor completamente"""
    try:
        result = motor_manager.reset_completo()
        return {
            "status": "success",
            "message": "Motor reseteado",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/motor/test-signal")
async def test_external_signal(duration: float = Query(2.0, ge=0.5, le=10.0)):
    """Test de señal externa sin mover motores"""
    try:
        result = motor_manager.test_senal_externa(duration)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ENDPOINTS ADICIONALES
# ==========================================

@app.get("/api/classes")
async def get_yolo_classes():
    """Obtener lista de clases YOLO"""
    return {
        "flowers": ["Lista_para_polinizar", "No_desarrollada", "Sin_polen"],
        "fruits": ["verde", "mixto", "rojo"],
        "total_classes": 6
    }

@app.delete("/api/plants/clear-all")
async def clear_all_plants():
    """Limpiar datos de todas las plantas"""
    global plants_data
    
    for plant_id in range(1, 5):
        for img in plants_data[plant_id]["images"]:
            filepath = img.get("filepath")
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
        
        plants_data[plant_id] = {
            "images": [],
            "analysis": None,
            "ready": False,
            "last_update": datetime.now().isoformat()
        }
    
    return {
        "status": "success",
        "message": "Datos de todas las plantas eliminados"
    }

@app.get("/api/system/info")
async def get_system_info():
    """Información del sistema"""
    motor_status = motor_manager.get_status()
    sensor_data = get_sensor_data_for_verification()
    
    return {
        "project": "Monitor de Tomates",
        "version": "3.1.0",
        "features": [
            "Grid 2x2 para 4 plantas",
            "Múltiples fotos por planta",
            "Análisis YOLO agregado",
            "Polinización selectiva",
            "Verificación ambiental automática",
            "Señal externa GPIO24"
        ],
        "motor_status": {
            "connected": motor_status.get("hardware_connected"),
            "signal_pin": motor_status.get("signal_pin"),
            "activations": motor_status.get("total_activations")
        },
        "sensors_status": {
            "available": sensor_data is not None,
            "count": len(sensor_data) if sensor_data else 0
        },
        "plants_summary": {
            pid: {
                "images": len(plants_data[pid]["images"]),
                "ready": plants_data[pid]["ready"]
            }
            for pid in range(1, 5)
        },
        "condiciones_optimas": motor_status.get("condiciones_optimas", {})
    }
@app.post("/api/motor/retry-hardware")
async def retry_hardware_connection():
    """
    Intenta reconectar el hardware del motor.
    Útil cuando se conecta el hardware después de iniciar el servidor.
    """
    try:
        result = motor_manager.retry_hardware()
        return {
            "status": "success",
            "result": result,
            "message": "Intento de reconexión completado"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("🍅 MONITOR DE TOMATES v3.1")
    print("   Con verificación ambiental automática")
    print("=" * 60)
    print()
    print("✅ Características:")
    print("   📷 Grid 2x2 - múltiples fotos por planta")
    print("   🔍 Análisis YOLO agregado")
    print("   🌡️ Verificación de condiciones ambientales")
    print("   🌸 Polinización selectiva inteligente")
    print("   📳 Señal externa GPIO24")
    print()
    print("📡 Endpoints principales:")
    print("   GET  /api/plants              - Estado plantas")
    print("   POST /api/plants/{id}/images  - Subir imagen")
    print("   POST /api/analyze-all         - Analizar todas")
    print("   GET  /api/environmental-check - Verificar ambiente")
    print("   POST /api/pollinate-ready     - Polinizar listas")
    print()
    print("🌐 Dashboard: http://localhost:8000")
    print("📊 API Docs:  http://localhost:8000/docs")
    print("=" * 60)
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

