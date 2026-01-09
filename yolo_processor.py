"""
Procesador YOLO para detección de flores y frutos de tomate
Uso de Ultralytics YOLO directamente
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    logger.info("✅ Ultralytics YOLO importado correctamente")
except ImportError:
    logger.error("❌ No se pudo importar Ultralytics. Instala: pip install ultralytics")
    raise


class YOLOProcessor:
    """Procesador YOLO para detección de flores y frutos de tomate usando Ultralytics"""
    
    def __init__(self, model_path="models/best.pt", conf_threshold=0.5, iou_threshold=0.45):
        """
        Inicializar procesador YOLO
        
        Args:
            model_path: Ruta al modelo YOLO (.pt o .onnx)
            conf_threshold: Umbral de confianza (0-1)
            iou_threshold: Umbral de IoU para NMS
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Cargar modelo
        self.model = None
        self.labels = None
        
        self._load_model()
        
        # Clases del modelo (obtenidas del modelo)
        self.classes = list(self.labels.values())
        
        # Colores para visualización (BGR) - Tableu 10 color scheme
        self.bbox_colors = [
            (164, 120, 87),   # Lista_para_polinizar
            (68, 148, 228),   # No_desarrollada
            (93, 97, 209),    # Sin_polen
            (178, 182, 133),  # mixto
            (88, 159, 106),   # rojo
            (96, 202, 231)    # verde
        ]
        
        # Categorías para análisis
        self.flores = ["Lista_para_polinizar", "No_desarrollada", "Sin_polen"]
        self.frutos = ["mixto", "rojo", "verde"]
        
        logger.info(f"✅ YOLO Processor (Ultralytics) inicializado")
        logger.info(f"   Modelo: {self.model_path.name}")
        logger.info(f"   Clases: {len(self.classes)}")
        logger.info(f"   Clases detectables: {', '.join(self.classes)}")
        logger.info(f"   Conf threshold: {self.conf_threshold}")
    
    def _load_model(self):
        """Cargar modelo YOLO"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            # Cargar modelo con Ultralytics
            self.model = YOLO(str(self.model_path), task='detect')
            self.labels = self.model.names
            
            logger.info(f"✅ Modelo YOLO cargado exitosamente")
            logger.info(f"   Tipo: {self.model_path.suffix}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def detect(self, image_path):
        """
        Detectar objetos en una imagen
        
        Args:
            image_path: Ruta a la imagen o array numpy
            
        Returns:
            dict: Resultados de detección
        """
        try:
            # Leer imagen
            if isinstance(image_path, (str, Path)):
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise ValueError(f"No se pudo leer la imagen: {image_path}")
            else:
                frame = image_path
            
            orig_shape = frame.shape[:2]
            
            # Inferencia usando Ultralytics (igual que tu código)
            results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # Extraer detecciones
            detections_boxes = results[0].boxes
            
            # Procesar detecciones
            detections = []
            
            for i in range(len(detections_boxes)):
                # Obtener coordenadas del bounding box
                xyxy_tensor = detections_boxes[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)
                
                # Obtener clase
                classidx = int(detections_boxes[i].cls.item())
                classname = self.labels[classidx]
                
                # Obtener confianza
                conf = detections_boxes[i].conf.item()
                
                # Agregar a lista de detecciones
                detections.append({
                    'class_id': classidx,
                    'class_name': classname,
                    'confidence': float(conf),
                    'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]
                })
            
            # Análisis de resultados
            analysis = self.analyze_detections(detections)
            
            logger.info(f"✅ Detección completada: {analysis['total']} objetos encontrados")
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'analysis': analysis,
                'image_shape': orig_shape
            }
            
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_detections(self, detections):
        """
        Analizar detecciones y generar estadísticas
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            dict: Análisis de detecciones
        """
        analysis = {
            'total': len(detections),
            'flores': 0,
            'frutos': 0,
            'por_clase': {},
            'flores_listas_polinizar': 0,
            'frutos_maduros': 0
        }
        
        # Contar por clase
        for det in detections:
            class_name = det['class_name']
            
            # Incrementar contador de clase
            if class_name not in analysis['por_clase']:
                analysis['por_clase'][class_name] = 0
            analysis['por_clase'][class_name] += 1
            
            # Categorizar
            if class_name in self.flores:
                analysis['flores'] += 1
                if class_name == "Lista_para_polinizar":
                    analysis['flores_listas_polinizar'] += 1
            elif class_name in self.frutos:
                analysis['frutos'] += 1
                if class_name == "rojo":
                    analysis['frutos_maduros'] += 1
        
        return analysis
    
    def draw_detections(self, image, detections):
        """
        Dibujar detecciones en la imagen (igual que tu código)
        
        Args:
            image: Imagen BGR de OpenCV
            detections: Lista de detecciones
            
        Returns:
            numpy.ndarray: Imagen con detecciones dibujadas
        """
        frame = image.copy()
        
        for det in detections:
            classidx = det['class_id']
            classname = det['class_name']
            conf = det['confidence']
            xmin, ymin, xmax, ymax = det['bbox']
            
            # Color según clase (Tableu 10)
            color = self.bbox_colors[classidx % len(self.bbox_colors)]
            
            # Dibujar bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Etiqueta (igual que tu código)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            
            # Fondo para texto
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                         (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            
            # Texto
            cv2.putText(frame, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Agregar contador total
        object_count = len(detections)
        cv2.putText(frame, f'Number of objects: {object_count}', (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def process_image(self, input_path, output_path):
        """
        Procesar imagen completa (detectar + dibujar + guardar)
        Esta función es la que llama app.py en el endpoint /api/detect
        
        Args:
            input_path: Ruta de la imagen de entrada
            output_path: Ruta donde guardar la imagen con detecciones
            
        Returns:
            dict: Resultados en el formato que espera app.py
        """
        try:
            logger.info(f"📸 Procesando imagen: {input_path}")
            
            # 1. Detectar objetos
            results = self.detect(input_path)
            
            if not results['success']:
                raise Exception(results.get('error', 'Error desconocido en detección'))
            
            # 2. Leer imagen original
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"No se pudo leer la imagen: {input_path}")
            
            # 3. Dibujar detecciones
            output_image = self.draw_detections(image, results['detections'])
            
            # 4. Guardar imagen procesada
            cv2.imwrite(str(output_path), output_image)
            logger.info(f"💾 Imagen guardada: {output_path}")
            
            # 5. Preparar respuesta en el formato que espera app.py
            analysis = results['analysis']
            
            # Construir diccionario de flores
            flowers_dict = {}
            for flower_class in self.flores:
                flowers_dict[flower_class] = analysis['por_clase'].get(flower_class, 0)
            
            # Construir diccionario de frutos
            fruits_dict = {}
            for fruit_class in self.frutos:
                fruits_dict[fruit_class] = analysis['por_clase'].get(fruit_class, 0)
            
            response = {
                'total_objects': analysis['total'],
                'flowers': flowers_dict,
                'fruits': fruits_dict,
                'detections': results['detections'],
                'timestamp': results['timestamp']
            }
            
            logger.info(f"✅ Procesamiento completado: {analysis['total']} objetos")
            logger.info(f"   Flores: {flowers_dict}")
            logger.info(f"   Frutos: {fruits_dict}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error en process_image: {e}", exc_info=True)
            raise
    
    def save_results(self, results, output_dir="outputs"):
        """
        Guardar resultados en JSON
        
        Args:
            results: Resultados de detección
            output_dir: Directorio de salida
            
        Returns:
            str: Ruta al archivo guardado
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"detection_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Resultados guardados: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error guardando resultados: {e}")
            return None


# Prueba del módulo
if __name__ == "__main__":
    import sys
    
    print("🧪 Probando YOLO Processor (Ultralytics)...")
    print("=" * 60)
    
    # Verificar que existe el modelo
    # Probar con .pt primero, luego .onnx
    model_path = None
    for ext in ['.pt', '.onnx']:
        test_path = Path(f"models/best{ext}")
        if test_path.exists():
            model_path = test_path
            break
    
    if not model_path:
        print(f"\n⚠️ Modelo no encontrado.")
        print(f"   Por favor, coloca tu modelo 'best.pt' o 'best.onnx' en la carpeta 'models/'")
        sys.exit(1)
    
    # Crear procesador
    try:
        processor = YOLOProcessor(model_path=model_path, conf_threshold=0.5)
        print(f"\n✅ Procesador Ultralytics creado exitosamente")
        
    except Exception as e:
        print(f"\n❌ Error creando procesador: {e}")
        sys.exit(1)
    
    # Si se proporciona una imagen como argumento, procesarla
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if not Path(image_path).exists():
            print(f"\n⚠️ Imagen no encontrada: {image_path}")
            sys.exit(1)
        
        print(f"\n📸 Procesando imagen: {image_path}")
        print("-" * 60)
        
        # Probar la función process_image (la que usa el endpoint)
        output_path = Path("outputs") / f"test_detection_{Path(image_path).name}"
        output_path.parent.mkdir(exist_ok=True)
        
        try:
            results = processor.process_image(image_path, output_path)
            
            print(f"\n✅ Procesamiento exitoso!")
            print(f"\n📊 RESULTADOS:")
            print(f"   Total objetos: {results['total_objects']}")
            print(f"   Flores: {results['flowers']}")
            print(f"   Frutos: {results['fruits']}")
            print(f"\n💾 Imagen guardada: {output_path}")
            
        except Exception as e:
            print(f"\n❌ Error en process_image: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("\n💡 Para probar con una imagen:")
        print(f"   python backend/yolo_processor.py /ruta/a/imagen.jpg")
    
    print("\n" + "=" * 60)
    print("✅ Prueba completada")