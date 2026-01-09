"""
Data Reader - Lectura de datos de sensores MiFlora desde CSV
Versión corregida para formato CSV sin headers
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataReader:
    """
    Lector de datos de sensores MiFlora desde archivo CSV
    """
    
    def __init__(self, csv_path: str):
        """
        Inicializar lector de datos
        
        Args:
            csv_path: Ruta al archivo CSV con datos de sensores
        """
        self.csv_path = csv_path
        self.df = None
        
        # Definir nombres de columnas (el CSV no tiene headers)
        self.column_names = [
            'timestamp',
            'sensor_id',
            'temperature',
            'moisture',
            'light',
            'conductivity',
            'battery'
        ]
        
        logger.info("=" * 60)
        logger.info("📊 Inicializando SensorDataReader")
        logger.info("=" * 60)
        logger.info(f"   CSV Path: {csv_path}")
        
        self._load_data()
    
    def _load_data(self):
        """Cargar datos del CSV"""
        try:
            # Leer CSV sin headers
            self.df = pd.read_csv(
                self.csv_path,
                header=None,
                names=self.column_names
            )
            
            # Filtrar filas válidas (que tienen sensor_id válido)
            # Eliminar filas con $announce u otros valores inválidos
            self.df = self.df[
                (self.df['sensor_id'].notna()) & 
                (self.df['sensor_id'].str.startswith('Sensor', na=False))
            ]
            
            # Convertir tipos de datos
            self.df['temperature'] = pd.to_numeric(self.df['temperature'], errors='coerce')
            self.df['moisture'] = pd.to_numeric(self.df['moisture'], errors='coerce')
            self.df['light'] = pd.to_numeric(self.df['light'], errors='coerce')
            self.df['conductivity'] = pd.to_numeric(self.df['conductivity'], errors='coerce')
            self.df['battery'] = pd.to_numeric(self.df['battery'], errors='coerce')
            
            # Eliminar filas con NaN en datos importantes
            self.df = self.df.dropna(subset=['sensor_id', 'temperature', 'moisture'])
            
            logger.info("✅ CSV encontrado: " + self.csv_path)
            logger.info(f"   Total registros válidos: {len(self.df)}")
            
            if len(self.df) > 0:
                sensors = self.df['sensor_id'].unique()
                logger.info(f"   Sensores detectados: {', '.join(sensors)}")
            else:
                logger.warning("⚠️  No se encontraron datos válidos en el CSV")
            
            logger.info("=" * 60)
            
        except FileNotFoundError:
            logger.error(f"❌ Archivo CSV no encontrado: {self.csv_path}")
            self.df = pd.DataFrame(columns=self.column_names)
        except Exception as e:
            logger.error(f"❌ Error leyendo CSV: {e}")
            self.df = pd.DataFrame(columns=self.column_names)
    
    def reload_data(self):
        """Recargar datos del CSV (útil si el archivo se actualiza)"""
        logger.info("🔄 Recargando datos del CSV...")
        self._load_data()
    
    def get_latest_data(self) -> pd.DataFrame:
        """
        Obtener los últimos datos de todos los sensores
        
        Returns:
            DataFrame con los últimos datos de cada sensor
        """
        try:
            if self.df.empty:
                logger.warning("⚠️  No hay datos disponibles")
                return pd.DataFrame()
            
            # Recargar datos para obtener lo más reciente
            self._load_data()
            
            # Obtener última lectura de cada sensor
            latest = self.df.groupby('sensor_id').tail(1).reset_index(drop=True)
            
            logger.info(f"📊 Datos más recientes obtenidos: {len(latest)} sensores")
            
            return latest
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo últimos datos: {e}")
            return pd.DataFrame()
    
    def get_sensor_data(self, sensor_id: str) -> pd.DataFrame:
        """
        Obtener todos los datos de un sensor específico
        
        Args:
            sensor_id: ID del sensor (ej: 'SensorTomate1')
        
        Returns:
            DataFrame con todos los datos del sensor
        """
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # Recargar datos
            self._load_data()
            
            sensor_data = self.df[self.df['sensor_id'] == sensor_id].copy()
            
            if sensor_data.empty:
                logger.warning(f"⚠️  No se encontraron datos para {sensor_id}")
            else:
                logger.info(f"📊 Datos de {sensor_id}: {len(sensor_data)} registros")
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos de {sensor_id}: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Calcular estadísticas de todos los sensores
        
        Returns:
            DataFrame con estadísticas (promedio, min, max) por sensor
        """
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # Recargar datos
            self._load_data()
            
            # Agrupar por sensor y calcular estadísticas
            stats = self.df.groupby('sensor_id').agg({
                'temperature': ['mean', 'min', 'max'],
                'moisture': ['mean', 'min', 'max'],
                'light': ['mean', 'min', 'max'],
                'conductivity': ['mean', 'min', 'max'],
                'battery': ['mean', 'min', 'max']
            }).round(2)
            
            logger.info(f"📊 Estadísticas calculadas para {len(stats)} sensores")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error calculando estadísticas: {e}")
            return pd.DataFrame()
    
    def get_sensor_count(self) -> int:
        """
        Obtener número de sensores únicos
        
        Returns:
            Número de sensores
        """
        if self.df.empty:
            return 0
        return self.df['sensor_id'].nunique()

if __name__ == "__main__":
    # Pruebas del lector
    print("\n🧪 Iniciando pruebas del SensorDataReader...\n")
    
    # Usar el path correcto
    csv_path = "/home/arthur/yolo_env/miflora_log.csv"
    
    # Crear instancia
    reader = SensorDataReader(csv_path)
    
    # Prueba 1: Obtener últimos datos
    print("\n1️⃣ Últimos datos de todos los sensores:")
    latest = reader.get_latest_data()
    if not latest.empty:
        print(latest[['timestamp', 'sensor_id', 'temperature', 'moisture', 'battery']])
    else:
        print("   No hay datos disponibles")
    
    # Prueba 2: Datos de un sensor específico
    print("\n2️⃣ Datos de SensorTomate1:")
    sensor1 = reader.get_sensor_data('SensorTomate1')
    if not sensor1.empty:
        print(f"   Total registros: {len(sensor1)}")
        print(f"   Última lectura:")
        print(sensor1[['timestamp', 'temperature', 'moisture', 'battery']].tail(1))
    else:
        print("   No hay datos disponibles")
    
    # Prueba 3: Estadísticas
    print("\n3️⃣ Estadísticas de todos los sensores:")
    stats = reader.get_statistics()
    if not stats.empty:
        print(stats)
    else:
        print("   No hay datos disponibles")
    
    # Prueba 4: Número de sensores
    print(f"\n4️⃣ Número de sensores detectados: {reader.get_sensor_count()}")
    
    print("\n✅ Todas las pruebas completadas")
    print("=" * 60)
    