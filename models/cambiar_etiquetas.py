import os
import shutil

# Mapeo de cambios: {clase_antigua: clase_nueva}
MAPEO_CLASES = {
    0: 4,  # 0 → 4
    1: 5,  # 1 → 5
    2: 3   # 2 → 3
}

def cambiar_etiquetas_archivo(ruta_archivo, mapeo):
    """
    Cambia las etiquetas en un archivo según el mapeo proporcionado
    """
    with open(ruta_archivo, 'r') as f:
        lineas = f.readlines()
    
    lineas_nuevas = []
    cambios_realizados = 0
    
    for linea in lineas:
        datos = linea.strip().split()
        if len(datos) >= 5:  # Formato YOLO válido
            clase_antigua = int(datos[0])
            
            # Cambiar clase si está en el mapeo
            if clase_antigua in mapeo:
                clase_nueva = mapeo[clase_antigua]
                datos[0] = str(clase_nueva)
                cambios_realizados += 1
            
            # Reconstruir línea
            linea_nueva = ' '.join(datos) + '\n'
            lineas_nuevas.append(linea_nueva)
        else:
            lineas_nuevas.append(linea)  # Mantener líneas inválidas sin cambios
    
    # Guardar archivo modificado
    with open(ruta_archivo, 'w') as f:
        f.writelines(lineas_nuevas)
    
    return cambios_realizados

def cambiar_etiquetas_carpeta(carpeta_labels, mapeo, crear_backup=True):
    """
    Cambia las etiquetas en todos los archivos .txt de una carpeta
    """
    # Crear backup si se solicita
    if crear_backup:
        carpeta_backup = carpeta_labels + "_backup"
        if os.path.exists(carpeta_backup):
            print(f"⚠️  El backup ya existe: {carpeta_backup}")
            respuesta = input("¿Deseas sobrescribirlo? (s/n): ")
            if respuesta.lower() != 's':
                print("Operación cancelada.")
                return
            shutil.rmtree(carpeta_backup)
        
        shutil.copytree(carpeta_labels, carpeta_backup)
        print(f"✓ Backup creado en: {carpeta_backup}\n")
    
    # Procesar todos los archivos .txt
    archivos_procesados = 0
    total_cambios = 0
    
    print("Procesando archivos...")
    print("=" * 60)
    
    for archivo in os.listdir(carpeta_labels):
        if archivo.endswith('.txt'):
            ruta_completa = os.path.join(carpeta_labels, archivo)
            cambios = cambiar_etiquetas_archivo(ruta_completa, mapeo)
            
            if cambios > 0:
                print(f"✓ {archivo}: {cambios} etiquetas cambiadas")
                archivos_procesados += 1
                total_cambios += cambios
    
    print("=" * 60)
    print(f"\n📊 RESUMEN:")
    print(f"   Archivos modificados: {archivos_procesados}")
    print(f"   Total de cambios: {total_cambios}")
    print(f"\n   Mapeo aplicado:")
    for antigua, nueva in mapeo.items():
        print(f"     Clase {antigua} → Clase {nueva}")
    print("\n✅ ¡Proceso completado!")

def verificar_cambios(carpeta_labels):
    """
    Verifica la distribución de clases después del cambio
    """
    from collections import Counter
    
    clases = []
    for archivo in os.listdir(carpeta_labels):
        if archivo.endswith('.txt'):
            ruta = os.path.join(carpeta_labels, archivo)
            with open(ruta, 'r') as f:
                for linea in f:
                    datos = linea.strip().split()
                    if len(datos) >= 5:
                        clases.append(int(datos[0]))
    
    contador = Counter(clases)
    print("\n📈 Distribución de clases actual:")
    print("=" * 40)
    for clase in sorted(contador.keys()):
        print(f"   Clase {clase}: {contador[clase]} etiquetas")
    print("=" * 40)

# USO DEL PROGRAMA
if __name__ == "__main__":
    
    # Configuración
    carpeta_labels = r"C:\Users\artur\Downloads\cherry tomato.v2i.yolov8\train\labels"
    
    print("🔄 CAMBIADOR DE ETIQUETAS YOLO\n")
    print(f"Carpeta: {carpeta_labels}")
    print(f"\nMapeo de cambios:")
    print(f"  0 → 4 (Maduro)")
    print(f"  1 → 5 (Verde)")
    print(f"  2 → 3 (Pinton)")
    print("\n" + "=" * 60)
    
    # Confirmación del usuario
    respuesta = input("\n¿Deseas continuar? (s/n): ")
    
    if respuesta.lower() == 's':
        # Ejecutar cambios (con backup automático)
        cambiar_etiquetas_carpeta(
            carpeta_labels=carpeta_labels,
            mapeo=MAPEO_CLASES,
            crear_backup=True  # Cambia a False si no quieres backup
        )
        
        # Verificar resultados
        verificar_cambios(carpeta_labels)
    else:
        print("Operación cancelada.")