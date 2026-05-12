"""
captura_datos.py — FLUX CNC IoT · Recolección de datos etiquetados
===================================================================
Lee datos del ESP32 por Serial y los guarda en archivos CSV separados
según la clase (etiqueta) que el usuario seleccione.

Uso:
    pip install pyserial
    python captura_datos.py

Genera (en la carpeta data/ del proyecto):
    data/reposo.csv            - clase 0
    data/operacion_normal.csv  - clase 1
    data/anomalia.csv          - clase 2

Formato CSV:
    timestamp_ms,accel_x,accel_y,accel_z,temperatura,humedad,label
"""

import serial
import serial.tools.list_ports
import csv
import os
import sys
import time

# ── Configuración ────────────────────────────────────────────────────────────
BAUDRATE       = 115200
SAMPLES_TARGET = 350   # muestras por sesión de captura

# Ruta a la carpeta data/ (sube un nivel desde Esp32/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")

CLASES = {
    0: ("Reposo",           "reposo.csv",           "Máquina apagada o en pausa"),
    1: ("Operacion Normal", "operacion_normal.csv", "Maquinado dentro de parámetros normales"),
    2: ("Anomalia",         "anomalia.csv",         "Vibración excesiva — posible falla mecánica"),
}

CSV_HEADER = ["timestamp_ms", "accel_x", "accel_y", "accel_z",
              "temperatura", "humedad", "label"]

# ── Helpers ──────────────────────────────────────────────────────────────────
def listar_puertos():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("ERROR: No se encontraron puertos seriales disponibles.")
        sys.exit(1)
    return ports

def seleccionar_puerto():
    ports = listar_puertos()
    print("\nPuertos seriales disponibles:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  —  {p.description}")
    if len(ports) == 1:
        print(f"\nUsando único puerto detectado: {ports[0].device}")
        return ports[0].device
    while True:
        idx = input("\nElige el número de puerto (0, 1, ...): ").strip()
        try:
            return ports[int(idx)].device
        except (ValueError, IndexError):
            print("  Número inválido, intenta de nuevo.")

def contar_muestras(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, newline="") as f:
        return max(0, sum(1 for _ in f) - 1)  # descuenta el header

def guardar_filas(filepath, filas):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    archivo_nuevo = not os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow(CSV_HEADER)
        writer.writerows(filas)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FLUX CNC — Captura de datos etiquetados")
    print("=" * 60)

    # Mostrar estado actual de cada clase
    print("\nMuestras ya capturadas:")
    for label, (nombre, archivo, _) in CLASES.items():
        filepath = os.path.join(DATA_DIR, archivo)
        n = contar_muestras(filepath)
        barra = "█" * (n // 10) + f"  {n} muestras"
        print(f"  [{label}] {nombre:20s}  {barra}")

    # Elegir clase a capturar
    print("\nClases disponibles:")
    for label, (nombre, _, desc) in CLASES.items():
        print(f"  [{label}] {nombre} — {desc}")
    while True:
        entrada = input("\nElige la clase a capturar (0 / 1 / 2): ").strip()
        if entrada in ("0", "1", "2"):
            label = int(entrada)
            break
        print("  Opción inválida.")

    nombre_clase, archivo_csv, desc_clase = CLASES[label]
    filepath = os.path.join(DATA_DIR, archivo_csv)

    # Elegir puerto y conectar
    puerto = seleccionar_puerto()
    print(f"\nConectando a {puerto} @ {BAUDRATE} baud...")
    try:
        ser = serial.Serial(puerto, BAUDRATE, timeout=2)
    except serial.SerialException as e:
        print(f"ERROR al abrir puerto: {e}")
        sys.exit(1)

    # Esperar que el ESP32 esté listo
    print("Esperando ESP32", end="", flush=True)
    inicio = time.time()
    listo = False
    while time.time() - inicio < 15:
        linea = ser.readline().decode("utf-8", errors="ignore").strip()
        if linea.startswith("#"):
            print(f"\n  {linea}")
        if "#LISTO" in linea:
            listo = True
            break
        print(".", end="", flush=True)
        time.sleep(0.2)

    if not listo:
        print("\nAVISO: no se recibió '#LISTO'. El ESP32 puede necesitar más tiempo.")

    # Instrucciones al usuario
    print(f"\n{'─' * 60}")
    print(f"Clase seleccionada: [{label}] {nombre_clase}")
    print(f"Descripción: {desc_clase}")
    print(f"Muestras a capturar: {SAMPLES_TARGET}")
    print(f"Archivo destino: data/{archivo_csv}")
    print(f"{'─' * 60}")
    input("\nPon el dispositivo en el estado correcto y presiona ENTER para iniciar...")

    # Limpiar buffer y enviar comando de inicio
    ser.reset_input_buffer()
    ser.write(b'S')
    print(f"\nCapturando... (Ctrl+C para cancelar)\n")

    filas    = []
    capturadas = 0
    errores  = 0

    try:
        while capturadas < SAMPLES_TARGET:
            linea = ser.readline().decode("utf-8", errors="ignore").strip()

            # Ignorar líneas de comentario
            if not linea or linea.startswith("#"):
                continue

            partes = linea.split(",")
            if len(partes) != 6:
                errores += 1
                continue

            try:
                fila = [
                    int(partes[0]),    # timestamp_ms
                    float(partes[1]),  # accel_x
                    float(partes[2]),  # accel_y
                    float(partes[3]),  # accel_z
                    float(partes[4]),  # temperatura
                    float(partes[5]),  # humedad
                    label,             # etiqueta
                ]
                filas.append(fila)
                capturadas += 1

                # Progreso cada 50 muestras
                if capturadas % 50 == 0 or capturadas == SAMPLES_TARGET:
                    pct = int(capturadas / SAMPLES_TARGET * 40)
                    barra = "█" * pct + "░" * (40 - pct)
                    print(f"  [{barra}] {capturadas}/{SAMPLES_TARGET}", end="\r", flush=True)

            except ValueError:
                errores += 1
                continue

    except KeyboardInterrupt:
        print(f"\n\nCaptura interrumpida por el usuario ({capturadas} muestras).")

    # Pausar ESP32 y cerrar puerto
    ser.write(b'X')
    ser.close()

    if not filas:
        print("No se capturó ninguna muestra. Revisa la conexión del ESP32.")
        sys.exit(1)

    # Guardar CSV
    guardar_filas(filepath, filas)

    print(f"\n{'─' * 60}")
    print(f"Guardado:  data/{archivo_csv}")
    print(f"Nuevas muestras:  {len(filas)}")
    print(f"Total en archivo: {contar_muestras(filepath)}")
    if errores:
        print(f"Líneas descartadas (formato incorrecto): {errores}")
    print(f"{'─' * 60}")

    # Resumen final
    print("\nResumen del dataset:")
    total = 0
    for lbl, (nombre, arch, _) in CLASES.items():
        n = contar_muestras(os.path.join(DATA_DIR, arch))
        total += n
        estado = "OK (>=200)" if n >= 200 else f"Faltan {200 - n}"
        print(f"  [{lbl}] {nombre:20s} {n:4d} muestras  —  {estado}")
    print(f"  Total: {total} muestras")
    if total >= 600:
        print("\n  Dataset listo para entregar a David (>= 600 muestras).")
    else:
        print(f"\n  Faltan {600 - total} muestras para alcanzar el minimo de 600.")


if __name__ == "__main__":
    main()
