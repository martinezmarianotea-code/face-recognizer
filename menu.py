"""
Menú interactivo para el sistema de reconocimiento facial.
Uso: python menu.py
"""

import os
import sys
import json
import subprocess
import shutil

# ─── Colores ──────────────────────────────────────────────────────────────────
R  = "\033[91m"
G  = "\033[92m"
Y  = "\033[93m"
B  = "\033[94m"
C  = "\033[96m"
W  = "\033[97m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def clear():
    os.system("clear")

def header():
    clear()
    print(f"{B}{BOLD}")
    print("  ╔══════════════════════════════════════════╗")
    print("  ║       RECONOCIMIENTO FACIAL              ║")
    print("  ║       YOLO + LBPH                        ║")
    print("  ╚══════════════════════════════════════════╝")
    print(f"{RESET}")

def personas_registradas():
    dataset = "data/known_faces"
    if not os.path.exists(dataset):
        return []
    return [d for d in os.listdir(dataset)
            if os.path.isdir(os.path.join(dataset, d))]

def modelo_entrenado():
    return os.path.exists("models/face_recognizer.yml")

def status_bar():
    personas = personas_registradas()
    trained  = modelo_entrenado()
    print(f"  {DIM}Personas: {W}{BOLD}{len(personas)}{RESET}{DIM}  │  "
          f"Modelo: {(G+'✔ listo' if trained else R+'✘ sin entrenar')}{RESET}{DIM}  │  "
          f"Fotos totales: {W}{BOLD}{total_fotos()}{RESET}")
    print()

def total_fotos():
    dataset = "data/known_faces"
    total = 0
    for p in personas_registradas():
        carpeta = os.path.join(dataset, p)
        total += len([f for f in os.listdir(carpeta)
                      if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
    return total

def listar_personas():
    header()
    status_bar()
    print(f"  {C}{BOLD}Personas registradas:{RESET}\n")
    personas = personas_registradas()
    if not personas:
        print(f"  {Y}  No hay personas registradas aún.{RESET}")
    else:
        for p in sorted(personas):
            carpeta = os.path.join("data/known_faces", p)
            n = len([f for f in os.listdir(carpeta)
                     if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
            barra = "█" * min(n, 30)
            print(f"  {W}  {p:<25}{RESET}{G}{barra}{RESET} {DIM}{n} fotos{RESET}")
    print()
    input(f"  {DIM}Presioná Enter para volver...{RESET}")

def agregar_persona():
    header()
    status_bar()
    print(f"  {C}{BOLD}Agregar / actualizar persona{RESET}\n")
    nombre = input(f"  {W}Nombre de la persona: {RESET}").strip()
    if not nombre:
        return
    carpeta = os.path.join("data/known_faces", nombre)
    os.makedirs(carpeta, exist_ok=True)

    print(f"\n  {Y}¿De dónde tomamos las fotos?{RESET}")
    print(f"  {W}  1.{RESET} Carpeta existente")
    print(f"  {W}  2.{RESET} Webcam")
    op = input(f"\n  Opción: ").strip()

    if op == "1":
        ruta = input(f"\n  {W}Ruta de la carpeta con fotos: {RESET}").strip()
        if not os.path.isdir(ruta):
            print(f"\n  {R}Carpeta no encontrada.{RESET}")
            input(f"  {DIM}Enter para volver...{RESET}")
            return
        exts = (".jpg",".jpeg",".png",".bmp",".webp")
        archivos = [f for f in os.listdir(ruta) if f.lower().endswith(exts)]
        for f in archivos:
            shutil.copy2(os.path.join(ruta, f), carpeta)
        print(f"\n  {G}✔ {len(archivos)} fotos copiadas a {carpeta}{RESET}")

    elif op == "2":
        max_fotos = input(f"\n  {W}¿Cuántas fotos capturar? (default 30): {RESET}").strip()
        max_fotos = max_fotos if max_fotos.isdigit() else "30"
        print(f"\n  {Y}Abriendo webcam... Espacio=captura  A=auto  Q=salir{RESET}\n")
        subprocess.run([sys.executable, "2_collect_faces.py",
                        "--name", nombre, "--source", "webcam",
                        "--max", max_fotos])

    print(f"\n  {G}✔ Persona '{nombre}' actualizada.{RESET}")
    print(f"  {Y}  Recordá re-entrenar el modelo para aplicar los cambios.{RESET}")
    input(f"\n  {DIM}Enter para volver...{RESET}")

def entrenar():
    header()
    status_bar()
    print(f"  {C}{BOLD}Entrenando modelo...{RESET}\n")
    personas = personas_registradas()
    if not personas:
        print(f"  {R}No hay personas registradas. Agrega fotos primero.{RESET}")
        input(f"\n  {DIM}Enter para volver...{RESET}")
        return
    subprocess.run([sys.executable, "3_train.py"])
    print(f"\n  {G}✔ Modelo listo.{RESET}")
    input(f"\n  {DIM}Enter para volver...{RESET}")

def reconocer():
    header()
    status_bar()

    if not modelo_entrenado():
        print(f"  {R}El modelo no está entrenado. Ve a la opción 3 primero.{RESET}")
        input(f"\n  {DIM}Enter para volver...{RESET}")
        return

    print(f"  {C}{BOLD}Reconocimiento facial{RESET}\n")
    print(f"  {W}  1.{RESET} Webcam en tiempo real")
    print(f"  {W}  2.{RESET} Imagen o carpeta")
    op = input(f"\n  Opción: ").strip()

    cmd = [sys.executable, "4_recognize.py"]

    if op == "1":
        print(f"\n  {Y}Abriendo webcam... Q=salir{RESET}\n")
        subprocess.run(cmd)

    elif op == "2":
        ruta = input(f"\n  {W}Ruta de la imagen o carpeta: {RESET}").strip()
        if not ruta:
            return
        guardar = input(f"  {W}¿Guardar resultado en output/? (s/n): {RESET}").strip().lower()
        cmd += ["--source", ruta]
        if guardar == "s":
            cmd += ["--save"]
        subprocess.run(cmd)

def eliminar_persona():
    header()
    status_bar()
    print(f"  {C}{BOLD}Eliminar persona{RESET}\n")
    personas = sorted(personas_registradas())
    if not personas:
        print(f"  {Y}No hay personas registradas.{RESET}")
        input(f"\n  {DIM}Enter para volver...{RESET}")
        return
    for i, p in enumerate(personas, 1):
        print(f"  {W}  {i}.{RESET} {p}")
    print(f"  {W}  0.{RESET} Cancelar")
    op = input(f"\n  Opción: ").strip()
    if not op.isdigit() or int(op) == 0:
        return
    idx = int(op) - 1
    if 0 <= idx < len(personas):
        nombre = personas[idx]
        confirmar = input(f"\n  {R}¿Eliminar '{nombre}'? (s/n): {RESET}").strip().lower()
        if confirmar == "s":
            shutil.rmtree(os.path.join("data/known_faces", nombre))
            print(f"\n  {G}✔ '{nombre}' eliminado. Re-entrená el modelo.{RESET}")
    input(f"\n  {DIM}Enter para volver...{RESET}")

def main():
    while True:
        header()
        status_bar()
        print(f"  {W}{BOLD}¿Qué querés hacer?{RESET}\n")
        print(f"  {G}  1.{RESET}  Ver personas registradas")
        print(f"  {G}  2.{RESET}  Agregar / actualizar persona")
        print(f"  {G}  3.{RESET}  Entrenar modelo")
        print(f"  {G}  4.{RESET}  {BOLD}Reconocer{RESET}{G} (cámara o imagen)")
        print(f"  {Y}  5.{RESET}  Eliminar persona")
        print(f"  {R}  0.{RESET}  Salir\n")

        op = input(f"  Opción: ").strip()

        if   op == "1": listar_personas()
        elif op == "2": agregar_persona()
        elif op == "3": entrenar()
        elif op == "4": reconocer()
        elif op == "5": eliminar_persona()
        elif op == "0":
            clear()
            print(f"\n  {DIM}Hasta luego.{RESET}\n")
            break

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
