#!/usr/bin/env python3
"""
chessboard_intrinsics.py  –  Calibración de cámara con tablero de ajedrez
clásico (sin ArUco) usando **solo 5 coeficientes de distorsión**
[k1, k2, p1, p2, k3].

Ejemplo con tu tablero (8 × 11 cuadros de 35 mm):

```bash
python3 chessboard_intrinsics.py \
  --img-dir data/intrinsics \
  --squares-x 8 --squares-y 11 \
  --square-length 35.0 \
  --out calib_chess.yaml
```

El script muestra cada captura para aceptar/descartar.  Se requieren al
menos 3 válidas.  Tras calibrar se imprime la matriz intrínseca **K** y los
5 coeficientes de distorsión, guardándolos (YAML o NPZ) si se pasa `--out`.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Argumentos CLI                                                               #
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Calibración con tablero de ajedrez (5 coef dist)")
    p.add_argument("--img-dir", default="data/intrinsics",
                   help="Carpeta con imágenes *.jpg / *.png")
    p.add_argument("--squares-x", type=int, default=8, help="Cuadros en X (horizontal)")
    p.add_argument("--squares-y", type=int, default=11, help="Cuadros en Y (vertical)")
    p.add_argument("--square-length", type=float, default=35.0,
                   help="Tamaño del lado de un cuadro, en mm")
    p.add_argument("--out", default="", help="Archivo YAML/NPZ para guardar el resultado")
    return p.parse_args()

###############################################################################
# Visualización                                                                #
###############################################################################

def review_image(img_bgr: np.ndarray, corners: np.ndarray, pattern_size: tuple[int, int], title: str) -> bool:
    vis = img_bgr.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()  # bloqueante hasta cerrar ventana
    plt.close()

    resp = input("¿Mantener esta imagen? (Enter/y = sí, n = no) → ").strip().lower()
    return resp in ("", "y", "yes", "s", "si")

###############################################################################
# Main                                                                         #
###############################################################################

def main():
    args = parse_args()

    pattern_size = (args.squares_x - 1, args.squares_y - 1)  # esquinas internas

    # Objeto patrón 3D (z=0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= args.square_length

    objpoints, imgpoints = [], []
    image_size = None

    img_dir = Path(args.img_dir)
    img_paths = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"No se hallaron imágenes en {img_dir}")

    print(f"[INFO] {len(img_paths)} imagen(es) encontradas en {img_dir}")

    flags_detect = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] No se pudo leer {path}; se omite.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags_detect)
        if not ret:
            print(f"[INFO] {Path(path).name}: tablero no encontrado → omite.")
            continue

        # Refinamiento sub‑píxel
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if review_image(img, corners, pattern_size, Path(path).name):
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"  → Aceptada ({len(corners)} esquinas)")
        else:
            print("  → Descartada por el usuario")

    if len(objpoints) < 3:
        raise RuntimeError("No hay suficientes imágenes válidas para calibrar (≥3)")

    print(f"[INFO] Calibrando con {len(objpoints)} imágenes (modelo 5 coef)…")

    # Solo 5 coeficientes (k1, k2, p1, p2, k3)
    dist_coeffs_init = np.zeros((5, 1))
    flags_calib = 0  # sin CALIB_RATIONAL_MODEL ni extras

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, dist_coeffs_init, flags=flags_calib)

    print("\n=== RESULTADOS ===")
    print(f"RMS reproj error : {rms:.4f} px")
    print("Matriz intrínseca (K):\n", camera_matrix)
    print("Distorsión [k1 k2 p1 p2 k3]:", dist_coeffs.ravel())

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() in {".yaml", ".yml"}:
            fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", camera_matrix)
            fs.write("dist_coeffs", dist_coeffs)
            fs.write("rms", rms)
            fs.release()
        else:
            np.savez(out_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rms=rms)
        print(f"[INFO] Guardado en {out_path}")

    print("[INFO] Terminado.")


if __name__ == "__main__":
    main()
