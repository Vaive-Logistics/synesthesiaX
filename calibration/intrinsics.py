#!/usr/bin/env python3
"""
charuco_intrinsics.py  –  Calibración de cámara con tablero ChArUco.

Cambios principales (v 1.3)
---------------------------
* **Solucionado el _TclError_** de Matplotlib: la función de revisión de
  imágenes ahora usa `plt.show()` bloqueante en vez de `waitforbuttonpress`,
  eliminando el conflicto con la ventana cerrada.
* Mensaje de ayuda sobre teclas (`Enter`/`y` para mantener, `n` para descartar)
  se muestra en consola justo después de cerrarse la ventana.

Ejemplo de uso (tablero 8 × 11, 35 mm, 20 mm, diccionario 4×4_50):

```bash
python3 charuco_intrinsics.py \
  --img-dir data/intrinsics \
  --squares-x 8 --squares-y 11 \
  --square-length 35.0 --marker-length 20.0 \
  --aruco-dict DICT_4X4_50 \
  --out calib.yaml
```
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
    p = argparse.ArgumentParser("Calibración de cámara con tablero ChArUco")
    p.add_argument("--img-dir", default="data/intrinsics",
                   help="Carpeta con *.jpg / *.png")

    # Parámetros del tablero (adaptados a tu medición)
    p.add_argument("--squares-x", type=int, default=8, help="Cuadros en X (horizontal)")
    p.add_argument("--squares-y", type=int, default=11, help="Cuadros en Y (vertical)")
    p.add_argument("--square-length", type=float, default=35.0,
                   help="Lado del cuadrado, en mm")
    p.add_argument("--marker-length", type=float, default=20.0,
                   help="Lado del marcador, en mm")

    p.add_argument("--aruco-dict", default="DICT_4X4_50",
                   help="Nombre del diccionario ArUco (ej. DICT_4X4_50, DICT_4X4_100, …)")

    # Salida
    p.add_argument("--out", type=str, default="",
                   help="Ruta donde guardar calib (yaml / npz); vacío = no guardar")
    return p.parse_args()

###############################################################################
# Utilidades de visualización                                                  #
###############################################################################

def review_image(img_bgr: np.ndarray, charuco_corners, charuco_ids, title: str) -> bool:
    """Muestra la imagen con las esquinas ChArUco y devuelve True (keep) / False."""
    vis = img_bgr.copy()
    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()               # Bloqueante; se cierra cuando el usuario cierra la ventana
    plt.close()

    resp = input("¿Mantener esta imagen? (Enter/y = sí, n = no) → ").strip().lower()
    return resp in ("", "y", "yes", "s", "si")

###############################################################################
# Main                                                                         #
###############################################################################

def main():
    args = parse_args()

    # Diccionario y tablero
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    except AttributeError:
        raise ValueError(f"Diccionario {args.aruco_dict} no existe en tu OpenCV.")

    board = cv2.aruco.CharucoBoard_create(
        squaresX=args.squares_x,
        squaresY=args.squares_y,
        squareLength=args.square_length,
        markerLength=args.marker_length,
        dictionary=aruco_dict)

    # Detector ArUco (API nueva u old)
    if hasattr(cv2.aruco, "DetectorParameters"):
        detector_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        detect = lambda gray: aruco_detector.detectMarkers(gray)
    else:
        detector_params = cv2.aruco.DetectorParameters_create()
        detect = lambda gray: cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

    # Cargar imágenes
    img_dir = Path(args.img_dir)
    img_paths = sorted(glob.glob(str(img_dir / "*.jpg")) +
                       glob.glob(str(img_dir / "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"No se hallaron imágenes en {img_dir}")

    print(f"[INFO] {len(img_paths)} imagen(es) encontradas en {img_dir}")

    all_charuco_corners, all_charuco_ids = [], []
    image_size = None

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] No se pudo leer {path}; se omite.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        corners, ids, _ = detect(gray)
        if ids is None or len(ids) < 4:
            print(f"[INFO] {Path(path).name}: solo {0 if ids is None else len(ids)} marcadores → omite.")
            continue

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)
        if retval < 4:
            print(f"[INFO] {Path(path).name}: {retval} esquinas charuco (<4) → omite.")
            continue

        if review_image(img, charuco_corners, charuco_ids, Path(path).name):
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f"  → Aceptada ({len(charuco_corners)} esquinas)")
        else:
            print("  → Descartada por el usuario")

    n_valid = len(all_charuco_corners)
    if n_valid < 1:
        raise RuntimeError("No hay suficientes imágenes válidas para calibrar.")

    print(f"[INFO] Calibrando con {n_valid} imágenes…")

    camera_matrix = np.eye(3, dtype=np.float64)
    dist_coeffs = np.zeros((5, 1))
    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_CB_FAST_CHECK
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    retval, camera_matrix, dist_coeffs, *_ = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=flags,
        criteria=criteria)

    print("\n=== RESULTADOS ===")
    print(f"RMS reproj error : {retval:.4f} px")
    print("Matriz intrínseca (K):\n", camera_matrix)
    print("Coeficientes de distorsión (k1 k2 p1 p2 k3):", dist_coeffs.ravel())

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() in {".yaml", ".yml"}:
            fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", camera_matrix)
            fs.write("dist_coeffs", dist_coeffs)
            fs.write("rms", retval)
            fs.release()
        else:
            np.savez(out_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rms=retval)
        print(f"[INFO] Guardado en {out_path}")

    print("[INFO] Terminado.")


if __name__ == "__main__":
    main()
