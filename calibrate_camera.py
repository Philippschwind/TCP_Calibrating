# Erstellt anhand von https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import json
import cv2
import numpy as np
import glob


def calibrate_camera(chessboard_size=(9,6), image_folder='calib_images/*.jpg'):
    # 3D Punkte im Weltkoordinatensystem vorbereiten
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3D Punkte in Weltkoordinaten
    imgpoints = []  # 2D Punkte in Bildkoordinaten

    images = glob.glob(image_folder)
    if len(images) == 0:
        print("Keine Bilder gefunden. Lege Schachbrettbilder in 'calib_images/' ab.")
        return None, None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ecken finden
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Gefundene Ecken anzeigen
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Ecken gefunden', img)
            cv2.waitKey(200)

    cv2.destroyAllWindows()

    # Kalibrierung durchführen
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n=== Ergebnisse ===")
    print("Kameramatrix:\n", mtx)
    print("Verzerrungskoeffizienten:\n", dist.ravel())

    return mtx, dist


if __name__ == "__main__":
    # Kamera kalibrieren
    # TODO: Chessboard size anpassen
    mtx, dist = calibrate_camera(chessboard_size=(9,6), image_folder='calib_images/*.jpg')
    params = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist()
    }
    with open("camera_params.json", "w") as f:
        json.dump(params, f, indent=4)
