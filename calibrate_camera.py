# Erstellt anhand von https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import json
import cv2
import numpy as np
import glob


def calibrate_camera(chessboard_size=(9,5), image_folder='calib_images/*.png'):
    # 3D Punkte im Weltkoordinatensystem vorbereiten
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3D Punkte in Weltkoordinaten
    imgpoints = []  # 2D Punkte in Bildkoordinaten

    images = glob.glob(image_folder)
    if len(images) == 0:
        print("Keine Bilder gefunden. Lege Schachbrettbilder in 'calib_images/' ab.")
        return None, None
    print("Anzahl Bilder: ", len(images))
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
        else:
            print("Kein Schachbrett")
    cv2.destroyAllWindows()

    # Kalibrierung durchführen
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n=== Ergebnisse ===")
    print("Kameramatrix:\n", mtx)
    print("Verzerrungskoeffizienten:\n", dist.ravel())

    return mtx, dist


def calibrate_camera_charuco(
    squares_x=5,
    squares_y=3,
    square_length=0.04,   # Kantenlänge eines Schachbrett-Quadrats in Metern
    marker_length=0.02,   # Kantenlänge des ArUco-Markers in Metern
    image_folder='calib_images/*.png',
    aruco_dict_name=cv2.aruco.DICT_5X5_1000,
    min_corners=4,
    show_detections=True
):
    # ArUco-Dictionary erstellen
    print(aruco_dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)

    # ChArUco-Board erstellen
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )

    detector_params = cv2.aruco.DetectorParameters()

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    images = glob.glob(image_folder)
    if len(images) == 0:
        print("Keine Bilder gefunden. Lege Kalibrierbilder in 'calib_images/' ab.")
        return None, None

    print("Anzahl Bilder:", len(images))

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Bild konnte nicht geladen werden: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        # ArUco-Marker erkennen
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )

        if ids is None or len(ids) == 0:
            print(f"Keine Marker gefunden in: {fname}")
            continue
        # ChArUco-Ecken interpolieren
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        if ret is not None and ret >= min_corners and charuco_ids is not None:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

            if show_detections:
                vis = img.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.aruco.drawDetectedCornersCharuco(
                    vis, charuco_corners, charuco_ids, (0, 255, 0)
                )
                cv2.imshow("ChArUco Ecken gefunden", vis)
                cv2.waitKey(200)

            print(f"OK: {fname} -> {len(charuco_ids)} ChArUco-Ecken")
        else:
            print(f"Zu wenige ChArUco-Ecken in: {fname}")

    cv2.destroyAllWindows()

    if len(all_charuco_corners) == 0:
        print("Keine brauchbaren ChArUco-Erkennungen gefunden.")
        return None, None

    # Kamera kalibrieren
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print("\n=== Ergebnisse ===")
    print("Reprojektionfehler:", ret)
    print("Kameramatrix:\n", camera_matrix)
    print("Verzerrungskoeffizienten:\n", dist_coeffs.ravel())

    return camera_matrix, dist_coeffs


if __name__ == "__main__":
    # Kamera kalibrieren
    mtx, dist = calibrate_camera_charuco(
        squares_x=5,
        squares_y=3,
        square_length=0.04,   # an dein gedrucktes Board anpassen
        marker_length=0.03,   # an dein gedrucktes Board anpassen
        image_folder='calib_images/*.png',
        aruco_dict_name=cv2.aruco.DICT_5X5_1000,
        min_corners=0
    )
    
    #mtx, dist = calibrate_camera(chessboard_size=(14,14), image_folder='calib_images/*.png')
    params = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist()
    }
    with open("camera_params.json", "w") as f:
        json.dump(params, f, indent=4)
