import cv2
import cv2.aruco as aruco
import numpy as np
import json
import glob

from load_positions import load_ur_transformations


# Image folder
image_folder = "reference_images/*.png"

# Camera params -- saved in json file
with open("camera_params.json", "r") as f:
    my_dict = json.load(f)

camera_matrix = np.array(my_dict["camera_matrix"], dtype=np.float32)
dist_coeffs = np.array(my_dict["dist_coeff"], dtype=np.float32)

# ChArUco board settings
# Adjust these values to match your printed board exactly
squares_x = 5
squares_y = 3
square_length = 0.04   # meters, e.g. 40 mm
marker_length = 0.03   # meters, e.g. 30 mm

# Dictionary
ar_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
detector_params = cv2.aruco.DetectorParameters()

# Create board
board = aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    ar_dict
)

def get_pose_from_charuco(gray, min_corners=4):
    # Detect ArUco markers
    marker_corners, marker_ids, _ = aruco.detectMarkers(
        gray, ar_dict, parameters=detector_params
    )

    if marker_ids is None or len(marker_ids) == 0:
        return None

    # Interpolate ChArUco corners
    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=gray,
        board=board
    )

    if ret is None or ret < min_corners or charuco_ids is None:
        return None

    # Estimate board pose
    success, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        rvec=None,
        tvec=None
    )

    if not success:
        return None

    # Convert rvec/tvec to homogeneous 4x4 matrix
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T


def average_transformations(transformations):
    """Average multiple 4x4 transformation matrices."""
    if not transformations:
        return None

    translations = []
    rvecs = []

    for T in transformations:
        R = T[:3, :3]
        t = T[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        translations.append(t)
        rvecs.append(rvec.flatten())

    t_avg = np.mean(np.array(translations, dtype=np.float64), axis=0)
    rvec_avg = np.mean(np.array(rvecs, dtype=np.float64), axis=0).reshape(3, 1)

    R_avg, _ = cv2.Rodrigues(rvec_avg)

    T_avg = np.eye(4, dtype=np.float32)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg

    return T_avg


def main():

    rob_transformations = load_ur_transformations("positionen.json")

    images = sorted(glob.glob(image_folder))

    if len(images) == 0:
        print("Keine Bilder gefunden. Lege Referenzbilder in 'reference_images/' ab.")
        return

    print("Gefundene Bilder:", images)

    poses = []

    for fname in images:
        img = cv2.imread(fname)

        if img is None:
            print(f"Bild konnte nicht geladen werden: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T_ref = get_pose_from_charuco(gray)

        if T_ref is None:
            print(f"Pose konnte nicht ermittelt werden: {fname}")
            continue

        poses.append(T_ref)
        print(f"Pose erfolgreich bestimmt für: {fname}")
        print(T_ref)

    if len(poses) == 0:
        print("Keine gültigen Posen erkannt.")
        return

    references = []
    for i, (T_rob, T_cam) in enumerate(zip(rob_transformations, poses)):
        T_ref = T_cam
        print(f"\nReferenzpose {i+1}:")
        print(T_ref)
        references.append(T_ref)

    params = {"Referenze_Positions": [T.tolist() for T in references]}

    print("\nGespeicherte Referenzen:")
    #print(json.dumps(params, indent=4))

    with open("reference_positions.json", "w") as f:
        json.dump(params, f, indent=4)

    print("\nReferenzpositionen wurden in 'reference_positions.json' gespeichert.")


if __name__ == "__main__":
    main()