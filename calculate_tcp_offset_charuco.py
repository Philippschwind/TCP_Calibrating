# Calculates offset between current pose and reference
# Calculates TCP offset based on that

import numpy as np
import cv2
import cv2.aruco as aruco
import json
import glob
from scipy.spatial.transform import Rotation

from load_positions import load_reference_transformations, load_ur_transformations

# Image folder
image_folder = "position_images/*.png"

# Camera params -- saved in json file
with open("camera_params.json", "r") as f:
    my_dict = json.load(f)

camera_matrix = np.array(my_dict["camera_matrix"], dtype=np.float32)
dist_coeffs = np.array(my_dict["dist_coeff"], dtype=np.float32)

T_robs = load_ur_transformations()
T_refs = load_reference_transformations()


print(len(T_refs))

# ChArUco board settings
# Must match the printed board exactly
squares_x = 5
squares_y = 3
square_length = 0.04   # meters
marker_length = 0.03   # meters

# ChArUco dictionary
ar_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
detector_params = aruco.DetectorParameters()

# Create ChArUco board
board = aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    ar_dict
)


def get_pose_from_charuco(gray, min_corners=4):
    marker_corners, marker_ids, _ = aruco.detectMarkers(
        gray, ar_dict, parameters=detector_params
    )

    if marker_ids is None or len(marker_ids) == 0:
        return None

    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=gray,
        board=board
    )

    if ret is None or ret < min_corners or charuco_ids is None:
        return None

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


def calculate_base_correction(T_base_cam, T_cam_mark_new, T_cam_mark_ref):
    """_summary_

    Args:
        T_base_cam (_type_): _description_
        T_cam_mark_new (_type_): _description_
        T_cam_mark_ref (_type_): _description_
    """

    T_base_new_camera_old = T_cam_mark_new @ np.linalg.inv(T_cam_mark_ref)
    T_base_new_base_old =  T_base_cam @ T_base_new_camera_old @ np.linalg.inv(T_base_cam)

    return np.linalg.inv(T_base_new_base_old)


def average_offsets(offsets):
    translations = []
    rotations = []
    
    for T in offsets:
        translations.append(T[:3, 3])
        rotations.append(Rotation.from_matrix(T[:3, :3]).as_rotvec())
    
    # Translation mitteln
    t_mean = np.mean(translations, axis=0)
    r_mean = np.mean(rotations, axis=0)
    # Rotation mitteln
    R_mean = Rotation.from_rotvec(r_mean).as_matrix()
    print("Here")
    T_mean = np.eye(4)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean
    return T_mean

def transformation_to_readable_values(T):
    """
    Convert a 4x4 transformation matrix into readable translation and rotation values.

    Returns:
        dict with:
            dx, dy, dz         -> translation
            roll, pitch, yaw   -> rotation in degrees
    """

    T = np.array(T, dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError("T must be a 4x4 transformation matrix.")

    # Translation
    dx, dy, dz = T[:3, 3]

    # Rotation matrix
    R = T[:3, :3]

    # Convert rotation matrix to Euler angles
    # Convention: XYZ / roll-pitch-yaw
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    # Convert radians to degrees
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    rvec,_ = cv2.Rodrigues(R)
    rx, ry, rz = rvec.flatten()
    return {
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "rx": float(rx),
        "ry": float(ry),
        "rz": float(rz)
    }


def print_readable_transformation(T, translation_unit="m"):
    values = transformation_to_readable_values(T)

    print("\n=== Readable Transformation ===")
    print(f"dx:    {values['dx']:.6f} {translation_unit}")
    print(f"dy:    {values['dy']:.6f} {translation_unit}")
    print(f"dz:    {values['dz']:.6f} {translation_unit}")
    print(f"rx:  {values['rx']:.3f} deg")
    print(f"ry: {values['ry']:.3f} deg")
    print(f"rz:   {values['rz']:.3f} deg")


def main():
    poses_cam_new = []
    images = glob.glob(image_folder)

    if len(images) == 0:
        print("Keine Bilder gefunden. Lege Positionsbilder in 'position_images/' ab.")
        return

    print("Gefundene Bilder:", images)

    for fname in images:
        img = cv2.imread(fname)

        if img is None:
            print(f"Bild konnte nicht geladen werden: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        T = get_pose_from_charuco(gray)

        if T is None:
            print(f"Pose konnte nicht bestimmt werden: {fname}")
            continue

        poses_cam_new.append(T)
        print(f"Pose erfolgreich bestimmt für: {fname}")
        print(T)

    if len(poses_cam_new) == 0:
        print("Keine gültigen Posen erkannt.")
        return

    positions = []

    #for i, (T_rob, T_cam) in enumerate(zip(T_robs, poses_cam_new)):
     #   T_pos = T_rob @ T_cam
      #  print(f"\Robopose {i+1}:")
       # print(T_pos)
        #positions.append(T_pos)


    # Calculate offsets relative to one reference pose
    offsets = []

    for i, (T_rob, T_pos, T_ref) in enumerate(zip(T_robs,poses_cam_new, T_refs)):
        T_offset = calculate_base_correction(T_rob, T_pos, T_ref)
        print(f"\nOffset pos {i+1}:")
        print(T_offset)
        offsets.append(T_offset)
    print_readable_transformation(offsets[0])
    print_readable_transformation(offsets[1])
    print_readable_transformation(offsets[2])
    print("\n=== Gemittelter Offset ===")

    T_offset_avg = average_offsets(offsets)
    print(T_offset_avg)

    result = {
        "T_ref": T_ref.tolist(),
        "T_offset_avg": T_offset_avg.tolist()
    }

    with open("pose_offset.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nErgebnis wurde in 'pose_offset.json' gespeichert.")

    print_readable_transformation(T_offset_avg)
    


if __name__ == "__main__":
    main()