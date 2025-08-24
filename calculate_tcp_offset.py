# Calculates offset between current pose and reference
# Calculates TCP offset based on that

import numpy as np
import cv2
import cv2.aruco as aruco
import json
import glob

# image folder
image_folder = "reference_pictures"

# Camera params -- Saved in json file
with open("camera_params.json", "r") as f:
    my_dict = json.load(f)

camera_matrix = my_dict['camera_matrix']
dist_coeffs = my_dict['dist_coeff']

# Reference Positions
with open("reference_postions.json", "r") as f:
    reference_dict = json.load(f)

T_ref_1 = reference_dict['T_ref_1']
T_ref_2 = reference_dict['T_ref_2']

# ArUco Marker 
ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_length = 0.05  # meters (set this to your printed marker size)


def get_pose_from_marker(gray):
    corners, ids, _ = aruco.detectMarkers(gray, ar_dict)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        # Just take the first detected marker
        rvec, tvec = rvecs[0], tvecs[0]

        # Convert rvec/tvec to homogeneous 4x4 matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        return T
    return None

def main():

    poses = []
    images = glob.glob(image_folder)
    if len(images) == 0:
        print("Keine Bilder gefunden. Lege Schachbrettbilder in 'reference_imgs/' ab.")

    # Get reference positions
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        T = get_pose_from_marker(gray)

        if not T:
            print("Pose konnte nicht bestimmt werden")
            return
        
        poses.append(T)

    T_offset_1 = np.linalg.inv(T_ref_1) @ poses[0]
    T_offset_2 = np.linalg.inv(T_ref_2) @ poses[1]

    #TODO: TCP Offset berechnen