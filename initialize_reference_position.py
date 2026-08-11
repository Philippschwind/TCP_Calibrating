# When setting up first use this script
# Reads out reference position from witch deviation will be calculated

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import glob


# image folder
image_folder = "reference_images/*.png"

# Camera params -- Saved in json file
with open("camera_params.json", "r") as f:
    my_dict = json.load(f)

camera_matrix = my_dict['camera_matrix']
dist_coeffs = my_dict['dist_coeff']

# ArUco Marker 
ar_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
marker_length = 0.03  # meters (set this to your printed marker size)
detector_params = cv2.aruco.DetectorParameters()

def get_pose_from_marker(gray):
    corners, ids, _ = aruco.detectMarkers(gray, ar_dict)

    if ids is not None:
        cm = np.array(camera_matrix)
        dc = np.array(dist_coeffs)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, cm, dc
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

    references = []
    images = glob.glob(image_folder)
    if len(images) == 0:
        print("Keine Bilder gefunden. Lege Schachbrettbilder in 'reference_imgs/' ab.")
    print(images)
    # Get reference positions
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        T_ref = get_pose_from_marker(gray)

        if not T_ref.any():
            print("Pose konnte nicht ermittelt werden")
            return

        references.append(T_ref)

    # Save references in json
    params = {
        'T_ref_1': references[0],
        'T_ref_2': references[1]
    }
    print(params)
    #with open("reference_postions.json", "w") as f:
    #    json.dump(params, f, indent=4)

    
if __name__ == "__main__":
    main()
