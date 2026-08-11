import numpy as np
import json
from scipy.spatial.transform import Rotation

# Rotation matrices for Camera position relative to robot flange
R1 = np.array([[0,0,1],
              [0,1,0],
              [-1,0,0]])
R2 = np.array([[1,0,0],
              [0,0,1],
              [0,-1,0]])

R3 = np.array([[0,-1,0],
               [1,0,0],
               [0,0,1]])

R = R3 @ R2 @ R1

t = np.array([0,-0.13, -0.03])

CAMERA_OFFSET = np.eye(4, dtype=float)
CAMERA_OFFSET[:3, :3] = R
CAMERA_OFFSET[:3, 3] = t


def load_ur_transformations(json_file="positionen.json"):
    """
    Liest UR-Roboterpositionen aus JSON und erzeugt pro Position
    eine homogene 4x4-Transformationsmatrix.

    UR-Format:
    x, y, z  = Translation
    rx, ry, rz = Rotationsvektor, nicht Eulerwinkel
    """

    with open(json_file, "r", encoding="utf-8") as f:
        positions = json.load(f)

    transformations = []

    for name, p in positions.items():
        rotvec = np.array([p["rx"], p["ry"], p["rz"]], dtype=float)

        R = Rotation.from_rotvec(rotvec).as_matrix()

        # Translation von mm -> m
        t = np.array([p["x"], p["y"], p["z"]], dtype=float) / 1000.0

        # Homogene Transformationsmatrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        T_cam = T @np.linalg.inv( CAMERA_OFFSET ) # Transform to camera frame

        transformations.append(T_cam)

    return transformations


def load_reference_transformations(json_file="reference_positions.json"):
    """
    Liest die Referenz-Transformationsmatrizen aus einer JSON-Datei.

    Returns
    -------
    list[np.ndarray]
        Liste von 4x4-Transformationsmatrizen.
    """

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    reference_transformations = [
        np.array(T, dtype=float)
        for T in data["Referenze_Positions"]
    ]

    return reference_transformations