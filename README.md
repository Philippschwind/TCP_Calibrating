# TCP_Calibrating
Uni Project: Calibrates the TCP for the robot via ArUco markers

# calibrate_camera.py
Findet die Kameramatrix und Verzerrungsparameter heraus.
Speichert diese in camera_params.json

Anwendung:
1. Schachbrettmuster finden/drucken
2. Bilder (ca 10 Stück) von Schachbrettmuster machen
3. Bilder in Ordner: 'calib_images' speichern
4. calibrate_camera.py ausführen 

# initialize_reference_position.py
Speichert die Referenzpostion.
Ausführen beim ersten Aufbau, wenn Roboter programmiert wird

Anwendung
1. ArUco Marker anbringen
2. 2 Bilder an vordefinierten Positionen aufnehmen
3. Bilder in Ordner: 'reference_images' speichern
4. initialize_reference_position.py ausführen

# calculate_tcp_offset.py
Berechnet zuerst den Offset zwischen Aufgenommenen Bildern und Referenzposition.
Ermittelt dann den TCP Offset basierend darauf

Anwendung
1. 2 Bilder an vorderfinierten Positionen aufnehmen (gleich wie Referenz)
2. Bilder in Ordner: 'position_images' speichern
3. calculate_tcp_offset.py ausführen
4. TCP offset auslesen und an Roboter übergeben

