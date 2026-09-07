"""Traductor del alfabeto dactilologico de la Lengua de Senas Mexicana (LSM).

El nucleo (`types`, `features`, `segmentation`, `classifiers`) es codigo puro:
no importa OpenCV ni MediaPipe y no toca disco. Toda la I/O vive en `lsm.io` y
`lsm.cli`.
"""
