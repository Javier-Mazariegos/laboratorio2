# Tarea 3 Speedup - Laboratorio 2
# Javier Mazariegos
# Lorena Perez

import cv2 as cv
import numpy as np
from external import sharpen_img_cython
import time 

"""
Calculamos el speedup de la tarea 3 con respecto a la tarea 2 usando el video de prueba video_speed.mp4

Speedup = 𝑡 𝑠ℎ𝑎𝑟p / t 𝑓𝑎𝑠𝑡−𝑠ℎ𝑎𝑟p

Speedup = 195.77 / 25.38 = 7.71

"""

def sharpen_img(imagen, kernel):
    alto, ancho, _ = imagen.shape
    tam_kernel = kernel.shape[0]

    imagen_conv = np.zeros((alto, ancho, 3))

    for y in range(alto - tam_kernel + 1):
        for x in range(ancho - tam_kernel + 1):
            for c in range(3):
                conv_value = (imagen[y:y + tam_kernel, x:x + tam_kernel, c] * kernel).sum()
                
                if conv_value <= 0:
                    imagen_conv[y, x, c] = 0
                elif conv_value >= 255:
                    imagen_conv[y, x, c] = 255
                else:
                    imagen_conv[y, x, c] = conv_value

    return imagen_conv.astype(np.uint8)

kernel = np.array([[-1, -1, -1],
                [-1, 9.5, -1],
                [-1, -1, -1]])



def noThreadingcython(source):
    cap = cv.VideoCapture(device_id)

    if (cap.isOpened() == False):
        print("Video capture failed to open")
    while True:
        ret, im_rgb = cap.read()
        if ret:
            a7 = sharpen_img_cython(im_rgb, kernel)
        else:
            break

def noThreading(source):
    cap = cv.VideoCapture(device_id)

    if (cap.isOpened() == False):
        print("Video capture failed to open")
    while True:
        ret, im_rgb = cap.read()
        if ret:
            a7 = sharpen_img(im_rgb, kernel)
        else:
            break

device_id = "./video/video_speed.mp4"
start = time.time()
noThreading(device_id)
end = time.time()
print("Tiempo python: ", end - start)
start = time.time()
noThreadingcython(device_id)
end = time.time()
print("Tiempo cython: ", end - start)
