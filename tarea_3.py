# Tarea 1 - Laboratorio 2
# Javier Mazariegos
# Lorena Perez

import cv2 as cv
import numpy as np
from external import sharpen_img_cython

"""
Calculamos el speedup de la tarea 3 con respecto a la tarea 2 usando el video de prueba video_speed.mp4

Speedup = 𝑡 𝑠ℎ𝑎𝑟p / t 𝑓𝑎𝑠𝑡−𝑠ℎ𝑎𝑟p

Speedup = 195.77 / 25.38 = 7.71

"""


kernel = np.array([[-1, -1, -1],
                [-1, 9.5, -1],
                [-1, -1, -1]])

device_id = 0
cap = cv.VideoCapture(device_id)

if (cap.isOpened() == False):
    print("Video capture failed to open")
depth = -1
while True:
    ret, im_rgb = cap.read()
    if ret:
        a7 = sharpen_img_cython(im_rgb, kernel)
        win0 = 'Original'
        win1 = 'Processed'

        r,c = im_rgb.shape[0:2]
        resize_factor = 1

        R = int(r//resize_factor)
        C = int(c//resize_factor)
        win_size = (C, R) 

        cv.namedWindow(win0, cv.WINDOW_NORMAL)
        cv.namedWindow(win1, cv.WINDOW_NORMAL)

        cv.resizeWindow(win0, (win_size[0]//2,win_size[1]//2))
        cv.resizeWindow(win1, win_size)

        cv.imshow(win0, im_rgb)
        cv.imshow(win1, a7)
	
        # align windows        
        cv.moveWindow(win1, 0, 0)
        cv.moveWindow(win0, C, 0)
        
        # exit with q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#clean up before exit
cap.release()
cv.destroyAllWindows()