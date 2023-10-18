# Tarea 1 - Laboratorio 2
# Javier Mazariegos
# Lorena Perez

import cv2 as cv
import numpy as np
#colour pencil sketch effect
def sharpen_img(imagen, kernel):
    """
    Aplica un filtro de Sharp Effect a una imagen utilizando un kernel de convolución.

    Esta función toma una imagen de entrada y un kernel de convolución y aplica un filtro deSharp Effect
    a la imagen. El filtro se aplica a cada canal de color por separado.

    Parameters:
    imagen (numpy.ndarray): Una matriz numpy que representa la imagen de entrada.
    kernel (numpy.ndarray): Una matriz numpy que representa el kernel de convolución para el filtro de mejora de nitidez.

    Returns:
        numpy.ndarray: Una nueva matriz numpy que representa la imagen con el filtro de mejora de nitidez aplicado.
    """
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
device_id = 0
cap = cv.VideoCapture(device_id)
if (cap.isOpened() == False):
    print("Video capture failed to open")
depth = -1
while True:
    ret, im_rgb = cap.read()
    if ret:
        a7 = sharpen_img(im_rgb, kernel)
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