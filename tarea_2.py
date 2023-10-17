import cv2 as cv
import numpy as np
#colour pencil sketch effect
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
depth = -1
# get camera handle 
device_id = 0
cap = cv.VideoCapture(device_id)
# a7 = sharpen_img(new_im, kernel)
# cv.imshow("img", a7)
# verify that video handle is open
if (cap.isOpened() == False):
    print("Video capture failed to open")
# get frame, apply processing and show result
depth = -1
while True:
    ret, im_rgb = cap.read()
    im = im_rgb[:,:,:]
    if ret:
        
        new_im = np.pad(im, ((1,1), (1,1), (0, 0)), 'constant', constant_values=255)
        #making the colour pencil sketch
        a7 = sharpen_img(new_im, kernel)
        # create windows
        win0 = 'Original'
        win1 = 'Processed'

        r,c = new_im.shape[0:2]
        resize_factor = 1

        R = int(r//resize_factor)
        C = int(c//resize_factor)
        win_size = (C, R) 

        cv.namedWindow(win0, cv.WINDOW_NORMAL)
        cv.namedWindow(win1, cv.WINDOW_NORMAL)

        cv.resizeWindow(win0, (win_size[0]//2,win_size[1]//2))
        cv.resizeWindow(win1, win_size)

        cv.imshow(win0, new_im)
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