import cv2 as cv
import numpy as np
#colour pencil sketch effect
def sharp_effect(img, depth, kernel):
    img_new = img.copy()
    for fila in range(1,img.shape[0]-1):
        for columna in range(1,img.shape[1]-1):
            img_new[fila-1:fila +2,columna-1:columna+2] = img[fila-1:fila +2,columna-1:columna+2] * kernel
            img_new[fila,columna] = np.sum(img_new[:,:])
    print("entrandooooo")
    return img

kernel = np.array([[-1,-1,-1], [-1,9.5,-1], [-1,-1,-1]])
depth = -1


# get camera handle 
device_id = 0
# cap = cv.VideoCapture(device_id)
im = cv.imread("img.jpg")
new_im = np.pad(im, ((1,1), (1,1), (0, 0)), 'constant', constant_values=255)
a7 = sharp_effect(new_im, depth, kernel)
cv.imshow("img", a7)
# verify that video handle is open
# if (cap.isOpened() == False):
#     print("Video capture failed to open")
# # get frame, apply processing and show result
# kernel = np.array([[-1,-1,-1], [-1,9.5,-1], [-1,-1,-1]])
# depth = -1
# while True:
#     ret, im_rgb = cap.read()
#     im = im_rgb[:,:,:]
#     if ret:
        
#         new_im = np.pad(im, ((1,1), (1,1), (0, 0)), 'constant', constant_values=255)
#         #making the colour pencil sketch
#         a7 = sharp_effect(new_im, depth, kernel)
#         # create windows
#         win0 = 'Original'
#         win1 = 'Processed'

#         r,c = new_im.shape[0:2]
#         resize_factor = 1

#         R = int(r//resize_factor)
#         C = int(c//resize_factor)
#         win_size = (C, R) 

#         cv.namedWindow(win0, cv.WINDOW_NORMAL)
#         cv.namedWindow(win1, cv.WINDOW_NORMAL)

#         cv.resizeWindow(win0, (win_size[0]//2,win_size[1]//2))
#         cv.resizeWindow(win1, win_size)

#         cv.imshow(win0, new_im)
#         cv.imshow(win1, a7)
	
#         # align windows        
#         cv.moveWindow(win1, 0, 0)
#         cv.moveWindow(win0, C, 0)
        
#         # exit with q
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# #clean up before exit
# cap.release()
# cv.destroyAllWindows()