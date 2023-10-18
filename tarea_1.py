import cv2 as cv
#colour pencil sketch effect
def pencil_sketch_col(img):
    """
    Aplica un efecto de  Pencil Sketch Effect:Colour a una imagen.

    Esta función toma una imagen de entrada y utiliza la función `cv.pencilSketch` de OpenCV
    para crear un efecto de Pencil Sketch Effect:Colour. 

    Parameters:
    img (numpy.ndarray): Una matriz numpy que representa la imagen de entrada.

    Returns:
        numpy.ndarray: Una nueva matriz numpy que representa la imagen con el efecto  Pencil Sketch Effect:Colour.
    """
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_color



# get camera handle 
device_id = 0
cap = cv.VideoCapture(device_id)

# verify that video handle is open
if (cap.isOpened() == False):
    print("Video capture failed to open")
# get frame, apply processing and show result
while True:
    ret, im_rgb = cap.read()
    im = im_rgb[:,:,:]
    if ret:
        #making the colour pencil sketch
        a7 = pencil_sketch_col(im)
        # create windows
        win0 = 'Original'
        win1 = 'Processed'

        r,c = im.shape[0:2]
        resize_factor = 1

        R = int(r//resize_factor)
        C = int(c//resize_factor)
        win_size = (C, R) 

        cv.namedWindow(win0, cv.WINDOW_NORMAL)
        cv.namedWindow(win1, cv.WINDOW_NORMAL)

        cv.resizeWindow(win0, (win_size[0]//2,win_size[1]//2))
        cv.resizeWindow(win1, win_size)

        cv.imshow(win0, im)
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