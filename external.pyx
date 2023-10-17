import cython
import numpy as np
cimport cython
cimport numpy as np

@cython.boundscheck(False)
cpdef np.ndarray[np.uint8_t,ndim=3] sharpen_img_cython(np.ndarray[np.uint8_t,ndim=3] imagen, np.ndarray[np.float64_t,ndim=2] kernel):
    cdef int alto, ancho, tam_kernel, y, x, c, tam_kernel_c, i, j
    cdef float t, conv_value
    cdef np.ndarray[np.float64_t,ndim=3] imagen_conv
    cdef unsigned char[:,:] r
    
    alto = imagen.shape[0]
    ancho = imagen.shape[1]
    tam_kernel = kernel.shape[0]
    tam_kernel_c = kernel.shape[1]

    imagen_conv = np.zeros((alto, ancho, 3))

    for y in range(alto - tam_kernel + 1):
        for x in range(ancho - tam_kernel + 1):
            for c in range(3):
                r = imagen[y:y + tam_kernel, x:x + tam_kernel, c]

                t = 0
                for i in range(tam_kernel):
                    for j in range(tam_kernel_c):
                            t += r[i,j] * kernel[i,j]


                conv_value = t
                
                if conv_value <= 0:
                    imagen_conv[y, x, c] = 0
                elif conv_value >= 255:
                    imagen_conv[y, x, c] = 255
                else:
                    imagen_conv[y, x, c] = conv_value

    return imagen_conv.astype(np.uint8)