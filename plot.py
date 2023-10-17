import cv2 as cv
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d    

import numpy as np

def hola():
    """
    Funcion de prueba xD
    """

    print("Hola :D")

def imgview(img, title=None, filename=None, axis=False, figsize=None):
    """
    imgview: funcion de visualizacion de imagen

    Par:
        img: matriz de la imagen a visualizar
        title: asignacion de titulo, por default no se coloca titulo
        filename: opcion para guardar la imagen, por default no se realiza la accion
        axis: visualizacion de los ejes, por default no se muestran
    """
    r,c = img.shape[0:2]
    if figsize != None:
        fig = plt.figure(figsize=figsize)
    else:
        k = 8
        fig = plt.figure(figsize=(k,k))
    ax = fig.add_subplot(111)
    
    if len(img.shape) == 3:
        img = ax.imshow(img,extent=None)
    else:
        img = ax.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255)
    if title != None:
        ax.set_title(title,fontsize=14)
    if not axis:
        plt.axis('off')
    else:
        ax.grid(c='w')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_xlabel('Columns',fontsize=14)
        ax.set_ylabel('Rows',fontsize=14)
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w',labelsize=14)
        ax.tick_params(axis='y', colors='w',labelsize=14)
        
    if filename != None:
        plt.savefig(filename)
    plt.show()

def splitrgb(img, filename=None):
    """
    splitrgb: funcion que separa y muestra los canales

    Par:
        img: matriz de la imagen a tratar
        filename: opcion para guardar, por default no se realiza la accion
    """
    fig = plt.figure(figsize=(10,10))

    if len(img.shape) == 3:
        channels = ['RGB', 'R', 'G', 'B'] # definiendo los canales
        channel_indices = [0, 1, 2, 3]  # y los indices para el for

        for i in channel_indices:
            if i == 0: # para visualizar la imagen RGB
                ax = fig.add_subplot(2, 2, i+1)
                ax.imshow(img)
                ax.set_title(channels[i])
                plt.axis('off')
            else:
                ax = fig.add_subplot(2, 2, i+1) # visualizacion por canal
                ax.imshow(img[:,:,i-1],cmap='gray', vmin=0, vmax=255)
                ax.set_title(channels[i])
                plt.axis('off')
    else:
        return "Imagen no RGB" 

    if filename != None:
        plt.savefig(filename)
    plt.show()

def hist_only(img, fill=False, filename=None):
    """
    hist_only: funcion que muestra solamente el histograma de distribucion de pixeles de la imagen recibida

    Par:
        img: matriz de la imagen a tratar
        fill: opcion de rellenado del histograma
        filename: opcion de guardado de la imagen
    """

    fig, ax = plt.subplots(figsize=(20,8))

    if len(img.shape) == 3:
        colors = ['r','g','b']
        for i, color in enumerate(colors):
            histr = cv.calcHist([img],[i],None,[256],[0,256])
            ax.plot(histr, c=color, alpha=0.9)
            x = np.arange(0.0, 256, 1)
            if fill:
                ax.fill_between(x, 0, histr.ravel(), alpha=0.2, color=color)
    else:
        histr = cv.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(histr, c='gray', alpha=0.9)
        x = np.arange(0.0, 256, 1)
    
        if fill:
            ax.fill_between(x, 0, histr.ravel(), alpha=0.2, color='gray')

    ax.set_xlim([0, 256])
    ax.grid(alpha=0.2)
    ax.set_facecolor('k')
    ax.set_title('Histogram', fontsize=k)
    ax.set_xlabel('Pixel value', fontsize=k)
    ax.set_ylabel('Pixel count', fontsize=k)

    if filename != None:
        plt.savefig(filename)
    
    plt.show()

def hist(img, title=None, filename=None, axis=False, fill=False):
    """
    hist: funcion que muestra la visualizacion de la imagen y el histograma a la par

    Par:
        img: matriz de la imagen a tratar
        title: opcion para agregar titulo
        filename: opcional para guardar la imagen
        axis: opcional para visualizar los ejes
        fill: opcional para rellenar el histograma
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5)) #subplots
    
    # axs[0] - primera imagen

    if len(img.shape) == 3:
        axs[0].imshow(img, extent=None)
    else:
        axs[0].imshow(img, extent=None, cmap='gray', vmin=0, vmax=255)
    if title != None:
        axs[0].set_title('RGB', fontsize=14)
    if not axis:
        axs[0].axis('off')
    else:
        axs[0].grid(c='w')
        axs[0].xaxis.tick_top()
        axs[0].xaxis.set_label_position('top') 
        axs[0].set_xlabel('Columns', fontsize=14)
        axs[0].set_ylabel('Rows', fontsize=14)
        axs[0].xaxis.label.set_color('w')
        axs[0].yaxis.label.set_color('w')
        axs[0].tick_params(axis='x', colors='w', labelsize=14)
        axs[0].tick_params(axis='y', colors='w', labelsize=14)

    ax = axs[1] # histograma

    if len(img.shape) == 3:
        colors = ['r', 'g', 'b']
        for i, color in enumerate(colors):
            histr = cv.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(histr, c=color, alpha=0.9)
            x = np.arange(0.0, 256, 1)
            if fill:
                ax.fill_between(x, 0, histr.ravel(), alpha=0.2, color=color)
    else:
        histr = cv.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(histr, c='gray', alpha=0.9)
        x = np.arange(0.0, 256, 1)
    
        if fill:
            ax.fill_between(x, 0, histr.ravel(), alpha=0.2, color='gray')

    ax.set_xlim([0, 256])
    ax.grid(alpha=0.2)
    ax.set_facecolor('k')
    ax.set_title('Histogram', fontsize=20)
    ax.set_xlabel('Pixel value', fontsize=20)
    ax.set_ylabel('Pixel count', fontsize=20)
    
    plt.tight_layout()

    if filename != None:
        plt.savefig(filename)

    plt.show()

def imgcmp(img1, img2, title1=None, title2=None, filename=None, axis=False):

    """
    imgcmp: funcion que permite comparar dos imagenes

    Par:
        img1: matriz de la primera imagen
        img2: matriz de la segunda imagen
        title1: title de la primera imagen
        title2: title de la segunda imagen
        filename: opcional para guardar
        axis: opcional para ver los ejes
    """
    imgs = [img1, img2]
    if isinstance(title2, str):
        titles = [title1, title2] # por si vienen separados para tenerlos en una lista
        # es decir title1="imagen1" y title2="imagen2"
    else:
        titles = title1 # por si vienen en una lista como ["imagen1", "imagen2"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(len(axs)):
        ax = axs[i]
        r, c = imgs[i].shape[0:2]
        if len(imgs[i].shape) == 3:
            im = ax.imshow(imgs[i], extent=None)
            if title1:
                ax.set_title(titles[i], fontsize=14)
        else:
            im = ax.imshow(imgs[i], extent=None, cmap='gray', vmin=0, vmax=255)
            if title1:
                ax.set_title(titles[i], fontsize=14)

        if not axis:
            ax.axis('off')
        else:
            ax.grid(c='w')
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top') 
            ax.set_xlabel('Columns', fontsize=14)
            ax.set_ylabel('Rows', fontsize=14)
            ax.xaxis.label.set_color('w')
            ax.yaxis.label.set_color('w')
            ax.tick_params(axis='x', colors='w', labelsize=14)
            ax.tick_params(axis='y', colors='w', labelsize=14)
        
    if filename is not None:
        plt.savefig(filename)
    plt.show()



# Add v1.1

def imgnorm(img):
    """Nomalize an image
    Args:
        img (numpy array): Source image
    Returns:
        normalized (numpy array): Nomalized image
    """
    vmin, vmax = img.min(), img.max()
    normalized_values = []
    delta = vmax-vmin

    for p in img.ravel():
        normalized_values.append(255*(p-vmin)/delta)

    normalized  = np.array(normalized_values).astype(np.uint8).reshape(img.shape[0],-1)
    return normalized

def imgcdf(img):
    """Compute the CDF on an image
    Args: 
        img (numpy array): Source image
    Returns:
        cdf (list): Computed CDf of img
        hist (list): Histogram of img
    """
    hist_list = cv.calcHist([img],[0],None,[256],[0,256])
    #hist = [(item) for sublist in hist_list for item in sublist]
    hist = hist_list.ravel()

    # define cdf placeholder
    cdf = []
    t = 0
    for p in hist:
        t += p
        cdf.append(t)
    return cdf, hist

def cdfview(cdf, hist, filename=None):
    """Plots the CDF and histogram in a shared x axis
    Args:
        cdf [list]: CDF value to compare
        hist [list]: Histogram to compare
        filename [string]: Optional filename to save output
    
    """
    fig, ax1 =plt.subplots(figsize=(12,8))
    ax2 = ax1.twinx()
    ax1.plot(hist, label='Value Prob',c='w', lw=0.9)
    ax2.plot(cdf, c='g', label='CDF')
    ax1.set_facecolor('k')
    ax1.set_title('Histogram Vs. CDF')
    ax2.legend()
    if filename != None:
        plt.savefig(filename)
    plt.show()

def imgeq(img):
    """ Equalize a grayscale image
    Args:
        img (numpy array): Grayscale image to equalize
    Returns:
        eq (numpy array): Equalized image
    """
    cdf = imgcdf(img)[0]
    cdf_eq = []
    n = img.shape[0] * img.shape[1]
    m = min(i for i in cdf if i > 0)

    for i in cdf:
        if i >= m:
            cdf_eq.append(int(round(255*(i-m)/(n-m))))
        else:
            cdf_eq.append(0)
    eq = cv.LUT(img, np.array(cdf_eq).astype(np.uint8))
    return eq

def rgb2gray(rgb):
    """ Convert an RGB image to grayscale
    Args:
        rgb (numpy array): RGB image
    Returns:
        gray (numpy array): Grayscale image
    """
    if len(rgb.shape) == 3:
        gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.144]).astype(np.uint8)
    else:
        gray=rgb
    return gray