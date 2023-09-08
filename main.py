import numpy as np
from scipy import signal
from PIL import Image as image
import matplotlib.pyplot as plt

im = image.open('sample.png')
im = im.convert('L')
im = np.array(im)

def circular_conv(a,b):
    return signal.convolve2d(a,b,mode="full")

def scale_down(im):
    new_array = []
    for i in range(len(im)):
        row = im[i].tolist()
        new_array += [row[1::2]]
    new_array = new_array[1::2]
    return np.array(new_array)

def scale_up(im):
    n = int(im.shape[0]*2)
    new_array = []
    add_zeros = False
    for i in range(int(n/2)):
        new_row = []
        for  j in range(int(n/2)):
            new_row += [im[i,j], 0]
            
        new_array += [new_row]
        new_array += [[0] * n]
    return np.array(new_array)

#H -> Top left
#G1 -> Top right
#G2 -> Bottom left
#G3 -> Bottom right

def dwt2d(im, lvl):
    result = im
    
    n = im.shape[0]
    H = np.array([[0.5,0.5],[0.5,0.5]])
    G1 = np.array([[-0.5,-0.5],[0.5,0.5]])
    G2 = np.array([[-0.5,0.5],[-0.5,0.5]])
    G3 = np.array([[0.5,-0.5],[-0.5,0.5]])

    new_im = scale_down(circular_conv(im,H))
    tr = scale_down(circular_conv(im,G1))
    bl = scale_down(circular_conv(im,G2))
    br = scale_down(circular_conv(im,G3))

    im = new_im
    result = np.bmat([[new_im, tr], [bl, br]])
    
    for i in range(lvl-1):
        
        H = np.array([[0.5,0.5],[0.5,0.5]])
        G1 = np.array([[-0.5,-0.5],[0.5,0.5]])
        G2 = np.array([[-0.5,0.5],[-0.5,0.5]])
        G3 = np.array([[0.5,-0.5],[-0.5,0.5]])
        
        new_im = scale_down(circular_conv(im,H))
        tr = scale_down(circular_conv(im,G1))
        bl = scale_down(circular_conv(im,G2))
        br = scale_down(circular_conv(im,G3))

        im = new_im
        change = np.bmat([[new_im, tr], [bl, br]])
        result[0:change.shape[0], 0:change.shape[0]] = change

    return result

def idwt2d(im, lvl):
    n = im.shape[0]

    dim = int(n / 2**lvl)
    tl = im[:dim, :dim]
    for i in range(lvl):
        
        tr = im[:dim, dim:int(dim*2)]
        bl = im[dim:int(dim*2), :dim]
        br = im[dim:int(dim*2), dim:int(dim*2)]

        #Scale up
        tl = scale_up(tl)
        tr = scale_up(tr)
        bl = scale_up(bl)
        
        br = scale_up(br)

        #Filter
        H = np.array([[0.5,0.5],[0.5,0.5]])
        G1 = np.array([[0.5,0.5],[-0.5,-0.5]])
        G2 = np.array([[0.5,-0.5],[0.5,-0.5]])
        G3 = np.array([[0.5,-0.5],[-0.5,0.5]])

        tl = circular_conv(tl,H)[:-1,:-1]
        tr = circular_conv(tr,G1)[:-1,:-1]
        bl = circular_conv(bl,G2)[:-1,:-1]
        br = circular_conv(br,G3)[:-1,:-1]


        tl = tl + tr + bl + br
        dim *= (2)
        
    return tl
