
import cv2
import numpy as np
import sys


# x = 18

# a = [13*x+7, 7*x+12, 9*x+1, 11*x+2]
# print(a)
import numpy as np
   
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
a = gkern()
b = np.ones((5,5))
print(a*b	)