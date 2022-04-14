from math import *

LOAD_DATA = False;

SCALE = 2
#alpha is the inter-pyramid level scale factor.

ALPHA = 2**(1/3)  #2**(1/3)    1.25 

#NN param
K=3;

#W is patch width.
W=5;

PATCH_SIZE = W**2;

STEP = floor(W/2);

W_BLUR = 3;
# BLUR_KERNEL = fspecial('gaussian',W_BLUR );

# BLUR = reshape(BLUR_KERNEL',1,W_BLUR*W_BLUR);


# BLUR_DIST = ones(W)*3;
# BLUR_DIST(2:4,2:4) = 2;
# BLUR_DIST(3,3) = 1;
# BLUR_DIST = reshape(BLUR_DIST',1,W*W);

NUMCELLS = int(2*log(SCALE)/log(1.25) + 1);

MID = ceil(NUMCELLS/2);

EPSILON = exp(-15);

MIN_WH = 2*W;

MIN_STD = 0.1;

MAX_PREDS = K*W*W*SCALE;

DEFAULT_BG_GREYVAL = 0;

INTERP_METHOD = 'cubic';

INTERP_METHOD_DS='bilinear';

DIST_THRESH_EPS = 1;

print("NUMCELLS",NUMCELLS)
print("MID", MID)