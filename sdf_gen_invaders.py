import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from PIL import Image
import sys

img = Image.open(sys.argv[1])

# [-- Custom figure plotting with labels
fig = plt.figure(figsize=(16, 8))
def plot(img, index, show_text=True, cmap="coolwarm"):
    global fig
    ax = fig.add_subplot(2, 4, index)
    im = ax.imshow(img, cmap=cmap)
    if show_text:
        for (j,i),label in np.ndenumerate(img):
            ax.text(i,j,label,ha='center',va='center', color='limegreen')
            ax.text(i,j,label,ha='center',va='center', color='limegreen')
    ax.axis('off')
    return ax
# --]

def evaluate_parabolla(height, px, x):
    return (px-x)*(px-x)+height

def compute_rows(img):
    new_img = np.ones_like(img)
    rows, cols = img.shape
    for y in range(0, cols): # for each row
        for x in range(0, rows): # for each cell
            sdf_min = img[y,x]
            for px in range(0, rows): # for each parabolla on this row
                pheight = img[y,px]
                sdf_min = min(sdf_min, evaluate_parabolla(pheight, px, x))
            new_img[y,x] = sdf_min
    return new_img

def compute_euclidian_distance(img):
    img = compute_rows(img)
    img = img.transpose()
    img = compute_rows(img)
    img = img.transpose()
    return img

img = np.array([
    [0, 0, 0, 0 ,0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0 ,0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1 ,0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1 ,1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0 ,1, 1, 0, 1, 1, 0],
    [1, 0, 1, 1 ,1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1 ,1, 1, 1, 1, 0, 1],
    [0, 0, 1, 0 ,0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1 ,1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0 ,0, 0, 0, 0, 0, 0],
], np.float32)

def edt_encode(img):
    img = img.copy()
    img[img == 0] = np.inf
    img[img == 1] = 0
    return img

img_positive = edt_encode(img)
img_negative = edt_encode(1 - img)

edt_positive = compute_euclidian_distance(img_positive)
edt_negative = compute_euclidian_distance(img_negative)

# NOTE: Data would normally be stored as "squared" integers and normalised as floats
sdf = np.sqrt(edt_positive) - np.sqrt(edt_negative)

biggest_dim = max(np.amax(sdf), abs(np.amin(sdf)))
sdf_gray = sdf / biggest_dim # remap to [-1, 1] range
sdf_gray = (sdf_gray / 2.0) + 0.5 # remap to [0, 1]  range
sdf_gray = (sdf_gray * 255.0).astype(np.uint8)  # Convert to grayscale [0, 255]


# -- Plot all the things:

plot(img_positive, 1).set_title('Positive encoded image')
plot(img_negative, 2).set_title('Negative encoded image')

plot(edt_positive, 5).set_title('Positive euclidian distance: A')
plot(edt_negative, 6).set_title('Negative euclidian distance: B')

plot(sdf, 7).set_title('Normalized SDF: sqrt(A)-sqrt(B)')
plot(sdf_gray, 8, cmap="gray").set_title('SDF to grayscale')

sdf_zoomed = scipy.ndimage.zoom(sdf_gray, 8, order=1) # Bilinear resample for clean contour
ε = 3
ax = plot(sdf_zoomed, 4, cmap="gray", show_text=False)
ax.contourf(sdf_zoomed, [255-128-ε, 255-128+ε], colors=["red"])
ax.set_title('Grayscale /w bilinear contour y=128')


plt.tight_layout()
plt.savefig("Figure.png")
plt.show()
