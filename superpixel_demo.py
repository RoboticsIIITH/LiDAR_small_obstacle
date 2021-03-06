from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io as skimageIO
from segraph import create_graph
import numpy as np

image = img_as_float(skimageIO.imread("test_2.png"))
segments = slic(image, n_segments=500, sigma=1.0)
# Create graph of superpixels
vertices, edges = create_graph(segments)

# Compute centers:
gridx, gridy = np.mgrid[:segments.shape[0], :segments.shape[1]]
centers = dict()
for v in vertices:
    centers[v] = [gridy[segments == v].mean(), gridx[segments == v].mean()]
print(centers)