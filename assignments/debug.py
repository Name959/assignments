import numpy as np

from dinov2_numpy import Dinov2Numpy
from preprocess_image import preprocess_dinov2

weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)

cat_pixel_values = preprocess_dinov2("./demo_data/cat.jpg")
cat_feat = vit(cat_pixel_values)

dog_pixel_values = preprocess_dinov2("./demo_data/dog.jpg")
dog_feat = vit(dog_pixel_values)

