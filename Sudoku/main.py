import glob
import numpy as np
from functions import vision, support

images = np.asarray(vision.get_images())
img1 = vision.resize(images[0],500)
img2 = vision.resize(images[1],500)

vision.show_image('kuva1',img1)
vision.show_image('kuva2',img2)
