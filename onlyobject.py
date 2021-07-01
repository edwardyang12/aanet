import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

if __name__ == "__main__":
    depth = np.array(Image.open('126gt.png'))/ 1000
    pred = np.array(Image.open('126.png'))/1000

    label = Image.open('126label.png').resize((960,540))
    label = np.array(label)

    depth[label>=17] = 0
    pred[label>=17] = 0

    plt.imshow(depth)
    plt.imshow(pred)
    plt.show()
