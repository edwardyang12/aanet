import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

if __name__ == "__main__":
    depth = np.array(Image.open('126gt.png'))/ 1000
    pred = np.array(Image.open('126.png'))/1000

    print(np.unique(np.array(Image.open('126label.png'))))
    label = Image.open('126label.png').resize((960,540), resample=Image.NEAREST)
    label = np.array(label)
    for x in np.unique(label):
        print(x)
        temp = pred.copy()
        temp[label!=x] = 0

        plt.imshow(temp)
        plt.show()
