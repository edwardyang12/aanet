from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle

# in mm
def convert(original, intrinsic, baseline, save):
    # load the image
    image = Image.open(original)

    # convert image to numpy array
    data = np.asarray(image)

    new_image = np.copy(data)
    for i in range(len(new_image)):
        for j in range(len(new_image[i])):
            if(new_image[i][j]!=0):
                new_image[i][j] = (baseline*1000*np.sqrt(intrinsic[0][0]**2+intrinsic[1][1]**2))/new_image[i][j]
            else:
                new_image[i][j] = 0

    plt.imsave(save,new_image)


# intrinsic = np.array([[1387.095, 0.0, 960.0], [0.0, 1387.095, 540.0], [0.0, 0.0, 1.0]])
# convert('depthL_fromR.png',intrinsic,0.055,'disp.png')

# with open('meta.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data['intrinsic'])
#     print(data)
    # print(data['extrinsic_l'])
    # print(data['extrinsic_r'])
