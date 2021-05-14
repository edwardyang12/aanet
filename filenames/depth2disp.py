from PIL import Image
import numpy as np
import pickle
from numpy import inf

# in mm
# intrinisc is matrix
# baseline is value
def convert_file(original, intrinsic, baseline, save):
    # load the image
    image = Image.open(original)

    # convert image to numpy array
    data = np.asarray(image)

    new_image = np.copy(data)
    new_image = (baseline*1000*intrinsic[0][0]/2)/new_image
    new_image[new_image== inf] = 0

    # save as 16 bit black and white image
    new_image = Image.fromarray(new_image.astype(np.uint16))
    new_image.save(save)


# intrinsic = np.array([[1387.095, 0.0, 960.0], [0.0, 1387.095, 540.0], [0.0, 0.0, 1.0]])
# convert('depthL_fromR.png',intrinsic,0.055,'disp.png')

# with open('meta.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data['intrinsic'])
#     print(data)
    # print(data['extrinsic_l'])
    # print(data['extrinsic_r'])
