from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import utils
from dataloader import transforms
import os

train_transform_list = [transforms.ToPILImage(),
                        transforms.RandomContrast(),
                        transforms.RandomBrightness(),
                        # transforms.ToNumpyArray(),

                        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]
train_transform = transforms.Compose(train_transform_list)

def gen_error_colormap(sim, real):
    cols = np.array(
        [[0, 0.5, 49, 54, 149],
         [0.5 , 1, 69, 117, 180],
         [1, 2, 116, 173, 209],
         [2, 4, 171, 217, 233],
         [4, 8, 224, 243, 248],
         [8, 16, 254, 224, 144],
         [16, 32, 253, 174, 97],
         [32, 64, 244, 109, 67],
         [64, 128, 215, 48, 39],
         [128, 256, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.

    H, W = sim.shape
    error = np.abs(sim - real)

    error_image = np.zeros([H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]

    for i in range(cols.shape[0]):
        distance = 20
        error_image[:10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return error_image

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def histogram(data_filenames, data_dir, depth=False, label=False, transforms=False, gauss=False):
    lines = utils.read_text_lines(data_filenames)

    occur = dict()
    for line in lines:
        splits = line.split()
        mask = []
        left_img, right_img = splits[:2]
        sample = {}

        if(transforms):
            if 'left' not in sample:
                left =  os.path.join(data_dir, left_img)
                right = os.path.join(data_dir, right_img)
                sample['left'] = np.array(Image.open(left).convert('RGB')).astype(np.float32)
                sample['right'] =np.array(Image.open(right).convert('RGB')).astype(np.float32)
            sample = train_transform(sample)
            sample['left'] = np.array(sample['left'])
            sample['right'] = np.array(sample['right'])

            temp = Image.fromarray(sample['left'].astype('uint8'))
            left = np.array((temp.convert('L')).flatten())

        # output in each sample is np.array
        if(gauss):

            if 'left' not in sample:
                left =  os.path.join(data_dir, left_img)
                right = os.path.join(data_dir, right_img)
                sample['left'] = np.array(Image.open(left).convert('RGB')).astype(np.float32)
                sample['right'] =np.array(Image.open(right).convert('RGB')).astype(np.float32)

            if np.random.random() < 0.5:
                kernel = np.random.choice([3,5,7,9])
                sample['left'] = cv2.GaussianBlur(np.array(sample['left']),(kernel, kernel),0)
                sample['right'] = cv2.GaussianBlur(np.array(sample['right']),(kernel, kernel),0)
            sample['left'] = np.array(sample['left'])
            sample['right'] = np.array(sample['right'])

            temp = Image.fromarray(sample['left'].astype('uint8'))
            left = np.array((temp.convert('L')).flatten())

        if(label):
            if 'left' not in sample:
                left =  os.path.join(data_dir, left_img)
                right = os.path.join(data_dir, right_img)
                sample['left'] = np.array(Image.open(left).convert('RGB')).astype(np.float32)
                sample['right'] =np.array(Image.open(right).convert('RGB')).astype(np.float32)
            temp = Image.fromarray(sample['left'].astype('uint8'))
            left = np.array(temp.convert('L'))

            mask = np.array(Image.open(os.path.join(data_dir, splits[4])).resize((960,540), resample=Image.NEAREST))
            left[mask>=17] = 0
            mask  = left>0.
            left = left[mask]

        if(depth):
            if 'left' not in sample:
                left =  os.path.join(data_dir, left_img)
                right = os.path.join(data_dir, right_img)
                sample['left'] = np.array(Image.open(left).convert('RGB')).astype(np.float32)
                sample['right'] =np.array(Image.open(right).convert('RGB')).astype(np.float32)
            #depthR so use right image
            temp = Image.fromarray(sample['right'].astype('uint8'))
            left = np.array(temp.convert('L'))

            mask = np.array(Image.open(os.path.join(data_dir, splits[2])))
            temp = (mask>0.) & (mask<2000.)
            left = left[temp]

        if(not depth and not label and not transforms and not gauss):
            left = np.array(Image.open(left).convert('L')).flatten()

        unique, counts = np.unique(left, return_counts=True)
        temp = dict(zip(unique, counts))
        occur = merge_two_dicts(occur, temp)

    return occur

if __name__ == "__main__":
    # difference between two images
    a = np.array(Image.open('pictures/real1.png'))
    b = np.array(Image.open('pictures/sim1.png').convert('L'))
    # b = cv2.GaussianBlur(a,(3, 3),0)

    c = gen_error_colormap(b, a)

    # plt.imshow(c)
    # plt.show()

    data_filenames = 'filenames/custom_test_sim.txt'
    data_dir = 'linked_sim_v9'

    # occur = histogram(data_filenames, data_dir)
    occur = {12: 2092, 13: 2033, 14: 1975, 15: 1896, 16: 2112, 17: 2647, 18: 2380, 19: 3159, 20: 8559, 21: 19214, 22: 2114, 23: 24582, 24: 20867, 25: 14437, 26: 30089, 27: 13310, 28: 24014, 29: 17857, 30: 23704, 31: 23303, 32: 17372, 33: 12505, 34: 17382, 35: 12590, 36: 9488, 37: 7977, 38: 6919, 39: 6651, 40: 6006, 41: 5532, 42: 5188, 43: 4705, 44: 4674, 45: 4159, 46: 4331, 47: 4105, 48: 3727, 49: 3646, 50: 3335, 51: 3143, 52: 3405, 53: 3260, 54: 3058, 55: 2958, 56: 2864, 57: 2629, 58: 2757, 59: 2485, 60: 2644, 61: 2561, 62: 2187, 63: 2516, 64: 2427, 65: 2306, 66: 2216, 67: 1911, 68: 1809, 69: 2140, 70: 2141, 71: 2080, 72: 2009, 73: 1979, 74: 1513, 75: 1862, 76: 1372, 77: 1791, 78: 1797, 79: 1298, 80: 1602, 81: 1618, 82: 1588, 83: 1149, 84: 1548, 85: 1052, 86: 1377, 87: 1409, 88: 1372, 89: 1341, 90: 1315, 91: 902, 92: 926, 93: 1208, 94: 1137, 95: 1176, 96: 1171, 97: 825, 98: 1065, 99: 1053, 100: 767, 101: 1079, 102: 751, 103: 983, 104: 987, 105: 928, 106: 909, 107: 867, 108: 630, 109: 656, 110: 878, 111: 800, 112: 806, 113: 827, 114: 622, 115: 746, 116: 822, 117: 604, 118: 805, 119: 580, 120: 744, 121: 756, 122: 773, 123: 764, 124: 423, 125: 418, 126: 742, 0: 2555, 1: 547, 3: 986, 7: 2489, 9: 2265, 11: 321, 127: 709, 129: 691, 131: 420, 135: 615, 137: 423, 139: 575, 141: 10094, 143: 590, 146: 598, 148: 1419, 150: 1320, 152: 514, 154: 917, 158: 445, 160: 464, 162: 446, 164: 436, 166: 457, 169: 434, 171: 339, 173: 399, 175: 387, 177: 246, 181: 452, 183: 180, 185: 5682, 187: 4979, 188: 173, 192: 1421, 194: 173, 196: 1373, 198: 143, 200: 1193, 204: 743, 206: 739, 208: 666, 209: 659, 211: 121, 215: 121, 217: 129, 219: 416, 221: 398, 223: 115, 227: 290, 229: 303, 230: 270, 232: 248, 234: 87, 238: 219, 240: 79, 242: 205, 244: 200, 246: 199, 248: 60, 251: 66, 253: 223, 255: 9762, 128: 710, 130: 625, 132: 596, 133: 386, 134: 634, 136: 607, 138: 584, 140: 618, 142: 10549, 144: 573, 145: 577, 147: 553, 149: 559, 151: 568, 153: 475, 155: 469, 156: 427, 157: 692, 159: 648, 161: 457, 163: 441, 165: 448, 167: 396, 168: 432, 170: 428, 172: 435, 174: 261, 176: 384, 178: 398, 179: 391, 180: 419, 182: 192, 184: 1241, 186: 10820, 189: 1963, 190: 151, 191: 1574, 193: 1371, 195: 1339, 197: 1333, 199: 160, 201: 1082, 202: 989, 203: 927, 205: 153, 207: 152, 210: 587, 212: 572, 213: 562, 214: 511, 216: 499, 218: 492, 220: 376, 222: 108, 224: 348, 225: 356, 226: 293, 228: 94, 231: 98, 8: 2464, 10: 290, 4: 1595, 5: 213, 6: 2323, 2: 195, 233: 226, 235: 224, 236: 229, 237: 223, 239: 65, 241: 225, 243: 232, 245: 54, 247: 201, 249: 222, 250: 221, 252: 199, 254: 190}
    plt.bar(list(occur.keys()), occur.values(), color='g')
    plt.show()
