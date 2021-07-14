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
            left = np.array(Image.open(temp).convert('L'))

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
            left = np.array(Image.open(temp).convert('L'))

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
    occur = {4: 34366, 5: 12517, 6: 5326, 7: 9432, 8: 21294, 9: 17800, 10: 7202, 11: 7633, 12: 10934, 13: 13621, 14: 16047, 15: 17516, 16: 20008, 17: 21061, 18: 19904, 19: 17729, 20: 15531, 21: 12881, 22: 11836, 23: 11626, 24: 10829, 25: 10063, 26: 9875, 27: 9784, 28: 9674, 29: 9470, 30: 9037, 31: 8399, 32: 8094, 33: 8389, 34: 8064, 35: 8196, 36: 8153, 37: 7637, 38: 6834, 39: 6526, 40: 5802, 41: 5128, 42: 4736, 43: 4340, 44: 4252, 45: 3938, 46: 3326, 47: 3133, 48: 2852, 49: 2665, 50: 2396, 51: 2365, 52: 2208, 53: 2033, 54: 1881, 55: 1805, 56: 1732, 57: 1544, 58: 1455, 59: 1418, 60: 1342, 61: 1220, 62: 1124, 63: 1085, 64: 1007, 65: 952, 66: 886, 67: 836, 68: 726, 69: 666, 70: 591, 71: 545, 72: 497, 73: 448, 74: 428, 75: 342, 76: 308, 77: 270, 78: 235, 79: 205, 80: 192, 81: 151, 82: 137, 83: 150, 84: 98, 85: 84, 86: 91, 87: 75, 88: 82, 89: 67, 90: 63, 91: 52, 92: 59, 93: 53, 94: 41, 95: 45, 96: 32, 97: 40, 98: 38, 99: 35, 100: 33, 101: 28, 102: 31, 103: 31, 104: 34, 105: 22, 106: 26, 107: 28, 108: 24, 109: 24, 110: 20, 111: 8, 112: 13, 113: 20, 114: 19, 115: 11, 116: 11, 117: 16, 118: 15, 119: 18, 120: 9, 121: 15, 122: 16, 123: 17, 124: 9, 125: 6, 126: 14, 127: 11, 128: 9, 129: 10, 130: 8, 131: 8, 132: 8, 133: 11, 134: 8, 135: 7, 136: 16, 137: 8, 138: 7, 139: 10, 140: 7, 141: 9, 142: 6, 143: 9, 144: 2, 145: 5, 146: 6, 147: 3, 148: 8, 149: 6, 150: 6, 151: 11, 152: 5, 153: 4, 154: 7, 155: 5, 156: 9, 157: 3, 158: 3, 159: 3, 160: 1, 161: 4, 162: 3, 163: 9, 164: 3, 165: 1, 166: 2, 167: 1, 168: 2, 169: 1, 170: 3, 171: 1, 172: 1, 173: 5, 174: 1, 175: 7, 176: 2, 177: 4, 178: 7, 179: 1, 180: 4, 181: 1, 182: 2, 183: 2, 184: 1, 185: 1, 186: 1, 187: 1, 188: 1, 189: 3, 190: 2, 191: 2, 192: 2, 193: 2, 194: 4, 195: 2, 196: 1, 197: 4, 198: 3, 199: 5, 200: 3, 201: 1, 202: 2, 203: 3, 204: 3, 205: 1, 206: 1, 207: 3, 208: 3, 209: 1, 210: 1, 211: 4, 212: 5, 213: 1, 214: 4, 215: 1, 216: 2, 217: 2, 218: 1, 219: 3, 220: 3, 221: 4, 222: 1, 223: 1, 224: 2, 225: 2, 226: 2, 227: 1, 228: 1, 229: 2, 230: 2, 231: 1, 232: 1, 233: 2, 234: 1, 235: 1, 236: 1, 237: 1, 238: 1, 239: 1, 240: 1, 241: 1, 242: 1, 243: 2, 244: 1, 245: 2, 246: 1, 247: 1, 248: 1, 249: 1, 250: 8, 251: 2, 252: 11, 253: 2, 254: 1, 255: 11, 3: 43}
    plt.bar(list(occur.keys()), occur.values(), color='g')
    plt.show()
