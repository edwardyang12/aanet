import os
from glob import glob
from PIL import Image
from depth2disp import convert_file
import numpy as np

intrinsic = np.array([[1387.095, 0.0, 960.0], [0.0, 1387.095, 540.0], [0.0, 0.0, 1.0]])

def gen_kitti_2015():
    data_dir = 'data/KITTI/kitti_2015/data_scene_flow'

    train_file = 'KITTI_2015_train.txt'
    val_file = 'KITTI_2015_val.txt'

    # Split the training set with 4:1 raito (160 for training, 40 for validation)
    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f:
        dir_name = 'image_2'
        left_dir = os.path.join(data_dir, 'training', dir_name)
        left_imgs = sorted(glob(left_dir + '/*_10.png'))

        print('Number of images: %d' % len(left_imgs))

        for left_img in left_imgs:
            right_img = left_img.replace(dir_name, 'image_3')
            disp_path = left_img.replace(dir_name, 'disp_occ_0')

            img_id = int(os.path.basename(left_img).split('_')[0])

            if img_id % 5 == 0:
                val_f.write(left_img.replace(data_dir + '/', '') + ' ')
                val_f.write(right_img.replace(data_dir + '/', '') + ' ')
                val_f.write(disp_path.replace(data_dir + '/', '') + '\n')
            else:
                train_f.write(left_img.replace(data_dir + '/', '') + ' ')
                train_f.write(right_img.replace(data_dir + '/', '') + ' ')
                train_f.write(disp_path.replace(data_dir + '/', '') + '\n')

def gen_own_data():
    data_dir = 'linked_v5'
    train_list = 'linked_v5/training_lists/200_train.txt'
    val_list = 'linked_v5/training_lists/200_val.txt'


    train_list_f = open(train_list, 'r')
    val_list_f = open(val_list, 'r')

    train_file = 'filenames/custom_train.txt'
    val_file = 'filenames/custom_val.txt'

    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f:
        while True:
            x = train_list_f.readline() # gets each directory
            if not x:
                break
            else:
                x = x.split()[0]
            if x[0]!='1' and x[0]!='0':
                continue
            left = 'training/' +  x + '/0128_irL_denoised_half.png '
            right = 'training/' +  x + '/0128_irR_denoised_half.png '

            # os.mkdir('/cephfs/edward/'+x)
            # image = Image.open(data_dir + '/training/' +  x + '/depthL_fromR.png')
            # new_image = image.resize((960,540))
            # new_image.save('/cephfs/edward/'+x +'/depthL_fromR_down.png')
            # gt = '/cephfs/edward/'+x +'/depthL_fromR_down.png \n'

            temp = '/cephfs/edward/'+x +'/depthL_fromR_down.png'
            out = '/cephfs/edward/'+x +'/disp.png'
            convert_file(temp, intrinsic,0.055,out)
            gt = '/cephfs/edward/'+x +'/disp.png \n'


            train_f.write(left)
            train_f.write(right)
            train_f.write(gt)

        while True:
            x = val_list_f.readline() # gets each directory
            if not x:
                break
            else:
                x = x.split()[0]
            if x[0]!='1' and x[0]!='0':
                continue
            left = 'training/' +  x + '/0128_irL_denoised_half.png '
            right = 'training/' +  x + '/0128_irR_denoised_half.png '

            # image = Image.open(data_dir + '/training/' +  x + '/depthL_fromR.png')
            # new_image = image.resize((960,540))
            # os.mkdir('/cephfs/edward/'+x)
            # new_image.save('/cephfs/edward/'+x +'/depthL_fromR_down.png')
            # gt = '/cephfs/edward/'+x +'/depthL_fromR_down.png \n'

            temp = '/cephfs/edward/'+x +'/depthL_fromR_down.png'
            out = '/cephfs/edward/'+x +'/disp.png'
            convert_file(temp, intrinsic,0.055,out)
            gt = '/cephfs/edward/'+x +'/disp.png \n'

            val_f.write(left)
            val_f.write(right)
            val_f.write(gt)

def gen_own_data_full():
    data_dir = 'linked_v9'
    train_list = 'linked_v9/training_lists/all_train.txt'
    val_list = 'linked_v9/training_lists/all_val.txt'


    train_list_f = open(train_list, 'r')
    val_list_f = open(val_list, 'r')

    train_file = 'filenames/custom_train_full.txt'
    val_file = 'filenames/custom_val_full.txt'

    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f:
        while True:
            x = train_list_f.readline() # gets each directory
            if not x:
                break
            else:
                x = x.split()[0]
            if x[0]!='1' and x[0]!='0':
                continue
            left = 'training/' +  x + '/0128_irL_denoised_half.png '
            right = 'training/' +  x + '/0128_irR_denoised_half.png '
            # gt = 'training/' +  x + '/depthL_fromR.png \n'

            train_f.write(left)
            train_f.write(right)
            train_f.write(gt)

        while True:
            x = val_list_f.readline() # gets each directory
            if not x:
                break
            else:
                x = x.split()[0]
            if x[0]!='1' and x[0]!='0':
                continue
            left = 'training/' +  x + '/0128_irL_denoised_half.png '
            right = 'training/' +  x + '/0128_irR_denoised_half.png '
            gt = 'training/' +  x + '/depthL_fromR.png \n'
            val_f.write(left)
            val_f.write(right)
            val_f.write(gt)

if __name__ == '__main__':
    gen_own_data()
