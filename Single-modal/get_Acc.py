import os
import glob
import numpy as np
import cv2
import sys
import argparse
from tqdm import tqdm

def get_metric(args):

    labels = [{'name':'road', 'catId':0, 'color': (128, 64, 128)},
              {'name':'sidewalk', 'catId':1, 'color': (244, 35, 232)},
              {'name':'building', 'catId':2, 'color': (70, 70, 70)},
              {'name':'wall', 'catId':3, 'color': (102, 102, 156)},
              {'name':'fence', 'catId':4, 'color': (190, 153, 153)},
              {'name':'pole', 'catId':5, 'color': (153, 153, 153)},
              {'name':'traffic_light', 'catId':6, 'color': (250, 170, 30)},
              {'name':'traffic_sign', 'catId':7, 'color': (220, 220, 0)},
              {'name':'vegetation', 'catId':8, 'color': (107, 142, 35)},
              {'name':'terrain', 'catId':9, 'color': (152, 251, 152)},
              {'name':'sky', 'catId':10, 'color': (70, 130, 180)},
              {'name':'person', 'catId':11, 'color': (220, 20, 60)},
              {'name':'rider', 'catId':12, 'color': (255, 0, 0)},
              {'name':'car', 'catId':13, 'color': (0, 0, 142)},
              {'name':'truck', 'catId':14, 'color': (0, 0, 70)},
              {'name':'bus', 'catId':15, 'color': (0, 60, 100)},
              {'name':'train', 'catId':16, 'color': (0, 80, 100)},
              {'name':'motorcycle', 'catId':17, 'color': (0, 0, 230)},
              {'name':'bicycle', 'catId':18, 'color': (119, 11, 32)},
              {'name':'ignore', 'catId':19, 'color': (0, 0, 0)}]

    reals = glob.glob(args.gt_path+'/*.png')
    fakes = glob.glob(args.output_path+'/*.png')
    reals.extend(glob.glob(args.gt_path + '/*.jpg'))
    fakes.extend(glob.glob(args.output_path + '/*.jpg'))

    reals = sorted(reals)
    fakes = sorted(fakes)
    num_imgs = len(reals)

    CM = np.zeros((19,19), dtype=np.float32)
    # test
    for i in tqdm(range(num_imgs)):
        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])

        real = cv2.resize(real, (args.w, args.h), interpolation=cv2.INTER_NEAREST)

        pred = fake
        label = real

        label_dis = np.zeros((20, args.h, args.w), dtype=np.float32)
        pred_dis = np.zeros((20, args.h, args.w), dtype=np.float32)

        for j in range(20):
            color = labels[j]['color']
            label_diff = np.abs(label - color)
            pred_diff = np.abs(pred - color)

            label_diff = np.sum(label_diff, axis=2)
            pred_diff = np.sum(pred_diff, axis=2)

            label_dis[j,:,:] = label_diff
            pred_dis[j,:,:] = pred_diff

        label_id = np.argmin(label_dis, axis=0)
        pred_id = np.argmin(pred_dis, axis=0)

        for j in range(19):
            coord = np.where(label_id == j)
            pred_j = pred_id[coord]
            for k in range(19):
                CM[j,k] = CM[j, k] + np.sum(pred_j == k)


    total_pix_num = 128*256*500
    count = 0
    for i in range(19):
        count = count + CM[i, i]
    pix_acc = count / total_pix_num


    count = 0
    num_class=19
    for i in range(19):
        temp = CM[i, :]
        count = count + CM[i,i]/(np.sum(temp) + 1e-12)
    mean_acc = count/num_class

    count = 0
    for i in range(19):
        temp_0 = CM[i, :]
        temp_1 = CM[:, i]
        count = count + CM[i, i]/(np.sum(temp_0) + np.sum(temp_1) - CM[i, i] + 1e-6)

    mean_IoU = count/19

    print('Pix_Acc: {0}, class_Acc: {1}, mIoU: {2}'.format(pix_acc*100, mean_acc*100, mean_IoU))



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_path', type=str,
                        default='/home/chanyong/Desktop/code/CO2/CUT_code/datasets/cityscapes_d/gtFine/val/frankfurt/*color.png')
    parser.add_argument('--output_path', type=str,
                        default='/home/chanyong/Desktop/code/CO2/CUT_code/datasets/cityscapes_d/gtFine/val/frankfurt/*color.png')

    parser.add_argument('--h', type=int, default=128)
    parser.add_argument('--w', type=int, default=256)
    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    return args

def main():
    args = parse_args()
    get_metric(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

if __name__ == '__main__':
    main()