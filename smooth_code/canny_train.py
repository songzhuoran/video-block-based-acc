import csv
import os
import re
import cv2
import numpy as np

O_DIR="/home/songzhuoran/video/video-block-based-acc/davis2016/data/"
Map_DIR="/home/songzhuoran/video/video-block-based-acc/mapping-val/"
R1_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result_and_val/"
R2_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result/"
IDX_DIR="/home/songzhuoran/video/video-block-based-acc/idx/"
RES_DIR="/home/songzhuoran/video/video-block-based-acc/residual/"




bflist = []  # aka b frame list
pflist = []  # aka b frame list

video_names = os.listdir(Map_DIR)
for video_name in video_names:
    print(video_name)
    with open(RES_DIR+video_name+".csv","r") as file:
        reader = csv.reader(file)
        data = []
        for item in reader:
            data.append(item)

    img_names=os.listdir(Map_DIR+video_name)
    for img_name in img_names:
        img1 = cv2.imread(Map_DIR+video_name+"/"+img_name,0)
        img_name=re.sub('[.png]', '', img_name)
        fcnt = int(img_name)
        img_name = img_name+".jpg"
        img2 = cv2.imread(O_DIR+video_name+"/"+img_name,0)
        img_name=re.sub('[.jpg]', '', img_name)
        img_name = img_name+".png"
        img3=cv2.Canny(img1,80,150) # image after canny
        img4=cv2.Canny(img2,80,150) # image after canny


        t_img2=np.zeros([480,854])
        for i in range(img3.shape[0]):
            for j in range(img3.shape[1]):
                if img3[i][j]!=0 and img4[i][j]!=0:
                    t_img2[i][j]=255
                else:
                    t_img2[i][j]=0
        cv2.imwrite(R1_DIR+video_name+"/"+img_name,t_img2)

    # img_names=os.listdir(Map_DIR+video_name)
    # for img_name in img_names:
    #     img1 = cv2.imread(Map_DIR+video_name+"/"+img_name,0)
    #     img2 = cv2.imread(R1_DIR+video_name+"/"+img_name,0)

    #     t_img2=np.zeros([480,854])
    #     for i in range(img1.shape[0]):
    #         for j in range(img1.shape[1]):
    #             if img1[i][j]!=0 or img2[i][j]!=0:
    #                 t_img2[i][j]=255
    #             else:
    #                 t_img2[i][j]=0
    #     cv2.imwrite(R2_DIR+video_name+"/"+img_name,t_img2)
