import os
import numpy as np
import cv2

B_DIR="/home/songzhuoran/video/video-block-based-acc/lossless/data/val/bframe/"
R_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result/"

video_names=os.listdir(B_DIR)
for video_name in video_names:
    img_names=os.listdir(B_DIR+video_name)
    for img_name in img_names:
        img1 = cv2.imread(B_DIR+video_name+"/"+img_name,0)
        img2=cv2.Canny(img1,80,150)
        t_img=np.zeros([480,854])
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if img1[i][j]!=0 and img2[i][j]!=0:
                    t_img[i][j]=255
                else:
                    t_img[i][j]=0

        t_img2=np.zeros([480,854])
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if img1[i][j]!=0 or t_img[i][j]!=0:
                    t_img2[i][j]=255
                else:
                    t_img2[i][j]=0
        
        cv2.imwrite(R_DIR+video_name+"/"+img_name,t_img2)


