import csv
import os
import re
import cv2
import numpy as np
import math


class Canny():
    
    def __init__(self):
        
        self.img_gray = None
        self.smoothed = None
        self.dx = None
        self.dy = None
        self.gradMat = None
        self.NMS_Mat = None
        self.img_final = None
        self.img_str = None
        self.canny_img = None
        self.map_img_str = None

    def edge_detection(self, img_path, video_name, img_name, img2):

        img_name=re.sub('[.jpg]', '', img_name)
        img_name = img_name+".png"

        self.map_img_str = R1_DIR+video_name+"/"+img_name
        self.img_str = R2_DIR+video_name+"/"+img_name
        self.canny_img = img2

        self.img_gray = self.greyed(img_path)
        self.smoothed = self.smooth(self.img_gray)
        self.dx, self.dy, self.gradMat, _ = self.gradients(self.smoothed) #dx,dy,gradMat 分别是x方向梯度、y方向梯度和梯度强度


    
    # 灰度化
    def greyed(self, img_path):
        """
        Calculate function:
        Gray(i,j) = [R(i,j) + G(i,j) + B(i,j)] / 3
        or :
        Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
        """
        # 读取图片
        img = cv2.imread(img_path)
        # 转换成 RGB 格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 灰度化
        img_gray = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
        
        # return img[...,1]
        return img_gray
    
    # 去除噪音 - 使用 5x5 的高斯滤波器
    def smooth(self, img_gray):
        
        # 生成高斯滤波器
        """
        要生成一个 (2k+1)x(2k+1) 的高斯滤波器，滤波器的各个元素计算公式如下：
        
        H[i, j] = (1/(2*pi*sigma**2))*exp(-1/2*sigma**2((i-k-1)**2 + (j-k-1)**2))
        """
        sigma1 = sigma2 = 1.4
        gau_sum = 0
        gaussian = np.zeros([5, 5])
        for i in range(5):
            for j in range(5):
                gaussian[i, j] = math.exp((-1/(2*sigma1*sigma2))*(np.square(i-3) 
                                + np.square(j-3)))/(2*math.pi*sigma1*sigma2)
                gau_sum =  gau_sum + gaussian[i, j]
                
        # 归一化处理
        gaussian = gaussian / gau_sum
        
        # 高斯滤波
        W, H = img_gray.shape
        new_gray = np.zeros([W-5, H-5])
        
        for i in range(W-5):
            for j in range(H-5):
                new_gray[i, j] = np.sum(img_gray[i:i+5, j:j+5] * gaussian)
                
        return new_gray
    
    # 计算梯度幅值
    def gradients(self, new_gray):
        
        W, H = new_gray.shape
        dx = np.zeros([W-1, H-1])
        dy = np.zeros([W-1, H-1])
        M = np.zeros([W-1, H-1])
        theta = np.zeros([W-1, H-1])
        
        for i in range(W-1):
            for j in range(H-1):
                dx[i, j] = new_gray[i+1, j] - new_gray[i, j]
                dy[i, j] = new_gray[i, j+1] - new_gray[i, j]
                M[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j])) # 图像梯度幅值作为图像强度值
                theta[i, j] = math.atan(dx[i, j] / (dy[i, j] + 0.000000001)) # 计算 slope θ - artan(dx/dy)

        cur_img = np.zeros((480,854))
        for i in range(self.canny_img.shape[0]):
            for j in range(self.canny_img.shape[1]):
                if self.canny_img[i,j]!=0 and i>=3 and j>=3 and i<=476 and j<=850:
                    # if dx[i-3,j-3]<0 and dy[i-3,j-3]<0:
                    #     for x in range(i,int((i)/8)*8+8):
                    #         for y in range(j,int((j)/8)*8+8):
                    #             if x < 480 and y < 854:
                    #                 cur_img[x,y] = 255
                    # elif dx[i-3,j-3]<0 and dy[i-3,j-3]>0:
                    #     for x in range(i,int((i)/8)*8+8):
                    #         for y in range(int((j)/8)*8,j):
                    #             if x < 480 and y < 854:
                    #                 cur_img[x,y] = 255
                    # elif dx[i-3,j-3]>0 and dy[i-3,j-3]<0:
                    #     for x in range(int((i)/8)*8,i):
                    #         for y in range(j,int((j)/8)*8+8):
                    #             if x < 480 and y < 854:
                    #                 cur_img[x,y] = 255
                    # else:
                    #     for x in range(int((i)/8)*8,i):
                    #         for y in range(int((j)/8)*8,j):
                    #             if x < 480 and y < 854:
                    #                 cur_img[x,y] = 255

                    if dx[i-3,j-3]>0 and dy[i-3,j-3]>0:
                        for x in range(i,int((i)/4)*4+4):
                            for y in range(j,int((j)/4)*4+4):
                                if x < 480 and y < 854:
                                    cur_img[x,y] = 255
                    elif dx[i-3,j-3]>0 and dy[i-3,j-3]<0:
                        for x in range(i,int((i)/4)*4+4):
                            for y in range(int((j)/4)*4,j):
                                if x < 480 and y < 854:
                                    cur_img[x,y] = 255
                    elif dx[i-3,j-3]<0 and dy[i-3,j-3]>0:
                        for x in range(int((i)/4)*4,i):
                            for y in range(j,int((j)/4)*4+4):
                                if x < 480 and y < 854:
                                    cur_img[x,y] = 255
                    else:
                        for x in range(int((i)/4)*4,i):
                            for y in range(int((j)/4)*4,j):
                                if x < 480 and y < 854:
                                    cur_img[x,y] = 255

        cv2.imwrite(self.img_str,cur_img)
        # cv2.imwrite(self.map_img_str,cur_img)
                
        return dx, dy, M, theta
    




O_DIR="/home/songzhuoran/video/video-block-based-acc/davis2016/data/"
Map_DIR="/home/songzhuoran/video/video-block-based-acc/mapping_val/"
R1_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result_mapping/"
R2_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result_raw_img/"


video_names = os.listdir(Map_DIR)
for video_name in video_names:
    # video_name="blackswan"
    print(video_name)


    img_names=os.listdir(Map_DIR+video_name)
    for img_name in img_names:

        img_name=re.sub('[.png]', '', img_name)
        img_name = img_name+".jpg"
        img1 = cv2.imread(O_DIR+video_name+"/"+img_name,0)
        img2=cv2.Canny(img1,80,150) # image after canny
        canny = Canny()
        canny.edge_detection(O_DIR+video_name+"/"+img_name,video_name,img_name,img2)
        # img1 = cv2.imread(Map_DIR+video_name+"/"+img_name,0)
        # img2=cv2.Canny(img1,80,150) # image after canny
        # canny = Canny()
        # canny.edge_detection(Map_DIR+video_name+"/"+img_name,video_name,img_name,img2)

        


