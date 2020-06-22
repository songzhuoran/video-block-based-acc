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
        self.and_img_str = None

    def edge_detection(self, img_path, video_name, img_name):

        img_name=re.sub('[.jpg]', '', img_name)
        img_name = img_name+".png"

        self.img_str = R2_DIR+video_name+"/"+img_name
        self.and_img_str = R1_DIR+video_name+"/"+img_name

        self.img_gray = self.greyed(img_path)
        self.smoothed = self.smooth(self.img_gray)
        self.dx, self.dy, self.gradMat, _ = self.gradients(self.smoothed) #dx,dy,gradMat 分别是x方向梯度、y方向梯度和梯度强度
        self.NMS_Mat = self.NMS(self.gradMat,self.dx, self.dy)
        self.img_final = self.double_threshold(self.NMS_Mat) 
        
        return self.img_final
    
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
                
        return dx, dy, M, theta
    
    # 非极大值抑制 NMS
    
    def NMS(self, M, dx, dy):
        
        d = np.copy(M)
        W, H = M.shape
        NMS = np.copy(d)
        NMS[0, :] = NMS[W-1, :] = NMS[:, 0] = NMS[:, H-1] = 0
        
        for i in range(1, W-1):
            for j in range(1, H-1):
                
                # 如果当前梯度为0，该点就不是边缘点
                if M[i, j] == 0:
                    NMS[i, j] = 0
                    
                else:
                    gradX = dx[i, j] # 当前点 x 方向导数
                    gradY = dy[i, j] # 当前点 y 方向导数
                    gradTemp = d[i, j] # 当前梯度点
                    
                    # 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
                    if np.abs(gradY) > np.abs(gradX):
                        weight = np.abs(gradX) / np.abs(gradY) # 权重
                        grad2 = d[i-1, j]
                        grad4 = d[i+1, j]
                        
                        # 如果 x, y 方向导数符号相同
                        # 像素点位置关系
                        # g1 g2
                        #    c
                        #    g4 g3
                        if gradX * gradY > 0:
                            grad1 = d[i-1, j-1]
                            grad3 = d[i+1, j+1]
                        
                        # 如果 x，y 方向导数符号相反
                        # 像素点位置关系
                        #    g2 g1
                        #    c
                        # g3 g4
                        else:
                            grad1 = d[i-1, j+1]
                            grad3 = d[i+1, j-1]
                    
                    # 如果 x 方向梯度值比较大
                    else:
                        weight = np.abs(gradY) / np.abs(gradX)
                        grad2 = d[i, j-1]
                        grad4 = d[i, j+1]
                        
                        # 如果 x, y 方向导数符号相同
                        # 像素点位置关系
                        #      g3
                        # g2 c g4
                        # g1
                        if gradX * gradY > 0:
                            grad1 = d[i+1, j-1]
                            grad3 = d[i-1, j+1]

                        
                        # 如果 x，y 方向导数符号相反
                        # 像素点位置关系
                        # g1
                        # g2 c g4
                        #      g3
                        else:
                            grad1 = d[i-1, j-1]
                            grad3 = d[i+1, j+1]
                            
                    # 利用 grad1-grad4 对梯度进行插值
                    gradTemp1 = weight * grad1 + (1 - weight) * grad2
                    gradTemp2 = weight * grad3 + (1 - weight) * grad4
                    
                    # 当前像素的梯度是局部的最大值，可能是边缘点
                    if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                        NMS[i, j] = gradTemp
                        
                    else:
                        # 不可能是边缘点
                        NMS[i, j] = 0
                        
        return NMS
    
    def double_threshold(self, NMS):
        
        W, H = NMS.shape
        DT = np.zeros([W, H])
        
        # 定义高低阈值
        TL = 0.2 * np.max(NMS)
        TH = 0.3 * np.max(NMS)
        # print(TL)
        # print(TH)
        # TL = 80
        # TH = 150
        
        for i in range(1, W-1):
            for j in range(1, H-1):
                if (NMS[i, j] < TL):
                    DT[i, j] = 0
                    
                elif (NMS[i, j] > TH):
                    DT[i, j] = 1
                    
                # 
                elif (NMS[i-1, j-1:j+1] < TH).any() or (NMS[i+1, j-1:j+1].any()
                        or (NMS[i, [j-1, j+1]] < TH).any()):
                    DT[i, j] = 1
                    
        cur_img = np.zeros((480,854))
        for i in range(1, W-1):
            for j in range(1, H-1):
                if DT[i,j]==1: #边缘点
                    cur_img[i+5,j+5] = 255

                    # if self.dx[i,j]>0 and self.dy[i,j]>0:
                    #     for x in range(i+5,int((i+5)/8)+8):
                    #         for y in range(j+5,int((j+5)/8)*8+8):
                    #             if int((i+5)/8)+8 <= 480 and int((j+5)/8)*8+8 <= 854:
                    #                 cur_img[x,y] = 255
                    # elif self.dx[i,j]>0 and self.dy[i,j]<0:
                    #     for x in range(i+5,int((i+5)/8)*8+8):
                    #         for y in range(int((j+5)/8)*8,j+5):
                    #             if int((i+5)/8)+8 <= 480 and j+5 <= 854:
                    #                 cur_img[x,y] = 255
                    # elif self.dx[i,j]<0 and self.dy[i,j]>0:
                    #     for x in range(int((i+5)/8)*8,i+5):
                    #         for y in range(j+5,int((j+5)/8)*8+8):
                    #             if i+5 <= 480 and int((j+5)/8)*8+8 <= 854:
                    #                 cur_img[x,y] = 255
                    # else:
                    #     for x in range(int((i+5)/8)*8,i+5):
                    #         for y in range(int((j+5)/8)*8,j+5):
                    #             if i+5 <= 480 and j+5 <= 854:
                    #                 cur_img[x,y] = 255


                    # if self.dx[i,j]<0 and self.dy[i,j]<0:
                    #     for x in range(i+5,int((i+5)/8)+8):
                    #         for y in range(j+5,int((j+5)/8)*8+8):
                    #             if int((i+5)/8)+8 <= 480 and int((j+5)/8)*8+8 <= 854:
                    #                 cur_img[x,y] = 255
                    # elif self.dx[i,j]<0 and self.dy[i,j]>0:
                    #     for x in range(i+5,int((i+5)/8)*8+8):
                    #         for y in range(int((j+5)/8)*8,j+5):
                    #             if int((i+5)/8)+8 <= 480 and j+5 <= 854:
                    #                 cur_img[x,y] = 255
                    # elif self.dx[i,j]>0 and self.dy[i,j]<0:
                    #     for x in range(int((i+5)/8)*8,i+5):
                    #         for y in range(j+5,int((j+5)/8)*8+8):
                    #             if i+5 <= 480 and int((j+5)/8)*8+8 <= 854:
                    #                 cur_img[x,y] = 255
                    # else:
                    #     for x in range(int((i+5)/8)*8,i+5):
                    #         for y in range(int((j+5)/8)*8,j+5):
                    #             if i+5 <= 480 and j+5 <= 854:
                    #                 cur_img[x,y] = 255
        
        cv2.imwrite(self.img_str,cur_img)


        return DT 


O_DIR="/home/songzhuoran/video/video-block-based-acc/davis2016/data/"
Map_DIR="/home/songzhuoran/video/video-block-based-acc/mapping_val/"
R1_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result_mapping/"
R2_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result_raw_img/"

video_names = os.listdir(Map_DIR)
for video_name in video_names:
    print(video_name)

    img_names=os.listdir(Map_DIR+video_name)
    for img_name in img_names:
        canny = Canny()
        # edges = canny.edge_detection(Map_DIR+video_name+"/"+img_name,video_name,img_name)

        img_name=re.sub('[.png]', '', img_name)
        img_name = img_name+".jpg"
        edges = canny.edge_detection(O_DIR+video_name+"/"+img_name,video_name,img_name)



