import csv
import os
import re
import cv2
import numpy as np

def main():
    mvspath = "/home/songzhuoran/video/video-block-based-acc/mvs/"
    videopath = "/home/songzhuoran/video/video-block-based-acc/davis2016/data/"
    residualpath = "/home/songzhuoran/video/video-block-based-acc/residual_img/"
    videofloders = os.listdir(residualpath)
    for videofloder in videofloders:
        resfiles= os.listdir(residualpath + videofloder)
        for resfile in resfiles:
            resimg_str = residualpath + videofloder + "/" + resfile
            print(resimg_str)
            resimg = cv2.imread(resimg_str)
            for c in range(3):
                for i in range(480):
                    for j in range(854):
                        resimg[i][j][c] = 0
            cv2.imwrite(resimg_str,resimg)
            



if __name__ == "__main__":
    main()
