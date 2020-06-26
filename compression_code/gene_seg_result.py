import csv
import os
import re
import cv2
import numpy as np

def main():
    mvspath = "/home/songzhuoran/video/video-block-based-acc/mvs/"
    videopath = "/home/songzhuoran/video/video-block-based-acc/davis2016/annotation/"
    resultpath = "/home/songzhuoran/video/video-block-based-acc/seg_result/"
    # mvsfiles= os.listdir(residualpath)
    mvsfiles = {"bmx-trees","drift-straight","horsejump-high","kite-surf"}
    for filename in mvsfiles:
        videofloder = filename
        os.mkdir(resultpath+videofloder)
        videofile = videopath + videofloder  #open the video, i.e., /home/szr/video/block-mvs/davis2016/bear
        imgstrs = os.listdir(videofile)
        for imgstr in imgstrs:
            res_img = cv2.imread(videofile+"/"+imgstr)
            imgstr=re.sub('[.png]', '', imgstr)
            fd = resultpath+filename+"/"+imgstr+".npy"
            print(resultpath+filename+"/"+imgstr+".npy")
            seg_res = np.zeros((480,854,3))
            for i in range(3):
                for j in range(480):
                    for t in range(854):
                        seg_res[j][t][i] = res_img[j][t][i]            
            
            np.save(fd,seg_res)
            



if __name__ == "__main__":
    main()
