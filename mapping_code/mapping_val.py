import csv
import os
import re
from skimage import io,data
from PIL import Image, ImageDraw, ImageColor
import cv2

# MV: [CurrentFrame, TargetFrame, BlockWidth, BlockHeight, CurrentBlockX, CurBlockY, TargetX, TargetY]

import sys
import os
import csv
import cv2
import numpy as np

IDX_DIR="/home/songzhuoran/video/video-block-based-acc/idx/"
RES_DIR="/home/songzhuoran/video/video-block-based-acc/residual/"
B_OUT_DIR="/home/songzhuoran/video/video-block-based-acc/mapping_val/"
P_DIR="/home/songzhuoran/video/video-block-based-acc/favos2016/"

mvsmat = []
vis = [False] * 200
classname = "111"
frame_mat = np.zeros((200,480,854),dtype="uint8")


def bframe_gen_kernel(fcnt):
    # bframe_img = np.zeros((480,854),dtype="uint8")
    img_vis = np.zeros((480,854))

    curstr = '%05d' % fcnt
    curstr = B_OUT_DIR + classname + "/" + curstr + ".png"
    # print curstr
    bframe_img = cv2.imread(curstr,0) #init frame


    with open(RES_DIR+classname+".csv","r") as file:
        reader = csv.reader(file)
        data = []
        for item in reader:
            data.append(item)

        totalnum = 0
        mappingnum = 0
        # print fcnt


        for row in data:
            if int(row[0]) == fcnt:
                totalnum = totalnum + 1

                dst = frame_mat[int(row[1])]
                # print int(row[1])
                lengthx = int(row[2])
                lengthy = int(row[3])
                srcy = int(row[4]) # current frame, max 854
                srcx = int(row[5]) #max 480
                dsty = int(row[6])
                dstx = int(row[7])
                res1 = int(row[8])
                res2 = int(row[9])
                res3 = int(row[10])
                res = int((res1+res2+res3)/3)
                if res < 20:
                    mappingnum = mappingnum + 1
                    for i in range(lengthx):
                        for j in range(lengthy):
                            if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                                bframe_img[srcx+i][srcy+j] = 0
                            if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                                if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                                    if img_vis[srcx+i][srcy+j] == 0:
                                        # if bframe_img[srcx+i][srcy+j] != dst[dstx+i][dsty+j]:
                                        #     print "====================================="
                                        bframe_img[srcx+i][srcy+j] = dst[dstx+i][dsty+j]
                                    else :
                                        # print "----------------------------------------"
                                        bframe_img[srcx+i][srcy+j] = (int(dst[dstx+i][dsty+j]) + int(bframe_img[srcx+i][srcy+j])) / 2
                                    img_vis[srcx+i][srcy+j] += 1
    
    curstr = '%05d' % fcnt
    curstr = B_OUT_DIR + classname + "/" + curstr + ".png"
    cv2.imwrite(curstr,bframe_img)
    mappingratio = float(float(mappingnum)/float(totalnum))
    print classname + str(fcnt) + " ratio : " + str(mappingratio)
    


def DFS(fcnt):
    # print(vis)
    if fcnt!=0: #if it is not a I frame
        if vis[fcnt]:
            return True
        else:
            for i in mvsmat[fcnt]:
                DFS(i)
            bframe_gen_kernel(fcnt)
            curstr = '%05d' % fcnt
            curstr = B_OUT_DIR + classname + "/" + curstr + ".png"
            frame_mat[fcnt] = cv2.imread(curstr,0)
            vis[fcnt] = True
            return True
    else:
        curstr = '%05d' % fcnt
        curstr = B_OUT_DIR + classname + "/" + curstr + ".png"
        frame_mat[fcnt] = cv2.imread(curstr,0)
        vis[fcnt] = True
        return True
    

def bframe_gen():
    
    bflist = []  # aka b frame list
    pflist = []  # aka b frame list
    print classname
    with open(IDX_DIR+"b/"+classname, "r") as file:
        for row in file:
            bflist.append(int(row)-1)
    # print(bflist)

    with open(IDX_DIR+"p/"+classname, "r") as file:
        for row in file:
            pflist.append(int(row)-1)
    # print(pflist)

    framecnt = pflist[-1] + 1

    # for i in pflist:
    #     vis[i] = False
    #     curstr = '%05d' % i
    #     curstr = P_DIR + classname + "/" + curstr + ".png"
    #     print curstr
    #     frame_mat[i] = cv2.imread(curstr,0)

    for i in range(framecnt):
        mvsmat.append(set())

    with open(RES_DIR+classname+".csv","r") as file:
        datainfo = csv.reader(file)
        for row in datainfo:
            mvsmat[int(row[0])].add(int(row[1]))
    

    for i in range(framecnt):
        if i !=0:
            if not vis[i]:
                DFS(i)

    for i in range(framecnt):
        if not vis[i]:
            print("ERROR")


mvsfiles= os.listdir(P_DIR)
for filename in mvsfiles: # i.e., bear
    classname = filename
    vis = [False] * 200
    frame_mat = np.zeros((200,480,854),dtype="uint8")
    mvsmat = []
    bframe_gen()
    

# def main():

#     mvsfiles= os.listdir(P_DIR)
#     for filename in mvsfiles: # i.e., bear
#         classname = filename
#         bframe_gen()




# if __name__ == "__main__":
#     main()
