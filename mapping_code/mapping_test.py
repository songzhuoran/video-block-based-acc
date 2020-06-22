import csv
import os
import re
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
B_OUT_DIR="/home/songzhuoran/video/video-block-based-acc/mapping_test/"
P_DIR="/home/songzhuoran/video/video-block-based-acc/favos2016/"
Bound_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result_raw_img/"
Bound2_DIR="/home/songzhuoran/video/video-block-based-acc/smooth_result_mapping/"

record_file = open("/home/songzhuoran/video/video-block-based-acc/result.csv", "a")

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
                    num_non_zero = 0
                    num_total = 0
                    for i in range(lengthx):
                        for j in range(lengthy):
                            if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                                num_total = num_total + 1
                                if dst[dstx+i][dsty+j] !=0:
                                    num_non_zero = num_non_zero + 1
                    if num_total!=0:
                        ratio_non_zero = float(num_non_zero/num_total)
                    else:
                        ratio_non_zero = 0.0
                    ######for fast object
                    # if ratio_non_zero >=0.5: 
                    if ratio_non_zero >=0.3:
                        # for i in range(lengthx):
                        #     for j in range(lengthy):
                        #         if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                        #             if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                        #                 if img_vis[srcx+i][srcy+j] == 0:
                        #                     bframe_img[srcx+i][srcy+j] = 0
                        #                 else :
                        #                     bframe_img[srcx+i][srcy+j] = (int(bframe_img[srcx+i][srcy+j])) / 2
                        #                 img_vis[srcx+i][srcy+j] += 1
                        for i in range(lengthx):
                            for j in range(lengthy):
                                if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                                    if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                                        if img_vis[srcx+i][srcy+j] == 0:
                                            bframe_img[srcx+i][srcy+j] = dst[dstx+i][dsty+j]
                                        else :
                                            bframe_img[srcx+i][srcy+j] = (int(dst[dstx+i][dsty+j]) + int(bframe_img[srcx+i][srcy+j])) / 2
                                        img_vis[srcx+i][srcy+j] += 1
                    ######for fast object
                    # elif ratio_non_zero >=0.3 and ratio_non_zero<0.5:
                    elif ratio_non_zero >=0.1 and ratio_non_zero<0.3:
                        bound_str = Bound_DIR + classname + "/" + '%05d' % int(row[1]) + ".png"
                        bound_img = cv2.imread(bound_str,0) #init frame
                        bound_str2 = Bound2_DIR + classname + "/" + '%05d' % int(row[1]) + ".png"
                        bound_img2 = cv2.imread(bound_str2,0) #init frame
                        for i in range(lengthx):
                            for j in range(lengthy):
                                if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                                    if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                                        if bound_img[dstx+i][dsty+j] !=0 and bound_img2[dstx+i][dsty+j] !=0:
                                            bound_img[dstx+i][dsty+j] =255
                                        else:
                                            bound_img[dstx+i][dsty+j] =0
                                        if img_vis[srcx+i][srcy+j] == 0:
                                            bframe_img[srcx+i][srcy+j] = bound_img[dstx+i][dsty+j]
                                        else :
                                            bframe_img[srcx+i][srcy+j] = (int(bound_img[dstx+i][dsty+j]) + int(bframe_img[srcx+i][srcy+j])) / 2
                                        img_vis[srcx+i][srcy+j] += 1
                    else:
                        for i in range(lengthx):
                            for j in range(lengthy):
                                if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                                    if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                                        if img_vis[srcx+i][srcy+j] == 0:
                                            bframe_img[srcx+i][srcy+j] = 0
                                        else :
                                            bframe_img[srcx+i][srcy+j] = (int(bframe_img[srcx+i][srcy+j])) / 2
                                        img_vis[srcx+i][srcy+j] += 1

    
    curstr = '%05d' % fcnt
    curstr = B_OUT_DIR + classname + "/" + curstr + ".png"
    cv2.imwrite(curstr,bframe_img)
    mappingratio = float(float(mappingnum)/float(totalnum))
    print(classname + str(fcnt) + " ratio : " + str(mappingratio))
    record_file.write(classname + str(fcnt) + " ratio : " + str(mappingratio) + "\n")
    


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
    print(classname)
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
    # classname = "scooter-black"
    vis = [False] * 200
    frame_mat = np.zeros((200,480,854),dtype="uint8")
    mvsmat = []
    bframe_gen()

record_file.close()
    

# def main():

#     mvsfiles= os.listdir(P_DIR)
#     for filename in mvsfiles: # i.e., bear
#         classname = filename
#         bframe_gen()




# if __name__ == "__main__":
#     main()
