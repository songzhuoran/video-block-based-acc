import sys
import os
import csv
import cv2
import numpy as np

IDX_DIR="../data/idx/"
MVS_DIR="../data/mvs/"
B_OUT_DIR="../data/val/"
B_INIT_DIR="/home/songzhuoran/video/video-block-based-acc/lossless/data/favos_add_Sseg_CRF_Tracker/"
P_DIR="/home/songzhuoran/video/video-block-based-acc/lossless/data/favos_add_Sseg_CRF_Tracker/"
B_DIR="../data/val/bframe/"
RES_DIR="/home/songzhuoran/video/video-block-based-acc/residual/"

vis = [False] * 200
mvsmat = []
bflist = []  # aka b frame list
pflist = []  # aka b frame list
classname = sys.argv[1]
frame_mat = np.zeros((200,480,854),dtype="uint8")

def check_x_outside(point_x):
    if 0 <= point_x < 854:
        return True
    else:
        return False

def check_y_outside(point_y):
    if 0 <= point_y < 480:
        return True
    else:
        return False

def bframe_gen_kernel(fcnt):
    
    img_vis = np.zeros((480,854))

    curstr = '%05d' % fcnt
    curstr = B_INIT_DIR + classname + "/" + curstr + ".png"
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
    
    cv2.imwrite(B_OUT_DIR+"bframe/"+classname+"/%05d.png" % fcnt,bframe_img)
    # exit()
    


def DFS(fcnt):
    # print(vis)
    if vis[fcnt]:
        return True
    else:
        for i in mvsmat[fcnt]:
            DFS(i)
        bframe_gen_kernel(fcnt)
        frame_mat[fcnt] = cv2.imread(B_DIR+"/"+classname+"/%05d.png" % fcnt,0)
        vis[fcnt] = True
        return True
    

def bframe_gen():
    with open(IDX_DIR+"b/"+classname, "r") as file:
        for row in file:
            bflist.append(int(row)-1)
    # print(bflist)

    with open(IDX_DIR+"p/"+classname, "r") as file:
        for row in file:
            pflist.append(int(row)-1)
    # print(pflist)

    framecnt = pflist[-1] + 1

    for i in pflist:
        vis[i] = True
        frame_mat[i] = cv2.imread(P_DIR+classname+"/%05d.png" % i,0)

    for i in range(framecnt):
        mvsmat.append(set())

    with open(MVS_DIR+classname+".csv","r") as file:
        datainfo = csv.reader(file)
        for row in datainfo:
            # print(int(row[0]))
            # print(int(row[1]))
            mvsmat[int(row[0])].add(int(row[1]))

    # for i in range(0, framecnt):
    #     print("frame "+str(i)+"'s mvsmat:")
    #     print(mvsmat[i])

    for i in bflist:
        # print(i)
        if not vis[i]:
            DFS(i)

    for i in range(framecnt):
        if not vis[i]:
            print("ERROR")



bframe_gen()