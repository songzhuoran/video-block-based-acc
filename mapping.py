import csv
import os
import re
from skimage import io,data
from PIL import Image, ImageDraw, ImageColor
import cv2

# MV: [CurrentFrame, TargetFrame, BlockWidth, BlockHeight, CurrentBlockX, CurBlockY, TargetX, TargetY]

def getFrameSavePath(name, frame) :
    dir = "/home/szr/video/block-mvs/res_draw/" + name 
    path = dir + "/" +  "%05d.png"%frame
    if not os.path.exists(dir) :
        os.mkdir(dir)
    return path

def normalize(dstmax, dstmin, srcmax, srcmin,x):
    res=round((dstmax-dstmin)*(x-srcmin)/(srcmax-srcmin)+dstmin)
    return res

def main():
    referencepath = "/home/songzhuoran/video/block-mvs/favos2016/"
    residualpath = "/home/songzhuoran/video/block-mvs/residual/"
    mappingpath = "/home/songzhuoran/video/block-mvs/mapping-thr20/"
    mvsfiles= os.listdir(referencepath)
    for filename in mvsfiles: # i.e., bear
        totalnum = 0
        mappingnum = 0
        tmp = residualpath + filename # i.e., /home/songzhuoran/video/block-mvs/residual/bear
        videofile = referencepath + filename  #open the video, i.e., /home/songzhuoran/video/block-mvs/favos2016/bear
        resfile = mappingpath + filename # i.e., /home/songzhuoran/video/block-mvs/mapping/bear

        tmp = tmp + ".csv"  # i.e., /home/songzhuoran/video/block-mvs/residual/bear.csv
        csvFile = open(tmp, mode='r')
        reader = csv.reader(csvFile)

        data = []
        for item in reader:
            data.append(item)
        csvFile.close()

        # convert string to int
        for row in data:
            # print row[0]
            # print len(row)
            for i in range(len(row )):
                row[i] = int(row[i])
                # print row[i]


        for row in data: # read residual
            totalnum = totalnum + 1
            cur = row[0]
            ref = row[1]
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
                curstr = '%05d' % cur
                curstr = resfile + "/" + curstr + ".png"
                curimg = cv2.imread(curstr)
                for i in range(lengthx):
                    for j in range(lengthy):
                        if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                            curimg[srcx+i,srcy+j] = 0 # initial the macro-block

                refstr = '%05d' % ref
                refstr = videofile + "/" + refstr + ".png"
                refimg = cv2.imread(refstr)
                for i in range(lengthx):
                    for j in range(lengthy):
                        if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                            if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                                curimg[srcx+i,srcy+j] = refimg[dstx+i,dsty+j]

                cv2.imwrite(curstr,curimg)

        mappingratio = float(float(mappingnum)/float(totalnum))
        print filename + " ratio : " + str(mappingratio)




if __name__ == "__main__":
    main()
