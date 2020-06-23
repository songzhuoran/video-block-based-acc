import csv
import os
import re
import cv2
import numpy as np

def main():
    mvspath = "/home/songzhuoran/video/video-block-based-acc/mvs/"
    videopath = "/home/songzhuoran/video/video-block-based-acc/davis2016/data/"
    residualpath = "/home/songzhuoran/video/video-block-based-acc/residual_img/"
    # mvsfiles= os.listdir(residualpath)
    mvsfiles = {"bmx-trees","drift-straight","horsejump-high","kite-surf"}
    for filename in mvsfiles:
        videofloder = filename
        tmp = mvspath + filename + ".csv"
        csvFile = open(tmp, mode='r')
        reader = csv.reader(csvFile)
        data = []
        for item in reader:
            data.append(item)
        # convert string to int
        for row in data:
            for i in range(len(row )):
                row[i] = int(row[i])
        csvFile.close()
        videofile = videopath + videofloder  #open the video, i.e., /home/szr/video/block-mvs/davis2016/bear
        imgstrs = os.listdir(residualpath+videofloder)
        for imgstr in imgstrs:
            resimg_str = residualpath + videofloder + "/" + imgstr
            resimg = cv2.imread(resimg_str)
            print(resimg_str)
            num_img=int(re.sub('[.png]', '', imgstr))

            for row in data:
                cur = row[0]
                if int(cur) == num_img:
                    ref = row[1]
                    lengthx = int(row[2])
                    lengthy = int(row[3])
                    srcy = int(row[4]) # current frame
                    srcx = int(row[5])
                    dsty = int(row[6])
                    dstx = int(row[7])
                    curstr = '%05d' % cur
                    curstr = videofile + "/" + curstr + ".jpg"
                    curimg = cv2.imread(curstr)  # read current file
                    refstr = '%05d' % ref
                    refstr = videofile + "/" + refstr + ".jpg"
                    refimg = cv2.imread(refstr)  # read reference file
                    for c in range(3):
                        res = 0
                        for i in range(lengthx):
                            for j in range(lengthy):
                                if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                                    tmpref = int(refimg[dstx+i][dsty+j][c])
                                else:
                                    tmpref = 0
                                if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                                    tmpcur = int(curimg[srcx+i][srcy+j][c])
                                    resimg[srcx+i][srcy+j][c] = tmpcur - tmpref
            cv2.imwrite(resimg_str,resimg)
            



if __name__ == "__main__":
    main()
