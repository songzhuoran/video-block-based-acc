import csv
import os
import re
from skimage import io,data

def main():
    mvspath = "/home/szr/video/block-mvs/mvs/"
    videopath = "/home/szr/video/block-mvs/davis2016/data/"
    residualpath = "/home/szr/video/block-mvs/residual/"
    mvsfiles= os.listdir(mvspath)
    residual = {}
    for filename in mvsfiles:
        tmp = mvspath + filename
        csvFile = open(tmp, mode='r')
        reader = csv.reader(csvFile)

        videofloder = re.sub('.csv','',filename) #delete .cvs, obtain the video folder name, i.e., bear
        videofile = videopath + videofloder  #open the video, i.e., /home/szr/video/block-mvs/davis2016/bear

        residualFile = open(residualpath+filename, mode='w')
        writer = csv.writer(residualFile) # write file

        data = []
        for item in reader:
            data.append(item)

        csvFile.close()

        # convert string to int
        for row in data:
            for i in range(len(row )):
                row[i] = int(row[i])

        for row in data:
            cur = row[0]
            ref = row[1]
            lengthx = int(row[2])
            lengthy = int(row[3])
            srcy = int(row[4]) # current frame
            srcx = int(row[5])
            dsty = int(row[6])
            dstx = int(row[7])
            curstr = '%05d' % cur
            curstr = videofile + "/" + curstr + ".jpg"
            curimg = io.imread(curstr)  # read current file
            refstr = '%05d' % ref
            refstr = videofile + "/" + refstr + ".jpg"
            refimg = io.imread(refstr)  # read reference file
            res = 0
            reslist = [0,0,0]
            for c in range(3):
                res = 0
                for i in range(lengthx):
                    for j in range(lengthy):
                        if ((srcx+i)>=0 and (srcx+i)<480) and ((srcy+j)>=0 and (srcy+j)<854):
                            tmpcur = int(curimg[srcx+i][srcy+j][c])
                        else:
                            tmpcur = 0
                        if ((dstx+i)>=0 and (dstx+i)<480) and ((dsty+j)>=0 and (dsty+j)<854):
                            tmpref = int(refimg[dstx+i][dsty+j][c])
                        else:
                            tmpref = 0
                        res = res + abs(tmpcur-tmpref)
                reslist[c] = res/lengthx/lengthy
            
            # print residualpath+filename
            writer.writerow([cur,ref,lengthx,lengthy,srcy,srcx,dsty,dstx,reslist[0],reslist[1],reslist[2]])
        residualFile.close()
        #     break
        # break



if __name__ == "__main__":
    main()
