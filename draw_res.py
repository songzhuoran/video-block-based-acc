import csv
import os
import re
from skimage import io,data
from PIL import Image, ImageDraw, ImageColor

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
    annopath = "/home/szr/video/block-mvs/davis2016/annotation/"
    residualpath = "/home/szr/video/block-mvs/residual/"
    drawpath = "/home/szr/video/block-mvs/res_draw/"
    mvsfiles= os.listdir(residualpath)
    for filename in mvsfiles:
        tmp = residualpath + filename
        # print tmp
        csvFile = open(tmp, mode='r')
        reader = csv.reader(csvFile)

        videofloder = re.sub('.csv','',filename) #delete .cvs, obtain the video folder name, i.e., bear
        # print videofloder
        videofile = annopath + videofloder  #open the video, i.e., /home/szr/video/block-mvs/davis2016/annotation/bear
        drawfile = drawpath + videofloder # i.e., /home/szr/video/block-mvs/res_draw/bear

        data = []
        for item in reader:
            data.append(item)
        # print data[0]
        csvFile.close()

        # convert string to int
        for row in data:
            # print row[0]
            # print len(row)
            for i in range(len(row )):
                row[i] = int(row[i])
                # print row[i]

        maxcur = 0 #find max index
        for row in data:
            cur = row[0]
            if(cur>maxcur):
                maxcur = cur

        for cur in range(maxcur+1): #make the image to be black
            curstr = '%05d' % cur
            drawimgstr = drawfile + "/" + curstr + ".png"
            curimg = Image.open(drawimgstr)  # read current file
            draw = ImageDraw.Draw(curimg)
            draw.rectangle((0,0,854,480),fill=0, outline=None)
            curimg.save(drawimgstr)


        for row in data: #draw pic
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
            # res = (res1,res2,res3)
            curstr = '%05d' % cur
            drawimgstr = drawfile + "/" + curstr + ".png"
            curimg = Image.open(drawimgstr)  # read current file
            # curimg = curimg.convert('RGB')
            draw = ImageDraw.Draw(curimg)
            # if (srcx>=0 and srcx<480) and (srcy>=0 and srcy<854):
            draw.rectangle((srcy,srcx,srcy+lengthy,srcx+lengthx),fill=res, outline=None)

            curimg.save(drawimgstr)



if __name__ == "__main__":
    main()
