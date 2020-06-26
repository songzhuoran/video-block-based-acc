import numpy as np
import os
import re
import csv

RES_DIR="/home/songzhuoran/video/video-block-based-acc/residual/"
P_DIR="/home/songzhuoran/video/video-block-based-acc/favos2016/"
ANA_DIR="/home/songzhuoran/video/video-block-based-acc/analysis_ref/"


classnames = os.listdir(P_DIR)
for classname in classnames:
    filenames = os.listdir(P_DIR+classname)
    os.mkdir(ANA_DIR+classname)

    #read mvs and residual
    tmp = RES_DIR + classname + ".csv"
    print(tmp)
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

    for filename in filenames: # 00001.png
        img_name=re.sub('[.png]', '', filename)
        fcnt = int(img_name)
        cnt_ref = np.zeros((60,107))
        fd = ANA_DIR+classname+"/"+img_name+".npy"

        for row in data:
            if row[1]==fcnt:
                cur = row[0]
                ref = row[1]
                lengthx = row[2]
                lengthy = row[3]
                srcy = row[4] # current frame
                srcx = row[5]
                dsty = row[6] #reference frame
                dstx = row[7]
                for i in range(int(lengthx/8)):
                    for j in range(int(lengthy/8)):
                        if (int(dstx/8)+i)<60 and (int(dsty/8)+j)<107:
                            cnt_ref[int(dstx/8)+i][int(dsty/8)+j] = cnt_ref[int(dstx/8)+i][int(dsty/8)+j] + 1
        np.save(fd,cnt_ref)
