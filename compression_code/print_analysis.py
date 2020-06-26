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
    for filename in filenames: # 00001.png
        img_name=re.sub('[.png]', '', filename)
        fd = ANA_DIR+classname+"/"+img_name+".npy"
        cnt_ref = np.load(fd)
        for i in range(60):
            for j in range(107):
                print(cnt_ref[i][j])
