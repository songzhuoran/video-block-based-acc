import cv2
import sys
import os


INPUT_DIR = "/home/songzhuoran/video/video-block-based-acc/block_result/"
ANNO_DIR = "/home/songzhuoran/video/video-block-based-acc/davis2016/annotation/"

record_file = open("/home/songzhuoran/video/video-block-based-acc/result.csv", "w")

dir_list = os.listdir(INPUT_DIR)
total_fscore = 0.0
total_iou = 0.0
total_cnt = 0
for dir in dir_list:
    print("Start calculate " + dir + " ...")
    file_list = os.listdir(INPUT_DIR + dir)

    dir_fscore = 0.0
    dir_iou = 0.0
    dir_cnt = 0


    for file in file_list:
        input_img = cv2.imread(INPUT_DIR + dir + "/" + file, 0)
        anno_img = cv2.imread(ANNO_DIR + dir + "/" + file, 0)
        tp_cnt = 0.0
        fp_cnt = 0.0
        fn_cnt = 0.0

        for i in range(0, 480):
            for j in range(0, 854):
                if input_img[i][j] == 255 and anno_img[i][j] == 255:
                    tp_cnt += 1
                if input_img[i][j] == 255 and anno_img[i][j] == 0:
                    fp_cnt += 1
                if input_img[i][j] == 0 and anno_img[i][j] == 255:
                    fn_cnt += 1

        precision = tp_cnt / (fn_cnt + tp_cnt)
        recall = tp_cnt / (fp_cnt + tp_cnt)
        
        if (precision + recall) == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)

        if (fp_cnt + tp_cnt + fn_cnt) == 0:
            iou = 0
        else:
            iou = tp_cnt / (fp_cnt + tp_cnt + fn_cnt)

        dir_fscore += fscore
        dir_iou += iou
        dir_cnt += 1

        total_fscore += fscore
        total_iou += iou
        total_cnt += 1

        print(file + " calculate down with fscore: " + str(fscore) + " and iou: " + str(iou))
    print(dir + " calculate down with fscore: " + str(dir_fscore/dir_cnt) + " and iou: " + str(dir_iou/dir_cnt))
    record_file.write(dir + "," + str(dir_fscore/dir_cnt) + "," + str(dir_iou/dir_cnt) + "\n")

record_file.write("total," + str(total_fscore/total_cnt) + "," + str(total_iou/total_cnt) + "\n")

record_file.close()