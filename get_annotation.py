import time
import cv2
import os

# ------ 视频数据集所在的路径 --------
data_path = "data"
# ---------------------------------


def Video2Mp4(videoPath, outVideoPath):
    capture = cv2.VideoCapture(videoPath)
    fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    suc = capture.isOpened()  # 是否成功打开

    allFrame = []
    while suc:
        suc, frame = capture.read()
        if suc:
            allFrame.append(frame)
    capture.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(outVideoPath, fourcc, fps, size)
    for aFrame in allFrame:
        videoWriter.write(aFrame)
    videoWriter.release()

data_classes = ["NonViolence", "Violence"]


train_file_path = "annotation/violence_train_videos.txt"
val_file_path = "annotation/violence_val_videos.txt"
train_file_path_io = open(train_file_path, "w")
val_file_path_io = open(val_file_path, "w")

del_count = 0
for one_class in data_classes:
    files = os.listdir(os.path.join(data_path, one_class))
    train_count = int(len(files) * 0.7)
    for index, file in enumerate(files):
        fname = f"{data_path}/{one_class}/{file}"
        wstr = f"{fname} {data_classes.index(one_class)}\n"
        fsize = os.path.getsize(fname)
        if fsize / 1024 / 1024 > 3:
            del_count += 1
            continue
        print(wstr,end="")
        if index < train_count:
            train_file_path_io.write(wstr)
        else:
            val_file_path_io.write(wstr)
train_file_path_io.close()
val_file_path_io.close()
print(del_count)


# for one_class in data_classes:
    # files = os.listdir("./" + one_class)
    # for file in files:
        # if file[-3:] == "mp4":
            # pass
        # else:
            # inputVideoPath = os.path.join(one_class, file)
            # outputVideoPath = os.path.join(one_class,file[:-3] + 'mp4')
            # print(inputVideoPath, outputVideoPath)
            # Video2Mp4(inputVideoPath, outputVideoPath)
            # print(inputVideoPath,"转换完毕")
            # os.remove(inputVideoPath)
            # print(inputVideoPath,"删除成功")

