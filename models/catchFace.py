import cv2 as cv
import time
import math

# 检测人脸并绘制人脸bounding box
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
    frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
    # blobFromImages表示处理多张图片，函数返回的blob是我们输入图像进行随意从中心裁剪，减均值、缩放和通道交换的结果。
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    #  blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]]) -> retval  返回值   # swapRB是交换第一个和最后一个通道   返回按NCHW尺寸顺序排列的4 Mat值
    net.setInput(blob)
    detections = net.forward()  # 网络进行前向传播，检测人脸
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])  # bounding box 的坐标
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),
                         8)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    return frameOpencvDnn, bboxes

# 网络模型  和  预训练模型
faceProto = "./opencv_face_detector.pbtxt"
faceModel = "./opencv_face_detector_uint8.pb"

ageProto = "./age_deploy.prototxt"
ageModel = "./age_net.caffemodel"

genderProto = "./gender_deploy.prototxt"
genderModel = "./gender_net.caffemodel"

# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['male', 'female']

# 加载网络
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
# 人脸检测的网络和模型
faceNet = cv.dnn.readNet(faceModel, faceProto)

# 打开一个视频文件或一张图片或一个摄像头
# 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
# cap = cv.VideoCapture('./xue.jpg')
cap = cv.VideoCapture(0)

padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    # 按帧读取视频, hasFrame(boolean)  frame获取的图像
    hasFrame, frame = cap.read()
    # 1 水平翻转，0垂直翻转 -1水平垂直翻转
    frame = cv.flip(frame, 1)
    # 不存在图片时
    if not hasFrame:
        cv.waitKey()
        break
    # 检测人脸并绘制人脸bounding box
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue
    fileName = str(time.time())
    for bbox in bboxes:
        # print(bbox)   # 取出box框住的脸部进行检测,返回的是脸部图片
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        # cv.imwrite('./catchImgs/' + fileName + "_face.png", face)
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)   # blob输入网络进行性别的检测
        genderPreds = genderNet.forward()   # 性别检测进行前向传播
        gender = genderList[genderPreds[0].argmax()]   # 分类  返回性别类型
        # 由于有一些女生性别判断不准，取值一下
        # gender = genderList[math.ceil(genderPreds[0][1])]   # 分类  返回性别类型
        print("*******************************")
        print(genderPreds[0][1])
        print(math.ceil(genderPreds[0][1]))
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("*******************************")
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                   cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)
        cut = frameFace[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        # cv.imwrite('./catchImgs/' + fileName + "_cut.png", cut)
    # cv.imwrite( './catchImgs/' + fileName + "_sceny.png", frameFace, [int(cv.IMWRITE_PNG_COMPRESSION), 5])
    print("time : {:.3f} ms".format(time.time() - t))