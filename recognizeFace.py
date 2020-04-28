import cv2,sys
faceClassifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # the HAAR cascade file, which contains the machine learned data for face detection
# 读取系统输入的图片
# objImage=cv2.imread(sys.argv[1])
# 直接读取文件
objImage=cv2.imread("./test.jpg", 1)

cvtImage=cv2.cvtColor(objImage,cv2.COLOR_BGR2GRAY) # convert the image to gray scale
foundFaces=faceClassifier.detectMultiScale(cvtImage,scaleFactor=1.3,minNeighbors=9,minSize=(50,50),flags = cv2.IMREAD_GRAYSCALE) # to detect faces
print(" There are {} faces in the input image".format(len(foundFaces)))
for (x,y,w,h) in foundFaces:# to iterate each faces founded
	cv2.rectangle(objImage,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow("Facial Recognition Result, click anykey of keyboard to exit", objImage) #show the image
# 后面的数字取值范围是[0,9]
# 图片质量会虽数字变大而变小
cv2.imwrite("./face.png",objImage,[int(cv2.IMWRITE_PNG_COMPRESSION),5])
cv2.waitKey(0)