import face_recognition
import os

face_arr = []
dir = "C:\\Users\\wangzhilv\\Desktop\\face\\pyFacialRecognition\\models\\knowImgs"
for root, dirs, files in os.walk(dir):
    for file in files:
        fileName = os.path.join(root,file)
        img = face_recognition.load_image_file(fileName)
        face_arr.append(face_recognition.face_encodings(img)[0])


# Load the jpg files into numpy arrays
# wzl_image = face_recognition.load_image_file("knowImgs/wzl.jpg")
# xue_image = face_recognition.load_image_file("knowImgs/xue.jpg")
unknown_image = face_recognition.load_image_file("testfemale.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    # wzl_face_encoding = face_recognition.face_encodings(wzl_image)[0]
    # xue_face_encoding = face_recognition.face_encodings(xue_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

# known_faces = [
#     wzl_face_encoding,
#     xue_face_encoding
# ]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(face_arr, unknown_face_encoding)

print("results {}".format(results))
print("Is the unknown face a picture of wzl? {}".format(results[0]))
print("Is the unknown face a picture of xue? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))