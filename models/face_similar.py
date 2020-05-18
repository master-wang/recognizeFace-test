import face_recognition

# Load a sample picture and learn how to recognize it.
wzl_image = face_recognition.load_image_file("testfemale.jpg")
wzl_face_encoding = face_recognition.face_encodings(wzl_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    wzl_face_encoding
]
known_face_names = [
    "wzl"
]

unknown_image = face_recognition.load_image_file("xue.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)

print("face_distances： {}".format(face_distances))
print("人脸相似度： {}%".format((1 - face_distances[0])*100))

