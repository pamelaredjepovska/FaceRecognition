Attendance Recording using Face Recognition
==================================================================

The consistent attendance of students in classes is a part of their grade at many universities. 
This project offers the approach for developing an attendance recording application that would use face recognition and deep learning. A face recognition system is a biometric-based verification system for determining a person's identity in two steps: face detection and face identification. 

The proposed system offers a solid accuracy of precision and recall in real-time video. It successfully marks the person’s attendance in an Excel sheet once they are identified. The development of this facial recognition system was done utilizing Adam Geitgey's **face recognition library**, the **Python** programming language and **PyCharm** as IDE. This library is based on dlib's deep learning-based state-of-the-art facial recognition. 

The basic concept is to read the image from the webcam and locate all of the faces and their encodings. The encodings of the person must be compared to our existing encodings (gotten from previously conducted training) in order to determine the person's name. The lowest distance is the best match, and we'll display the person's name depending on its index. If the user in the camera already has an entry in the file, nothing will happen. If the user is new, however, the minimum distance will be checked. If the distance is less than 0.5, the user's name and current timestamp will be recorded. If it isn't, the program will display 'Unknown', and the attendance will not be recorded.

**Required libraries for installation:** cmake, cv2, dlib-19.18.0, face_recognition, pip
