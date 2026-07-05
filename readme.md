Face Detection using MTCNN

This project implements a Face Detection System using Python, OpenCV, and the MTCNN (Multi-task Cascaded Convolutional Neural Network) algorithm. The system detects faces in an input image, draws bounding boxes around each detected face, and marks important facial landmarks such as the eyes, nose, and mouth.

Features
Detects one or multiple faces in an image.
Draws a bounding box around each detected face.
Detects facial landmarks:
Left Eye
Right Eye
Nose
Left Mouth Corner
Right Mouth Corner
Visualizes detection results using OpenCV.
Displays face coordinates and landmark positions.
Technologies Used
Python
OpenCV
MTCNN
NumPy
How It Works
Load the input image using OpenCV.
Initialize the MTCNN face detector.
Detect all faces present in the image.
Extract the face bounding box coordinates.
Draw a rectangle around the detected face.
Extract facial landmarks (eyes, nose, and mouth).
Draw landmark points on the detected face.
Display the final annotated image.

This project demonstrates accurate face detection and facial landmark localization, making it a foundation for applications such as face recognition, attendance systems, emotion detection, and biometric authentication.