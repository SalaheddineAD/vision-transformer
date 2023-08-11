import cv2
import numpy as np
import random

def random_transform(image):
    choice = random.randint(0, 5)

    rows, cols, _ = image.shape

    if choice == 0: # Translation
        tx, ty = random.randint(-20, 20), random.randint(-20, 20)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (cols, rows))

    elif choice == 1: # Rotation
        angle = random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))

    elif choice == 2: # Scaling
        scale_factor = random.uniform(0.8, 1.2)
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    elif choice == 3: # Skewing
        pts1 = np.float32([[5,5],[20,5],[5,20]])
        pt1 = 5+10*np.random.uniform()-10/2
        pt2 = 20+10*np.random.uniform()-10/2
        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
        M = cv2.getAffineTransform(pts1,pts2)
        return cv2.warpAffine(image, M, (cols, rows))

    elif choice == 4: # Perspective distortion
        pts1 = np.float32([[5,5],[20,5],[5,20],[20,20]])
        pts2 = np.float32([[0,0],[20,5],[5,20],[20,20]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        return cv2.warpPerspective(image, M, (cols, rows))

    else: # Pincushion distortion
        distortion = np.float32([1, 1, 0, 0])
        camera_matrix = np.eye(3)
        return cv2.undistort(image, camera_matrix, distortion)
