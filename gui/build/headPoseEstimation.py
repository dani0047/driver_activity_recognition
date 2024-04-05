import cv2 as cv
import numpy as np

def headPoseEstimation(frame, landmarks):
    img_h, img_w, img_c = frame.shape

    face_3d = []
    face_2d = []
    x,y,z = 0, 0 ,0

    for idx, lm in enumerate(landmarks.landmark):
        if idx == 33 or idx == 263 or idx == 1  or idx == 61 or idx == 291 or idx == 199:

            x, y = int(lm.x * img_w), int(lm.y * img_h)

            #Get 2D Coordinates
            face_2d.append([x,y])

            #Get 3D Coordinates
            face_3d.append([x, y, lm.z])

    #Convert to NumPy arrays
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    #Camera matrix
    focal_length = img_w

    cam_marix = np.array([[focal_length, 0 , img_w/2],
                        [0, focal_length, img_h/2],
                        [0, 0, 1]])

    #The distance matrix
    dist_matrix = np.zeros((4,1), dtype=np.float64)

    #Solve PnP
    success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_marix, dist_matrix)

    #Rotational matrix
    rmat, jac = cv.Rodrigues(rot_vec)

    #Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

    #Get rotation angle
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    return x, y, z
