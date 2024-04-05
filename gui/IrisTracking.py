import numpy as np
import cv2 as cv
import mediapipe as mp


def gaze(frame, landmarks):
        img_h, img_w, img_c = frame.shape

        face_2d = []
        face = []
        x,y,z = 0, 0 ,0


        face_3d = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (43.3, 32.7, -26),  # Right eye, right corner
            (28.9, -28.9, -24.1),  # Right mouth corner
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
        ])

        Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

        for idx, lm in enumerate(landmarks):
            if idx == 33 or idx == 287 or idx == 4  or idx == 152 or idx == 263 or idx == 57:

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                #Get 2D Coordinates
                face_2d.append((x,y))

                #Get 3D Coordinates
                face.append((x,y,0))


            if idx == 473:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                right_pupil = (x,y)


            if idx == 468:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                left_pupil = (x,y)

                    
            #Convert to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face = np.array(face, dtype = np.float64)

            '''
            camera matrix estimation
            '''
            focal_length = frame.shape[1]
            print(focal_length)
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv.solvePnP(face_3d, face_2d, camera_matrix,
                                                                        dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
            
            # Transformation between image point to world point
            _, transformation, _ = cv.estimateAffine3D(face, face_3d)  # image to world transformation

            if transformation is not None:  # if estimateAffine3D secsseded
                # project pupil image point into 3d world point 
                pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

                # 3D gaze point (10 is arbitrary value denoting gaze distance)
                S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

                # Project a 3D gaze direction onto the image plane.
                (eye_pupil2D, _) = cv.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                                    translation_vector, camera_matrix, dist_coeffs)
                # project 3D head pose into the image plane
                (head_pose, _) = cv.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                                rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
                # correct gaze for head rotation
                gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

                # Draw gaze line into screen
                p1 = (int(left_pupil[0]), int(left_pupil[1]))
                p2 = (int(gaze[0]), int(gaze[1]))
                cv.line(frame, p1, p2, (0, 0, 255), 2)

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(color=(255, 255, 255),thickness=1,circle_radius=1)


cap = cv.VideoCapture(0)

while True:
     ret, frame = cap.read()
     if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = faceMesh.process(frame)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                gaze(frame, faceLms.landmark)  # gaze estimation

        cv.imshow("frame", frame)
        cv.waitKey(1)   
          


cap.release()
cv.destroyAllWindows


