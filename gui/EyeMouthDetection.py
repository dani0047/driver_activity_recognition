import numpy as np
import cv2 as cv
import math


right_eye_indices = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
left_eye_indices = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
left_pupil_indices = [469, 470, 471, 472]
right_pupil_indices = [474, 475, 476, 477]
mouth_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183 ]

def euclaidean_distance(point,point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1-x)**2 + (y1-y)**2)
    return distance

def mouth_detection(img, mesh_coords):
    global black_mask_mouth
    black_mask_mouth = np.zeros((90, 120, 3), dtype=np.uint8)
    img_height, img_width= img.shape[:2]
    mouth_coords = [mesh_coords[p] for p in mouth_indices]
    mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    mask_width, mask_height = 120, 90

    try:
        cv.polylines(mask,  [np.array(mouth_coords, dtype=np.int32)], True,(255,255,255), 1, cv.LINE_AA)

        # For mouth
        m_max_x = (max(mouth_coords, key=lambda item: item[0]))[0]
        m_min_x = (min(mouth_coords, key=lambda item: item[0]))[0]
        m_max_y = (max(mouth_coords, key=lambda item : item[1]))[1]
        m_min_y = (min(mouth_coords, key=lambda item: item[1]))[1]

        cropped_mouth = mask[m_min_y-5: m_max_y+5, m_min_x-20: m_max_x+20]
        scale_factor = mask_width / cropped_mouth.shape[1]
        new_height = int(cropped_mouth.shape[0] * scale_factor)
        resized_cropped_mouth = cv.resize(cropped_mouth, (mask_width, new_height))

        # Calculate the top and bottom padding needed
        top_padding_mouth = (mask_height - new_height) // 2
        black_mask_mouth[top_padding_mouth:top_padding_mouth+new_height, :] = resized_cropped_mouth

    except:
        black_mask_mouth = np.zeros((90, 120, 3), dtype=np.uint8)

    return black_mask_mouth

def eye_detection(img, results):
    global black_mask_left, black_mask_right
    black_mask_left = np.zeros((90, 120, 3), dtype=np.uint8)
    black_mask_right = np.zeros((90, 120, 3), dtype=np.uint8)
    img_height, img_width= img.shape[:2]
    mesh_coords = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    left_eye_coords = [mesh_coords[p] for p in left_eye_indices]
    left_pupil = [mesh_coords[p] for p in left_pupil_indices]
    right_eye_coords = [mesh_coords[p] for p in right_eye_indices]
    right_pupil = [mesh_coords[p] for p in right_pupil_indices]
    mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    mask_width, mask_height = 120, 90
    try:
        cv.polylines(mask,  [np.array(left_eye_coords, dtype=np.int32)], True,(255,255,255), 1, cv.LINE_AA)
        cv.polylines(mask,  [np.array(left_pupil, dtype=np.int32)], True,(255,255,255), 1, cv.LINE_AA)

        # For LEFT Eye
        l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
        l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
        l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
        l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

        cropped_left = mask[l_min_y-10: l_max_y+10, l_min_x-10: l_max_x+10]
        left_scale_factor = mask_width / cropped_left.shape[1]
        left_new_height = int(cropped_left.shape[0] * left_scale_factor)
        resized_cropped_left = cv.resize(cropped_left, (mask_width, left_new_height))

        # Calculate the top and bottom padding needed
        top_padding_left = (mask_height - left_new_height) // 2
        black_mask_left[top_padding_left:top_padding_left+left_new_height, :] = resized_cropped_left

    except:
        black_mask_left = np.zeros((90, 120, 3), dtype=np.uint8)

    try:
        cv.polylines(mask,  [np.array(right_eye_coords, dtype=np.int32)], True,(255,255,255), 1, cv.LINE_AA)
        cv.polylines(mask,  [np.array(right_pupil, dtype=np.int32)], True,(255,255,255), 1, cv.LINE_AA)

        # For LEFT Eye
        r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
        r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
        r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
        r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

        cropped_right = mask[r_min_y-10: r_max_y+10, r_min_x-10: r_max_x+10]
        right_scale_factor = mask_width / cropped_right.shape[1]
        right_new_height = int(cropped_right.shape[0] * right_scale_factor)
        resized_cropped_right = cv.resize(cropped_right, (mask_width, right_new_height))

        # Calculate the top and bottom padding needed
        top_padding_right = (mask_height - right_new_height) // 2
        black_mask_right[top_padding_right:top_padding_right+right_new_height, :] = resized_cropped_right
    
    except:
        black_mask_right = np.zeros((90, 120, 3), dtype=np.uint8)

    return black_mask_left, black_mask_right, mesh_coords

def blink_ratio_cal(landmarks = None):
    global blink_ratio

    if landmarks is not None:
        
        rh_right = landmarks[right_eye_indices[0]] 
        rh_left = landmarks[right_eye_indices[8]]
        # vertical line 
        rv_top1 = landmarks[right_eye_indices[13]] #385
        rv_bottom1 = landmarks[right_eye_indices[3]] #380

        rv_top2 = landmarks[right_eye_indices[11]] #387
        rv_bottom2 = landmarks[right_eye_indices[5]] #373

        # LEFT_EYE 
        # horizontal line 
        lh_right = landmarks[left_eye_indices[0]] 
        lh_left = landmarks[left_eye_indices[8]]

        # vertical line 
        lv_top1 = landmarks[left_eye_indices[13]] #160
        lv_bottom1 = landmarks[left_eye_indices[3]] #144

        lv_top2 = landmarks[left_eye_indices[11]] #158
        lv_bottom2 = landmarks[left_eye_indices[5]] #153

        rhDistance = euclaidean_distance(rh_right, rh_left)
        rv1Distance = euclaidean_distance(rv_top1, rv_bottom1)
        rv2Distance = euclaidean_distance(rv_top2, rv_bottom2)

        
        lhDistance = euclaidean_distance(lh_right, lh_left)
        lv1Distance = euclaidean_distance(lv_top1, lv_bottom1)
        lv2Distance = euclaidean_distance(lv_top2, lv_bottom2)

        reRatio = (rv1Distance + rv2Distance) / (2*rhDistance)
        leRatio = (lv1Distance + lv2Distance) / (2*lhDistance)


        blink_ratio = (reRatio+leRatio)/2
       

        blink_ratio = 0 if blink_ratio > 0.2 else blink_ratio
        blink_ratio = math.ceil(blink_ratio*100)/100


    else:
        blink_ratio = 0


    return blink_ratio

def yawn_ratio_cal(landmarks=None):
    global yawn_ratio
    if landmarks is not None:
        mh_right = landmarks[mouth_indices[0]]
        mh_left = landmarks[mouth_indices[10]]
        mv_top = landmarks[mouth_indices[15]]
        mv_bottom = landmarks[mouth_indices[5]]

        mhDistance = euclaidean_distance(mh_right, mh_left)
        mvDistance = euclaidean_distance(mv_top, mv_bottom)

        yawn_ratio = (mhDistance + mvDistance)/2
        yawn_ratio = 0 if yawn_ratio <= 30 else yawn_ratio

    else:
        yawn_ratio = 0
    
    return yawn_ratio