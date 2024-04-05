from utils import create_rounded_rectangle_mask, masked_image
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog
import customtkinter as CTk
import cv2 as cv
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import mediapipe as mp
import numpy as np
import time
import os
from cvZone import LivePlot
from datetime import datetime
import gaze
from headPoseEstimation import headPoseEstimation
from EyeMouthDetection import eye_detection, mouth_detection,  blink_ratio_cal, yawn_ratio_cal
import load_model as lm

CTk.deactivate_automatic_dpi_awareness()

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\LENOVO\Desktop\Uni\Year 4\Final Year Project\gui\build\assets\frame0")

front_file_dir = r"C:\Users\LENOVO\Desktop\Uni\Year 4\Final Year Project\Model\training_log\single_front_refine_all"
side_file_dir = r"C:\Users\LENOVO\Desktop\Uni\Year 4\Final Year Project\Model\training_log\side_refined_all"
dual_file_dir = r"C:\Users\LENOVO\Desktop\Uni\Year 4\Final Year Project\Model\training_log\dual_refined_all"

front_model = lm.LoadModel('mobilenetv3l', front_file_dir, 'sam-ddd')
side_model = lm.LoadModel('mobilenetv3l', side_file_dir, 'sam-ddd')
dual_model = lm.LoadModel('twostreammobilenet', dual_file_dir, 'sam-ddd')

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(color=(255, 255, 255),thickness=1,circle_radius=1)

#Initialise global variables
tester_name = None
front_mode = True
side_mode = True
front_cap = None
side_cap = None
front_frame = None
side_frame = None
folder_selected = None
front_frame_pil = None
side_frame_pil = None
record_status = False
ellipse_front = None
ellipse_side = None
front_writer = None
side_writer = None
result = None
blink_ratio = 0
blink_ratio_reciprocal = 0
drowsiness_level = 0
drowsiness_pTime = time.time()
drowsiness_cTime = drowsiness_pTime
black_mask_left = np.zeros((90, 120, 3), dtype=np.uint8)
black_mask_right = np.zeros((90, 120, 3), dtype=np.uint8)
black_mask_mouth = np.zeros((90, 120, 3), dtype=np.uint8)
blink_plot = np.zeros((150, 200, 3), dtype=np.uint8)
yawn_plot = np.zeros((150, 200, 3), dtype=np.uint8)


#Initilize plot parameters
plotB = LivePlot(200, 150, [0,0.2], interval = 0.001)
plotY = LivePlot(200, 150, [20,80], interval = 0.001)


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

'''             Button Toggle Functions          '''
def save_name():
    global tester_name
    tester_name = name_entry.get()
    entry_button.config(image= entry_button_img, activebackground="#2C2C2E")

def toggle_front():
    global front_mode
    front_mode = not front_mode  # Toggle the state
    capture_front_video(canvas)
    front_button.config(image=on_img if front_mode else front_off_img, activebackground="#2C2C2E")

def toggle_side():
    global side_mode
    side_mode = not side_mode  # Toggle the state
    capture_side_video(canvas)
    side_button.config(image=on_img if side_mode else side_off_img, activebackground="#2C2C2E")

def select_folder():
    global folder_selected
    folder_selected = filedialog.askdirectory()
    capture_front_video(canvas)
    capture_side_video(canvas)
    canvas.itemconfig(chosen_folder, text=folder_selected)
    folder_button.configure(image = select_folder_img, activebackground="#2C2C2E")

def toggle_capture():
    global folder_selected, front_mode, side_mode, front_frame_pil, side_frame_pil, tester_name
    capture_button.config(image= capture_button_img, activebackground="#2C2C2E")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_captured = -1
    label_text = ['Front image', 'Side Image', 'Front and side images']
    if folder_selected is not None and os.path.isdir(folder_selected):
        try:
            if isinstance(front_frame_pil, Image.Image):
                front_frame_pil.save(os.path.join(folder_selected, f"front_image_{tester_name}_{timestamp}.png"))
                img_captured += 1
            if isinstance(side_frame_pil, Image.Image):
                side_frame_pil.save(os.path.join(folder_selected, f"side_image_{tester_name}_{timestamp}.png"))
                img_captured +=2
            canvas.itemconfig(capture_text, text=f"{label_text[img_captured]} successfully captured.")

        except:
            canvas.itemconfig(capture_text, text="Error in saving image")
    else:
        canvas.itemconfig(capture_text, text="No selected folder to save image.")

def toggle_recording():
    global record_status, ellipse_front, ellipse_side, canvas, ellipse_img, actual_front_fps, actual_side_fps
    global folder_selected, front_writer, side_writer, front_cap, side_cap, tester_name
    ellipse_img = PhotoImage(file=relative_to_assets("image_5.png"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    record_idx = 0
    record_view = ['front camera', 'side camera', 'front and side cameras']
    if folder_selected is not None and os.path.isdir(folder_selected):
        ret_front, _ = front_cap.read()
        ret_side, _ = side_cap.read()
        if ret_front or ret_side:
            if not record_status:
                record_status = True
                
                if ret_front:
                    front_video_path = os.path.join(folder_selected, f"front_video_{tester_name}_{timestamp}.mp4")
                    # Start recording
                    fourcc= cv.VideoWriter_fourcc('m', 'p','4','v')
                    front_writer = cv.VideoWriter(front_video_path, fourcc, actual_front_fps, (400, 300))

                    # Show recording icon
                    ellipse_front = canvas.create_image(404.0, 41.0, image=ellipse_img)
                    record_idx +=1

                if ret_side:
                    side_video_path = os.path.join(folder_selected, f"side_video_{tester_name}_{timestamp}.mp4")
                    # Start recording
                    fourcc_side = cv.VideoWriter_fourcc('m', 'p','4','v')
                    side_writer = cv.VideoWriter(side_video_path, fourcc_side, actual_side_fps, (400, 300))
                    # Show recording icon
                    ellipse_side = canvas.create_image(404.0, 388.0, image=ellipse_img)
                    record_idx +=2

                canvas.itemconfig(record_text, text=f"Recording {record_view[record_idx-1]}...")
            else:
                record_status = False

                if ellipse_front is not None:
                    canvas.delete(ellipse_front)
                    ellipse_front = None

                if ellipse_side is not None:
                    canvas.delete(ellipse_side)
                    ellipse_side = None

                canvas.itemconfig(record_text, text="Recording stopped.")


        else:
            canvas.itemconfig(record_text, text="Unable to access camera.")

    else:
        canvas.itemconfig(record_text, text="No selected folder.")
    recording_button.config(image=stop_recording_img if record_status else start_recording_img, activebackground="#2C2C2E")

'''             Program Function            '''
def update_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    canvas.itemconfig(time_string, text=str(formatted_time))
    window.after(1000, update_time)

def run_distraction():
    global blink_ratio
    print(blink_ratio)
    new_blink_ratio = blink_ratio/10
    new_blink_ratio = max(0, min(1, new_blink_ratio))
    distraction_bar.set(new_blink_ratio)
    window.after(1, run_distraction)

def run_drowsiness():
    global blink_ratio_reciprocal, drowsiness_level, drowsiness_count_time, drowsiness_pTime, drowsiness_cTime
    drowsiness_cTime = time.time()
    if blink_ratio_reciprocal >= 0.04:
        drowsiness_count_time = drowsiness_cTime - drowsiness_pTime
        drowsiness_level += 0.05
        drowsiness_level = min(1, drowsiness_level)

    else:
        drowsiness_level = 0
        drowsiness_count_time = 0
        drowsiness_pTime = drowsiness_cTime

    canvas.itemconfig(drowsiness_intensity, text=int(drowsiness_level*100))
    canvas.itemconfig(eyes_closed_duration, text=round(drowsiness_count_time,2))
    drowsiness_bar.set(drowsiness_level)
    
    window.after(1, run_drowsiness)
    
def blink_freq_plot(blink_ratio):
    global blink_ratio_reciprocal
    try:
        blink_ratio_reciprocal = 1/(blink_ratio*100)
    except ZeroDivisionError as e:
        blink_ratio_reciprocal = 0

    blink_ratio_reciprocal = 0 if blink_ratio_reciprocal < 0.04 else blink_ratio_reciprocal
    blink_plot = plotB.update(blink_ratio_reciprocal)
    return blink_plot

def yawn_freq_plot(yawn_ratio):
    yawn_plot = plotY.update(yawn_ratio)
    return yawn_plot

def display_empty_frame(canvas, view):
    global front_window_img, side_window_img, mesh_window_img, left_window_img, right_window_img, mouth_window_img
    if view == 0:
        front_window_img = PhotoImage(file=relative_to_assets("image_6.png"))
        front_empty = canvas.create_image(223.0, 206.0, image=front_window_img)

        mesh_window_img = PhotoImage(file=relative_to_assets("image_11.png"))
        mesh_empty = canvas.create_image(620.0, 175.0, image=mesh_window_img)
    if view == 1:
        side_window_img = PhotoImage(file=relative_to_assets("image_4.png"))
        side_empty = canvas.create_image(223.0, 549.0, image=side_window_img)

    if view == 2:
        left_window_img = PhotoImage(file=relative_to_assets("image_8.png"))
        left_eye_window = canvas.create_image(877.0, 99.0, image=left_window_img)
    
    if view == 3:
        right_window_img = PhotoImage(file=relative_to_assets("image_9.png"))
        right_eye_window = canvas.create_image(1003.0, 99.0, image=right_window_img)
    
    if view == 4:
        mouth_window_img = PhotoImage(file=relative_to_assets("image_10.png"))
        mouth_window = canvas.create_image(1132.0, 99.0, image=mouth_window_img)

    if view == 5:
        blink_window_img = PhotoImage(file=relative_to_assets("image_2.png"))
        blink_window = canvas.create_image(1138.0, 387.0, image=blink_window_img)

    if view == 6:
        yawn_window_img = PhotoImage(file=relative_to_assets("image_1.png"))
        yawn_window = canvas.create_image(917.0, 389.0, image=yawn_window_img)

    
def update_front_frame(canvas,cap, canvas_image_item, mesh_frame, lefteye_frame, righteye_frame, mouth_frame, blink_frame, yawn_frame, face_mesh, pTime=0):
    global front_mode, front_frame_pil, black_mask_left, black_mask_right, black_mask_mouth 
    global blink_plot, yawn_plot, front_writer, blink_ratio, drowsiness_level, front_frame, result
    if not front_mode:
        fps = 0
        canvas.itemconfig(front_fps_text, text=fps)
        return
    else:
        ret, frame = cap.read()
        if ret:
            frame = cv.resize(frame, (400, 300))
            cTime= time.time()
            fps = int(1/(cTime-pTime))
            pTime = cTime
            
            front_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            cv.putText(frame,str(f"FPS:{int(fps)}"),(10,20), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv.putText(frame,str(f"Pred Class: {result}"),(10,270), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv.putText(front_frame,str(f"FPS:{int(fps)}"),(10,20), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            cv.putText(front_frame,str(f"Pred Class: {result}"),(10,270), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            
            if record_status:
                front_writer.write(frame)
            elif not record_status and front_writer is not None:
                front_writer.release()
                front_writer = None
            front_frame_pil = Image.fromarray(front_frame)
            front_frame_pil = front_frame_pil.resize((400, 300), Image.Resampling.LANCZOS)

            #Apply mask
            mask = create_rounded_rectangle_mask(400, 300, 10)
            front_frame_pil.putalpha(mask)

            img = ImageTk.PhotoImage(front_frame_pil)
            canvas.itemconfig(canvas_image_item, image=img)
            canvas.front_image = img

            if face_mesh:
                # Create a black NumPy array for drawing
                black_image_face = np.zeros((240, 320, 3), dtype=np.uint8)
                
                front_frame.flags.writeable = False

                # Process face mesh and draw landmarks
                results = faceMesh.process(front_frame)

                front_frame.flags.writeable = True

                x, y, z = 0, 0, 0
                gaze_yaw, gaze_pitch = 0, 0

                if results.multi_face_landmarks:
                    for faceLms in results.multi_face_landmarks:
                        mpDraw.draw_landmarks(black_image_face, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                        x, y, z = headPoseEstimation(front_frame, faceLms)

                    gaze_yaw, gaze_pitch = gaze.gaze(black_image_face, results.multi_face_landmarks[0])
                    black_mask_left, black_mask_right, mesh_coords = eye_detection(frame, results)
                    black_mask_mouth = mouth_detection(frame, mesh_coords)
                    blink_ratio = blink_ratio_cal(mesh_coords)
                    blink_plot = blink_freq_plot(blink_ratio)
                    yawn_ratio = yawn_ratio_cal(mesh_coords)
                    yawn_plot = yawn_freq_plot(yawn_ratio)

                else:
                    black_mask_left = np.zeros((90, 120, 3), dtype=np.uint8)
                    black_mask_right = np.zeros((90, 120, 3), dtype=np.uint8)
                    black_mask_mouth = np.zeros((90, 120, 3), dtype=np.uint8)
                    blink_ratio = blink_ratio_cal(landmarks = None)
                    yawn_ratio = yawn_ratio_cal(landmarks = None)
                    blink_plot = blink_freq_plot(blink_ratio)
                    yawn_plot = yawn_freq_plot(yawn_ratio)


                mesh_img = masked_image(black_image_face, [320,240,10])
                lefteye_img = masked_image(black_mask_left, [120,90,10])
                righteye_img = masked_image(black_mask_right, [120,90,10])
                mouth_img = masked_image(black_mask_mouth, [120,90,10])
                blink_img = masked_image(blink_plot, [200,150,10])
                yawn_img = masked_image(yawn_plot, [200,150,10])


                # Convert back to PhotoImage for display
                canvas.itemconfig(mesh_frame, image=mesh_img)
                canvas.itemconfig(lefteye_frame, image=lefteye_img)
                canvas.itemconfig(righteye_frame, image=righteye_img) 
                canvas.itemconfig(mouth_frame, image=mouth_img)
                canvas.itemconfig(blink_frame, image=blink_img)
                canvas.itemconfig(yawn_frame, image=yawn_img)         
                canvas.itemconfig(roll_angle, text=round(z,2))
                canvas.itemconfig(yaw_angle, text=round(y,2))
                canvas.itemconfig(pitch_angle, text=round(x,2))
                canvas.itemconfig(gaze_yaw_string, text=round(gaze_yaw,2))
                canvas.itemconfig(gaze_pitch_string, text=round(gaze_pitch,2))
                canvas.image_mesh = mesh_img  
                canvas.left_eye = lefteye_img
                canvas.right_eye = righteye_img
                canvas.mouth = mouth_img
                canvas.blink = blink_img
                canvas.yawn = yawn_img
    
        else:
            display_empty_frame(canvas, 0)
            display_empty_frame(canvas, 2)
            display_empty_frame(canvas, 3)
            display_empty_frame(canvas, 4)
            display_empty_frame(canvas, 5)
            display_empty_frame(canvas, 6)
        print(blink_ratio)

        canvas.itemconfig(front_fps_text, text=fps)
        
    window.after(30, update_front_frame, canvas, cap, canvas_image_item, mesh_frame, lefteye_frame, righteye_frame, mouth_frame, blink_frame,yawn_frame, face_mesh, pTime)

def update_side_frame(canvas,cap, canvas_image_item, pTime=0):
    global side_mode, side_frame_pil, side_writer, side_frame, result
    if not side_mode:
        fps = 0
        display_empty_frame(canvas, 1)
        canvas.itemconfig(side_fps_text, text=fps)
        return
    else:
        ret, frame = cap.read()
        if ret:
            frame = cv.resize(frame, (400, 300))
            cTime= time.time()
            fps = int(1/(cTime-pTime))
            pTime = cTime
            side_frame = cv.cvtColor(cv.flip(frame,1), cv.COLOR_BGR2RGB)
            side_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            cv.putText(frame,str(f"FPS:{int(fps)}"),(10,20), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv.putText(frame,str(f"Pred Class: {result}"),(10,270), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv.putText(side_frame,str(f"FPS:{int(fps)}"),(10,20), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            cv.putText(side_frame,str(f"Pred Class: {result}"),(10,270), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            if record_status:
                side_writer.write(frame)
            elif not record_status and side_writer is not None:
                side_writer.release()
                side_writer = None
            side_frame_pil = Image.fromarray(side_frame)
            side_frame_pil = side_frame_pil.resize((400, 300), Image.Resampling.LANCZOS)

            #Apply mask
            mask = create_rounded_rectangle_mask(400, 300, 10)
            side_frame_pil.putalpha(mask)

            img = ImageTk.PhotoImage(side_frame_pil)
            canvas.itemconfig(canvas_image_item, image=img)
            canvas.side_image = img

        else:
            display_empty_frame(canvas, 1)
            fps = 0
        canvas.itemconfig(side_fps_text, text=fps)

    window.after(30, update_side_frame, canvas, cap, canvas_image_item, pTime)

def capture_front_video(canvas):
    global front_mode, front_cap, front_canvas_image, mesh_frame, lefteye_frame, righteye_frame, actual_front_fps
    if front_mode:
        if front_cap is None or not front_cap.isOpened():
            front_cap = cv.VideoCapture(0)
            actual_front_fps = front_cap.get(cv.CAP_PROP_FPS)
            if not front_cap.isOpened():
                display_empty_frame(canvas, 0)
            else:
                face_mesh=True
                front_canvas_image = canvas.create_image(223.0, 206.0, image=tk.PhotoImage())
                mesh_frame = canvas.create_image(620.0, 175.0, image=tk.PhotoImage())
                lefteye_frame = canvas.create_image(877.0, 99.0, image=tk.PhotoImage())
                righteye_frame = canvas.create_image(1003.0, 99.0, image=tk.PhotoImage())
                mouth_frame = canvas.create_image(1132.0, 99.0, image=tk.PhotoImage())
                blink_frame = canvas.create_image(1138.0, 387.0, image=tk.PhotoImage())
                yawn_frame = canvas.create_image(917.0, 389.0, image=tk.PhotoImage())
                update_front_frame(canvas, front_cap, front_canvas_image, mesh_frame, lefteye_frame, righteye_frame, mouth_frame, blink_frame, yawn_frame, face_mesh)
    else:
        if front_cap is not None:
            front_cap.release()
            front_cap = None
        display_empty_frame(canvas, 0)
        display_empty_frame(canvas, 2)
        display_empty_frame(canvas, 3)
        display_empty_frame(canvas, 4)

def capture_side_video(canvas):
    global side_mode, side_cap, side_canvas_image, actual_side_fps
    if side_mode:
        if side_cap is None or side_cap.isOpened():
            side_cap = cv.VideoCapture(1)
            actual_side_fps = side_cap.get(cv.CAP_PROP_FPS)
            if not side_cap.isOpened():
                display_empty_frame(canvas, 1)
            else:
                side_canvas_image = canvas.create_image(223.0, 549.0, image=tk.PhotoImage())
                update_side_frame(canvas, side_cap, side_canvas_image)

    else:
        if side_cap is not None:
            side_cap.release()
            side_cap = None
        display_empty_frame(canvas, 1)

def class_prediction():
    global front_frame, side_frame, result
    try:
        if front_cap is not None:
            if side_cap is None:
                model = front_model
                result = model.model_inference(front_frame)
                print('front')
            else:
                model = dual_model
                result = model.model_inference(front_frame, side_frame)
                print('dual')
        else:
            if side_cap is not None:
                model = side_model
                result = model.model_inference(side_frame)
                print('side')
            else:
                model = None
                result = "None"
                print('Cameras off')
    except:
        print("Unable to detect model")
    
            
    canvas.itemconfig(predicted_class, text=str(result))
    window.after(100, class_prediction) 

'''                 Window Settings                 '''
window = CTk.CTk()
window.geometry("1280x740")
window.configure(bg = "#2C2C2E")
window.title("Driver State Monitoring Interface")
CTk.set_appearance_mode("dark")

canvas = Canvas(
    window,
    bg = "#2C2C2E",
    height = 740,
    width = 1280,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)

'''                             Window Configurations                              '''

side_window_img = PhotoImage(file=relative_to_assets("image_6.png"))
side_empty = canvas.create_image(223.0, 549.0, image=side_window_img)

front_window_img = PhotoImage(file=relative_to_assets("image_4.png"))
front_empty = canvas.create_image(223.0, 206.0, image=front_window_img)

mesh_window_img = PhotoImage(file=relative_to_assets("image_11.png"))
mesh_empty = canvas.create_image(620.0, 175.0, image=mesh_window_img)

left_window_img = PhotoImage(file=relative_to_assets("image_8.png"))
left_eye_window = canvas.create_image(877.0, 99.0, image=left_window_img)

right_window_img = PhotoImage(file=relative_to_assets("image_9.png"))
right_eye_window = canvas.create_image(1003.0, 99.0, image=right_window_img)

mouth_window_img = PhotoImage(file=relative_to_assets("image_10.png"))
mouth_window = canvas.create_image(1132.0, 99.0, image=mouth_window_img)

blink_window_img = PhotoImage(file=relative_to_assets("image_2.png"))
blink_window = canvas.create_image(1138.0, 387.0, image=blink_window_img)

yawn_window_img = PhotoImage(file=relative_to_assets("image_1.png"))
yawn_window = canvas.create_image(917.0, 389.0, image=yawn_window_img)


'''                 Labels Configuration            '''
yawn_label = canvas.create_text(
    817.0,
    285.0,
    anchor="nw",
    text="YAWN FREQUENCY",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

blink_label = canvas.create_text(
    1039.0,
    283.0,
    anchor="nw",
    text="BLINK FREQUENCY",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

tester_label = canvas.create_text(
    819.0,
    513.0,
    anchor="nw",
    text="TESTER NAME: ",
    fill="#FFFFFF",
    font=("Inter", 13 * -1)
)

folder_label = canvas.create_text(
    819.0,
    554.0,
    anchor="nw",
    text="SAVED FOLDER:",
    fill="#FFFFFF",
    font=("Inter", 13 * -1)
)

chosen_folder = canvas.create_text(
    850,
    584.0,
    anchor="nw",
    text="No chosen folder",
    fill="#FFFFFF",
    font=("Inter", 13 * -1)
)

front_label = canvas.create_text(
    23.0,
    32.0,
    anchor="nw",
    text="FRONT VIEW",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

front_fps_label = canvas.create_text(
    242.0,
    34.0,
    anchor="nw",
    text="FPS:",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)


front_fps_text = canvas.create_text(
    272.0,
    34.0,
    anchor="nw",
    text= "0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

side_label = canvas.create_text(
    23.0,
    373.0,
    anchor="nw",
    text="SIDE VIEW",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

side_fps_label = canvas.create_text(
    242.0,
    374.0,
    anchor="nw",
    text="FPS:",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

side_fps_text = canvas.create_text(
    272.0,
    374.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

record_text = canvas.create_text(
    462,
    668,
    anchor="nw",
    text="Click button to start recording",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

capture_text = canvas.create_text(
    462,
    607,
    anchor="nw",
    text="Click button to screenshot",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)


#### head pose angles ####
yaw_angle = canvas.create_text(
    505.0,
    320.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

pitch_angle = canvas.create_text(
    620.0,
    320.0,
    anchor="nw",
    text=f"0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

roll_angle = canvas.create_text(
    720.0,
    320.0,
    anchor="nw",
    text=f"0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

gaze_yaw_string = canvas.create_text(
    864.0,
    234.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

gaze_pitch_string = canvas.create_text(
    999.0,
    234.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

eye_mouth_label = canvas.create_text(
    817.0,
    29.0,
    anchor="nw",
    text="EYES AND MOUTH STATE",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)


lefteye_label = canvas.create_text(
    844.0,
    154.0,
    anchor="nw",
    text="Left Eye",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)



righteye_label = canvas.create_text(
    969.0,
    154.0,
    anchor="nw",
    text="Right Eye",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

mouth_label = canvas.create_text(
    1108.0,
    154.0,
    anchor="nw",
    text="Mouth",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

gaze_label = canvas.create_text(
    817.0,
    198.0,
    anchor="nw",
    text="GAZE DIRECTION",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

headpose_label = canvas.create_text(
    460.0,
    29.0,
    anchor="nw",
    text="HEAD POSE",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

yaw_label = canvas.create_text(
    460.0,
    320.0,
    anchor="nw",
    text="Yaw",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

pitch_label = canvas.create_text(
    568.0,
    320.0,
    anchor="nw",
    text="Pitch",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

roll_label = canvas.create_text(
    687.0,
    320.0,
    anchor="nw",
    text="Roll",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

gaze_pitch_label = canvas.create_text(
    945.0,
    234.0,
    anchor="nw",
    text="Pitch",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

gaze_yaw_label = canvas.create_text(
    817.0,
    234.0,
    anchor="nw",
    text="Yaw",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

eyes_closed_string = canvas.create_text(
    460.0,
    410.0,
    anchor="nw",
    text="Closed duration: ",
    fill="#FFFFFF",
    font=("Inter", 12 * -1)
)

eyes_closed_duration = canvas.create_text(
    580.0,
    410.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("Inter", 12 * -1)
)

eyes_closed_unit = canvas.create_text(
    610.0,
    410.0,
    anchor="nw",
    text="s",
    fill="#FFFFFF",
    font=("Inter", 12 * -1)
)

drowsiness_intensity = canvas.create_text(
    741.0,
    410.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("Inter", 12 * -1)
)

drowsiness_unit = canvas.create_text(
    761.0,
    410.0,
    anchor="nw",
    text="%",
    fill="#FFFFFF",
    font=("Inter", 12 * -1)
)

distraction_intensity = canvas.create_text(
    741.0,
    480.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

distraction_unit = canvas.create_text(
    761.0,
    480.0,
    anchor="nw",
    text="%",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)


drowsiness_label = canvas.create_text(
    449.0,
    372.0,
    anchor="nw",
    text="Drowsiness",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

canvas.create_text(
    449.0,
    441.0,
    anchor="nw",
    text="Distraction",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

prediction_label = canvas.create_text(
    449.0,
    520.0,
    anchor="nw",
    text="Predicted Class:",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)

predicted_class = canvas.create_text(
    600.0,
    520.0,
    anchor="nw",
    text="None",
    fill="#FFFFFF",
    font=("Inter Bold", 15 * -1)
)


time_string = canvas.create_text(
    1111.0,
    704.0,
    anchor="nw",
    text="",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)


'''                 Buttons Configuration                    '''
entry_button_img = PhotoImage(
    file=relative_to_assets("button_1.png"))

entry_button = Button(
    image=entry_button_img ,
    borderwidth=0,
    highlightthickness=0,
    command=save_name,
    relief="flat",
    activebackground="#2C2C2E"
)

entry_button.place(
    x=1111.0,
    y=508.0,
    width=78.0,
    height=28.0
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    1015.0,
    522.0,
    image=image_image_3
)

name_entry = CTk.CTkEntry(
    window,
    width=151,
    height=28,
    corner_radius=10,  
    fg_color="#EEEEEE",  
    text_color="#000716" 
)


name_entry.place(
    x=940.0,
    y=508.0
)



select_folder_img = PhotoImage(file=relative_to_assets("button_2.png"))
folder_button = Button(
    image=select_folder_img,
    borderwidth=0,
    highlightthickness=0,
    command=select_folder,
    relief="flat",
    activebackground="#2C2C2E"
)
folder_button.place(
    x=939.0,
    y=549.0,
    width=127.0,
    height=28.0
)

capture_button_img = PhotoImage(
    file=relative_to_assets("Button_Capture.png"))
capture_button = Button(
    image=capture_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=toggle_capture,
    activebackground="#2C2C2E",
    relief="flat"
)
capture_button.place(
    x=462.0,
    y=576.0,
    width=127.0,
    height=28.0
)

''' front camera switch '''
front_off_img = PhotoImage(file=relative_to_assets("button_4.png"))
on_img = PhotoImage(file=relative_to_assets("Button_on.png"))

front_button = Button(
    image=on_img,
    borderwidth=0,
    highlightthickness=0,
    command=toggle_front,
    activebackground="#2C2C2E",
    relief="flat"
)

front_button.place(
    x=127.0,
    y=29.0,
    width=49.0,
    height=25.0
)

''' side camera switch '''
side_off_img = PhotoImage(file=relative_to_assets("button_5.png"))

side_button = Button(
    image=on_img,
    borderwidth=0,
    highlightthickness=0,
    command=toggle_side,
    activebackground="#2C2C2E",
    relief="flat"
)

side_button.place(
    x=127.0,
    y=371.0,
    width=49.0,
    height=25.0
)

start_recording_img = PhotoImage(file=relative_to_assets("Button_start_recording.png"))
stop_recording_img = PhotoImage(file=relative_to_assets("Button_stop_recording.png"))

recording_button = Button(
    image=start_recording_img,
    borderwidth=0,
    highlightthickness=0,
    command=toggle_recording,
    activebackground="#2C2C2E",
    relief="flat"
)
recording_button.place(
    x=462.0,
    y=637.0,
    width=127.0,
    height=28.0
)

distraction_bar = CTk.CTkProgressBar(window, width = 330.64, height=9.54)
distraction_bar.place(x = 449, y = 465)

drowsiness_bar = CTk.CTkProgressBar(window, width = 330.64, height=9.54)
drowsiness_bar.place(x = 449, y = 400)

update_time()
run_drowsiness()
capture_front_video(canvas)
capture_side_video(canvas)
class_prediction()

window.resizable(True, True)
window.mainloop()
