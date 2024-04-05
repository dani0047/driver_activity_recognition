#Import tkinter modules
import tkinter as tk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog
import customtkinter as CTk

# Other necessary libraries
import cv2 as cv
from PIL import Image, ImageTk, ImageDraw
import time
from pathlib import Path
from datetime import datetime
import os
import load_model as lm

CTk.deactivate_automatic_dpi_awareness()

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\LENOVO\Desktop\Uni\Year 4\Final Year Project\gui\build\assets\frame0")

front_file_dir = r"C:\Users\LENOVO\Desktop\Uni\Year 4\Final Year Project\Model\training_log\single_front_refine_all"
side_file_dir = r"C:\Users\LENOVO\Desktop\Uni\Year 4\Final Year Project\Model\training_log\side_refined_all"
dual_file_dir = r"C:\Users\LENOVO\Desktop\Uni\Year 4\Final Year Project\Model\training_log\dual_refined_all"

front_model = lm.LoadModel('mobilenetv3l', front_file_dir, 'sam-ddd')
side_model = lm.LoadModel('mobilenetv3l', side_file_dir, 'sam-ddd')
dual_model = lm.LoadModel('twostreammobilenet', dual_file_dir, 'sam-ddd')

###### Initiliase parameters ########
tester_name = None
front_mode = True
side_mode = False
front_frame = None
side_frame = None
result = None
front_frame_pil = None
side_frame_pil = None
front_cap = None
side_cap = None
folder_selected = None
record_status = False
ellipse_front = None
ellipse_side = None
front_writer = None
side_writer = None
actual_front_fps = 0
actual_side_fps = 0

results_queue = Queue()

#Toggle Function
def save_name():
    global tester_name
    tester_name = name_entry.get()
    entry_button.config(image= entry_button_img, activebackground="#2C2C2E")
    canvas.itemconfig(name_display, text=tester_name)

def select_folder():
    global folder_selected
    folder_selected = filedialog.askdirectory()
    print(folder_selected)
    # capture_front_video(canvas)
    # capture_side_video(canvas)
    canvas.itemconfig(chosen_folder, text=folder_selected)
    folder_button.configure(image = select_folder_img, activebackground="#2C2C2E")

def toggle_front():
    global front_mode
    front_mode = not front_mode  # Toggle the state
    capture_front_video(canvas)
    front_button.config(image=on_img if front_mode else off_img, activebackground="#2C2C2E")

def toggle_side():
    global side_mode
    side_mode = not side_mode  # Toggle the state
    capture_side_video(canvas)
    side_button.config(image=on_img if side_mode else off_img, activebackground="#2C2C2E")

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
            else:
                pass
            if isinstance(side_frame_pil, Image.Image):  
                side_frame_pil.save(os.path.join(folder_selected, f"side_image_{tester_name}_{timestamp}.png"))
                img_captured +=2
            else:
                pass
            canvas.itemconfig(capture_text, text=f"{label_text[img_captured]} successfully captured.")

        except:
            canvas.itemconfig(capture_text, text="Error in saving image")
    else:
        canvas.itemconfig(capture_text, text="No selected folder to save image.")

def display_empty_frame(canvas, view):
    global front_window_img, side_window_img
    if view == 0:
        front_window_img = PhotoImage(file=relative_to_assets("front_frame.png"))
        front_empty = canvas.create_image(323.0, 328.0, image=front_window_img)

    if view == 1:
        side_window_img = PhotoImage(file=relative_to_assets("side_frame.png"))
        side_empty = canvas.create_image(949.0, 328.0, image=side_window_img)

def toggle_recording():
    global record_status, ellipse_front, ellipse_side, canvas, ellipse_img, actual_front_fps, actual_side_fps
    global folder_selected, front_writer, side_writer, front_cap, side_cap, tester_name
    ellipse_img = PhotoImage(file=relative_to_assets("image_5.png"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    record_idx = 0
    record_view = ['front camera', 'side camera', 'front and side cameras']
    if folder_selected is not None and os.path.isdir(folder_selected):
        if front_cap is not None:
            ret_front, _ = front_cap.read()
        else:
            ret_front = False
        if side_cap is not None:
            ret_side, _ = side_cap.read()
        else:
            ret_side = False
        if ret_front or ret_side:
            if not record_status:
                record_status = True
                
                if ret_front:
                    front_video_path = os.path.join(folder_selected, f"front_video_{tester_name}_{timestamp}.mp4")
                    # Start recording
                    fourcc= cv.VideoWriter_fourcc('m', 'p','4','v')
                    front_writer = cv.VideoWriter(front_video_path, fourcc, actual_front_fps, (600, 450))

                    # Show recording icon
                    ellipse_front = canvas.create_image(490.0, 76.0, image=ellipse_img)
                    record_idx +=1

                if ret_side:
                    side_video_path = os.path.join(folder_selected, f"side_video_{tester_name}_{timestamp}.mp4")
                    # Start recording
                    fourcc_side = cv.VideoWriter_fourcc('m', 'p','4','v')
                    side_writer = cv.VideoWriter(side_video_path, fourcc_side, actual_side_fps, (600, 450))
                    # Show recording icon
                    ellipse_side = canvas.create_image(1109.0, 76.0, image=ellipse_img)
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

def update_time():
    # Get the current time
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    canvas.itemconfig(time_string, text=str(formatted_time))
    window.after(1000, update_time)

def update_front_frame(canvas,cap, canvas_image_item, pTime=0):
    global front_mode, front_frame_pil, front_writer, actual_front_fps, front_frame, result
    if not front_mode:
        fps = 0
        canvas.itemconfig(front_fps_text, text=fps)
        return
    else:
        ret, frame = cap.read()
        actual_front_fps = cap.get(cv.CAP_PROP_FPS)
        if ret:
            frame = cv.resize(frame, (600, 450))
            cTime= time.time()
            fps = int(1/(cTime-pTime))
            pTime = cTime
            front_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
           
            cv.putText(frame,str(f"FPS:{int(fps)}"),(10,20), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv.putText(frame,str(f"Pred Class: {result}"),(10,430), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv.putText(front_frame,str(f"FPS:{int(fps)}"),(10,20), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            cv.putText(front_frame,str(f"Pred Class: {result}"),(10,430), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            if record_status:
                front_writer.write(frame)
            elif not record_status and side_writer is not None:
                front_writer.release()
                front_writer = None
            front_frame_pil = Image.fromarray(front_frame)
            front_frame_pil = front_frame_pil.resize((600, 450), Image.Resampling.LANCZOS)
            front_img = ImageTk.PhotoImage(front_frame_pil)
            canvas.itemconfig(canvas_image_item, image=front_img)
            canvas.front_image = front_img

        else:
            display_empty_frame(canvas,0)
            fps = 0
            front_frame = None
        canvas.itemconfig(front_fps_text, text=fps)

    window.after(30, update_front_frame, canvas, front_cap, canvas_image_item, pTime)


def update_side_frame(canvas,cap, canvas_image_item, pTime=0):
    global side_mode, side_frame_pil, side_writer, actual_side_fps, side_frame, result
    if not side_mode:
        fps = 0
        canvas.itemconfig(side_fps_text, text=fps)
        return
    else:
        ret, frame = cap.read()
        actual_side_fps = cap.get(cv.CAP_PROP_FPS)
        if ret:
            frame = cv.resize(frame, (600, 450))
            cTime= time.time()
            fps = int(1/(cTime-pTime))
            pTime = cTime
            side_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            cv.putText(frame,str(f"FPS:{int(fps)}"),(10,20), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv.putText(frame,str(f"Pred Class: {result}"),(10,430), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
            cv.putText(side_frame,str(f"FPS:{int(fps)}"),(10,20), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            cv.putText(side_frame,str(f"Pred Class: {result}"),(10,430), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
            if record_status:
                side_writer.write(frame)
            elif not record_status and side_writer is not None:
                side_writer.release()
                side_writer = None
            side_frame_pil = Image.fromarray(side_frame)
            side_frame_pil = side_frame_pil.resize((600, 450), Image.Resampling.LANCZOS)
            side_img = ImageTk.PhotoImage(side_frame_pil)
            canvas.itemconfig(canvas_image_item, image=side_img)
            canvas.side_image = side_img

        else:
            display_empty_frame(canvas, 1)
            fps = 0
            side_frame = None
        canvas.itemconfig(side_fps_text, text=fps)

    window.after(30, update_side_frame, canvas, side_cap, canvas_image_item, pTime)

def capture_front_video(canvas):
    global front_mode, front_cap, front_canvas_image
    if front_mode:
        if front_cap is None or front_cap.isOpened():
            front_cap = cv.VideoCapture(0)
            if not front_cap.isOpened():
                window.after(0, lambda: display_empty_frame(canvas, 0))
            else:
                front_canvas_image = canvas.create_image(323.0, 328.0, image=tk.PhotoImage())
                update_front_frame(canvas, front_cap, front_canvas_image)

    else:
        if front_cap is not None:
            front_cap.release()
        front_cap = None
        window.after(0, lambda: display_empty_frame(canvas, 0))

def capture_side_video(canvas):
    global side_mode, side_cap, side_canvas_image
    if side_mode:
        if side_cap is None or side_cap.isOpened():
            side_cap = cv.VideoCapture(1)
            if not side_cap.isOpened():
                window.after(0, lambda: display_empty_frame(canvas, 1))
            else:
                side_canvas_image = canvas.create_image(949.0, 328.0, image=tk.PhotoImage())
                update_side_frame(canvas, side_cap, side_canvas_image)

    else:
        if side_cap is not None:
            side_cap.release()
        side_cap = None
        window.after(0, lambda: display_empty_frame(canvas, 1))


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
    
            
    canvas.itemconfig(predidcted_class, text=str(result))
    window.after(100, class_prediction) 


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

front_window_img = PhotoImage(file=relative_to_assets("front_frame.png"))
front_empty = canvas.create_image(323.0, 328.0, image=front_window_img)
side_window_img = PhotoImage(file=relative_to_assets("side_frame.png"))
side_empty = canvas.create_image(949.0, 328.0, image=side_window_img)

canvas.create_text(
    45.0,
    30.0,
    anchor="nw",
    text="Predicted Class:",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

predidcted_class = canvas.create_text(
    170.0,
    30.0,
    anchor="nw",
    text="",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)



canvas.create_text(
    45.0,
    67.0,
    anchor="nw",
    text="FRONT VIEW",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

canvas.create_text(
    263.0,
    68.0,
    anchor="nw",
    text="FPS:",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)


front_fps_text = canvas.create_text(
    293.0,
    68.0,
    anchor="nw",
    text= "0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

canvas.create_text(
    649.0,
    67.0,
    anchor="nw",
    text="SIDE VIEW",
    fill="#FFFFFF",
    font=("Inter Bold", 16 * -1)
)

canvas.create_text(
    857.0,
    68.0,
    anchor="nw",
    text="FPS:",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

side_fps_text = canvas.create_text(
    887.0,
    68.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

canvas.create_text(
    57.0,
    661.0,
    anchor="nw",
    text="SAVED FOLDER:",
    fill="#FFFFFF",
    font=("Inter", 13 * -1)
)

chosen_folder = canvas.create_text(
    175,
    690.0,
    anchor="nw",
    text="No chosen folder",
    fill="#FFFFFF",
    font=("Inter", 13 * -1)
)

time_string = canvas.create_text(
    1109.0,
    709.0,
    anchor="nw",
    text="",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

record_text = canvas.create_text(
    804,
    641,
    anchor="nw",
    text="Click button to start recording",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

name_display = canvas.create_text(
    175,
    625,
    anchor="nw",
    text="",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

capture_text = canvas.create_text(
    804,
    585,
    anchor="nw",
    text="Click button to screenshot",
    fill="#FFFFFF",
    font=("Inter", 14 * -1)
)

canvas.create_text(
    57.0,
    597.0,
    anchor="nw",
    text="TESTER NAME: ",
    fill="#FFFFFF",
    font=("Inter", 13 * -1)
)


#########  Button Configuration ##########
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
    x=167.0,
    y=655.0,
    width=127.0,
    height=28.0
)

#Front button
off_img = PhotoImage(file=relative_to_assets("button_4.png"))
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
    x=152.0,
    y=64.0,
    width=49.0,
    height=25.0
)

#Side Button
side_button = Button(
    image=off_img,
    borderwidth=0,
    highlightthickness=0,
    command=toggle_side,
    activebackground="#2C2C2E",
    relief="flat"
)

side_button.place(
    x=741.0,
    y=64.0,
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
    x=661.0,
    y=635.0,
    width=127.0,
    height=28.0
)

capture_button_img = PhotoImage(file=relative_to_assets("Button_Capture.png"))

capture_button = Button(
    image=capture_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=toggle_capture,
    activebackground="#2C2C2E",
    relief="flat"
)
capture_button.place(
    x=661.0,
    y=584.0,
    width=127.0,
    height=28.0
)


entry_button_img = PhotoImage(file=relative_to_assets("button_1.png"))

entry_button = Button(
    image=entry_button_img ,
    borderwidth=0,
    highlightthickness=0,
    command=save_name,
    relief="flat",
    activebackground="#2C2C2E"
)

entry_button.place(
    x=330.0,
    y=591.0,
    width=78.0,
    height=28.0
)

name_entry = CTk.CTkEntry(
    window,
    width=151,
    height=28,
    corner_radius=10,  # Adjust the corner radius if needed
    fg_color="#EEEEEE",  # This sets the background color
    text_color="#000716"  # This sets the text color
)

name_entry.place(
    x=172.0,
    y=591.0
)


update_time()
capture_front_video(canvas)
capture_side_video(canvas)
class_prediction()
window.resizable(False, False)
window.mainloop()


