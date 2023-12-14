from tkinter import *
import cv2
import sys
import os
from PIL import ImageTk, Image 



counter = 0
flag = 0
image_type = None

class GUI(Tk):
    def __init__(self):
        super().__init__()
        # Status bar variable
        self.back_image = PhotoImage(file="backg.png")
        self.background_label = Label(self, image=self.back_image)
        self.background_label.pack(side='top', fill='both', expand='yes')
        self.status = StringVar()
        self.msg_var = StringVar()
        self.status.set("ready")
        self.msg_var.set("")
        self.status_bar = Label(self.background_label, textvar=self.status, relief=SUNKEN, anchor="w")
        self.msg_box = Text(self.background_label, height=18, width=110,font=('Arial', 12), bg="grey")
        self.cam_win = Label(self.background_label, textvar=self.status, relief=SUNKEN, anchor="n")

    def set_status(self, status_value, color):
        self.status.set(f"{status_value}")
        self.status_bar.config(bg=f"{color}")
        self.status_bar.update()

    def set_msg(self, message):
        self.msg_box.insert(END, f"{message}\n")
        self.msg_box.update()


    def create_statusbar_msgbox(self):
        self.status_bar.pack(fill=X, side=BOTTOM)
        self.cam_win.pack(fill=X, side=TOP)
        self.msg_box.pack(side=BOTTOM, padx=10, pady=10)

def show_frames():

   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   start_point = (5, 5)
   global flag
   global counter
   global image_type

   # represents the bottom right corner of rectangle
   end_point = (220, 220)

   if(flag):
       img = cv2image[start_point[0]:end_point[0], start_point[1]:end_point[1]]
       filename = f"{image_type}/{counter}.jpg"
       cv2.imwrite(filename, img)
       flag = 0


   # Blue color in BGR
   color = (255, 0, 0)
   # Line thickness of 2 px
   thickness = 2
   # Draw a rectangle with blue line borders of thickness of 2 px
   cv2image = cv2.rectangle(cv2image, start_point, end_point, color, thickness)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   label.after(20, show_frames)



def down(e):
    global flag
    flag = 1
    global counter
    counter += 1
    print(counter)


if __name__ == '__main__':

    image_type = sys.argv[1]

    os.makedirs(f'data/{image_type}', exist_ok=True)

    window = GUI()
    window_width = 700  # width of window
    window_height = 900  # height of window

    window.geometry(f"{window_width}x{window_height}")
    window.title("Hand Gesture Typer")
    label = window.cam_win
    cap= cv2.VideoCapture(0)

    window.bind('<Return>', down)
    show_frames()
    window.create_statusbar_msgbox()

    window.set_status("Ready", "blue")
    window.mainloop()
