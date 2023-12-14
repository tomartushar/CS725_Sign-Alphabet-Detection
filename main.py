from tkinter import *
import cv2
from PIL import ImageTk, Image
import time
import numpy as np
from tensorflow.keras.models import load_model


global model
global counter
global flag

model = load_model('model_s1_.h5')
counter = 0
flag=True


class GUI(Tk):
    def __init__(self):
        super().__init__()
        self.back_image = PhotoImage(file="backg.png")
        self.background_label = Label(self, image=self.back_image)
        self.background_label.pack(side='top', fill='both', expand='yes')
        self.status = StringVar()
        self.msg_var = StringVar()
        self.status.set("ready")
        self.msg_var.set("")
        self.out_text = ''
        self.status_bar = Label(self.background_label, textvar=self.status, relief=SUNKEN, anchor="w")
        self.msg_box = Text(self.background_label, height=18, width=110,font=('Arial', 12), bg="grey")
        self.cam_win = Label(self.background_label, textvar=self.status, relief=SUNKEN, anchor="n")

    def set_status(self, status_value):
        self.status.set(f"{status_value}")
        self.status_bar.config(bg="blue")
        self.status_bar.update()

    def set_msg(self, message):
        if message == 'nothing':
            pass
        elif message == 'del':
            print(self.out_text)
            self.out_text = self.out_text[:-1]
            print(self.out_text)
        elif message == 'space':
            self.out_text += ' '
        else:
            self.out_text += message
        self.msg_box.delete("1.0", "end")
        self.msg_box.insert(END, f"{self.out_text}")
        self.set_status(f'Detected "{message}"')
        self.msg_box.update()


    def create_statusbar_msgbox(self):
        self.status_bar.pack(fill=X, side=BOTTOM)
        self.cam_win.pack(fill=X, side=TOP)
        self.msg_box.pack(side=BOTTOM, padx=10, pady=10)


def model_func(image, classes):
    global counter
    counter+=1
    print(image.shape)
    image=image[5:205,5:205]
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img=np.array(img)
    img=cv2.resize(img,(200,200),interpolation = cv2.INTER_AREA)
    img = img.astype('float32')
    img_=img/255
    img_=np.expand_dims(img_,axis=0)
    result=model.predict(img_)
    arr=np.array(result[0])
    max_prob=arr.argmax(axis=0)
    return classes[max_prob]


def show_frames(label, classes):
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   start_point = (5, 5)
   global flag
   end_point = (205, 205)
   seconds = time.time()
   seconds = int(seconds*1000)%1000

   color = (255, 0, 0)
   if seconds < 500:
       color = (0,255,0)
       if flag:
           charac = model_func(cv2image, classes)
           window.set_msg(charac)
           print(charac)
           
           flag = False
   else:
       flag = True
   thickness = 2
   cv2image = cv2.rectangle(cv2image, start_point, end_point, color, thickness)
   img = Image.fromarray(cv2image)
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   label.after(50, show_frames)



if __name__ == '__main__':
    window = GUI()

    classes = [
        'nothing', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
        'V', 'W', 'X', 'Y', 'Z', 'del', 'space' 
    ]

    window_width = 700  # width of window
    window_height = 900  # height of window

    window.geometry(f"{window_width}x{window_height}")
    window.title("Hand Gesture Typer")

    label = window.cam_win
    cap= cv2.VideoCapture(0)
    show_frames(label, classes)
    window.create_statusbar_msgbox()
    window.mainloop()
