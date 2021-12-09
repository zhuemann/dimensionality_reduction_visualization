import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import scipy.spatial

import tkinter as tk
from tkinter import ttk
from make_umap import MyUmap
#from tkinter import *
from tkinter import HORIZONTAL

import numpy as np


LARGE_FONT = ("Verdana", 12)
DATASET = 2

class NearNeighborVisualization(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        #tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "NearNeighborVisualization")


        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)


        self.frames = {}

        for F in (StartPage, PageOne):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Welcome", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = ttk.Button(self, text="UMap",
                            command=lambda: controller.show_frame(PageOne))
        button.place(x=380, y=360)

        options = ["MNIST Digits", "MNIST Fashion", "Select Your Own"]
        value_inside = tk.StringVar(self)
        value_inside.set("Select an Option")
        drop = tk.OptionMenu(self, value_inside, *options)
        drop.place(x=350, y=140)

        def print_answers():
            global DATASET
            print("Selected Option: {}".format(value_inside.get()))
            if value_inside.get() == options[0]:
                print("put Umap digits call here")
                DATASET = 1
            if value_inside.get() == options[1]:
                print("put Umap fashion call here")
                DATASET = 2
            if value_inside.get() == options[2]:
                print("put select your own pathing fucntion here")
                browseFiles()
                DATASET = '/<path to dataset>' # DATASET will be path for custom dataset

            return None
        submit_button = tk.Button(self, text='Load', command=print_answers)
        submit_button.place(x=395, y=180)

        def browseFiles():
            filename = tk.filedialog.askopenfilename(initialdir="/", title="Select a File",
                                                     filetypes=(("Text files","*.txt*"), ("all files","*.*")))
        welcome_message = tk.Label(self, text="Hi welcome to our dimensionality reduction tool. In this tool we will attempt"
                                                    " to help you visualization distortions in UMAP imparted in 2D!")
        welcome_message.place(x=50,y=100)
        welcome_message_2 = tk.Label(self, text="To get started must selected a dataset.")
        welcome_message_2.place(x=50,y=120)
        instruction_1 = tk.Label(self, text="1. Select your data set")
        instruction_1.place(x=50,y=220)

        instruction_2 = tk.Label(self, text="2. Check out your UMAP")
        instruction_2.place(x=50,y=240)

        instruction_3 = tk.Label(self, text="3. Select number neighbors by keeping the number of neighbors low you "
                                              "focus on local structure as k increases more global structure is probed")
        instruction_3.place(x=50, y=260)
        instruction_4 = tk.Label(self, text="4. Click show distortions, when k-neighbors is low local structure is "
                                            "probed when k is high more global structure is probed")
        instruction_4.place(x=50,y=280)
        instruction_5 = tk.Label(self, text="5. Explore! By clicking on the map you can see the point you clicked closest"
                                            "to in high dimension and its 4 nearest neighbors in high and low dimension")
        instruction_5.place(x=50, y=300)
        instruction_6 = tk.Label(self, text="Protip: Along the bottom are additional zoom and pan functionality to help"
                                            " you explore as you wish")
        instruction_6.place(x=50,y=320)



class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.place(x=50,y=1)

        #Rahul's code
        myumap = MyUmap()
        myumap.load_data(DATASET)
        myumap.make_umap()


        fig = plt.figure(figsize=(4, 4),dpi=100)
        myumap.show_classes(fig)
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().place(x=10, y=40)
        # canvas._tkcanvas.place(x=10, y=40)

        toolbar = NavigationToolbar2Tk(fig.canvas, self)
        toolbar.update()


        fig_2 = myumap.generate_sidepanel()
        canvas_2 = FigureCanvasTkAgg(fig_2, self)
        canvas_2.draw()
        canvas_2.get_tk_widget().pack(side=tk.RIGHT, fill=None, expand=False)
        canvas_2._tkcanvas.pack(side=tk.RIGHT, fill=None, expand=False)
        canvas_2._tkcanvas.place(x=420, y=240)


        fig_3 = plt.figure(figsize=(2, 2),dpi=100)
        fig_3.add_subplot(111)
        canvas_3 = FigureCanvasTkAgg(fig_3, self)
        canvas_3.draw()
        canvas_3.get_tk_widget().pack(side=tk.RIGHT, fill=None, expand=False)
        canvas_3._tkcanvas.pack(side=tk.RIGHT, fill=None, expand=False)
        canvas_3._tkcanvas.place(x=600, y=40)


        def switch_distortion_class():
            if btn_text.get() == "Show Distortions":
                k = scalevar.get()
                myumap.show_distortion(fig,k)
                canvas.draw()
                print(fig.axes)
                btn_text.set("Show Classes")
            else:
                myumap.show_classes(fig)
                canvas.draw()
                btn_text.set("Show Distortions")

        def update_k(value=None):
            if btn_text.get() == "Show Classes":
                k = scalevar.get()
                print('k = ',k)
                myumap.show_distortion(fig,k)
                canvas.draw()

        btn_text = tk.StringVar()
        btn_text.set("Show Distortions")

        scalevar = tk.IntVar()
        scalevar.set(1)
        w = tk.Scale(self, label="neighbors", from_=1, to=100, orient=HORIZONTAL,
                     variable=scalevar)  # orient=HORIZONTAL
        w.bind("<ButtonRelease-1>", update_k)
        w.place(x=450, y=40)

        button2 = ttk.Button(self, textvariable=btn_text,
                             command=lambda: switch_distortion_class())

        button2.place(x=450, y=110)

        def onclick(event):
            x = event.xdata
            y = event.ydata

            point = np.array([x,y])
            print(point)
            k = scalevar.get()
            # # updates the plots for closest neighbors
            myumap.show_click_response(fig,canvas,fig_2,fig_3,k,point)
            canvas_2.draw()
            canvas_3.draw()


        fig.canvas.mpl_connect('button_press_event', onclick)



def create_GUI():
    app = NearNeighborVisualization()
    app.geometry("850x470")

    app.mainloop()
