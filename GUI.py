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
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = ttk.Button(self, text="UMap",
                            command=lambda: controller.show_frame(PageOne))
        button.place(x=1,y=1)

        options = ["MNIST Digits", "MNIST Fashion", "Select Your Own"]
        value_inside = tk.StringVar(self)
        value_inside.set("Select an Option")
        drop = tk.OptionMenu(self, value_inside, *options)
        drop.place(x=100,y=100)

        def print_answers():
            print("Selected Option: {}".format(value_inside.get()))
            if value_inside.get() == options[0]:
                print("put Umap digits call here")
            if value_inside.get() == options[1]:
                print("put Umap fashion call here")
            if value_inside.get() == options[2]:
                print("put select your own pathing fucntion here")
                browseFiles()

            return None
        submit_button = tk.Button(self, text='Submit', command=print_answers)
        submit_button.place(x=100,y=140)

        def browseFiles():
            filename = tk.filedialog.askopenfilename(initialdir="/", title="Select a File",
                                                     filetypes=(("Text files","*.txt*"), ("all files","*.*")))

        instructions = tk.Label(self, text="1. Select your data set"
                                           "2. Check out your UMAP")
        instructions.place(x=250,y=250)

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
        myumap.make_umap()
        fig = myumap.show_classes()

        def update_canvas(fig):
            canvas = FigureCanvasTkAgg(fig, self)
            canvas.draw()
            canvas.get_tk_widget().place(x=10, y=40)
            #toolbar = NavigationToolbar2Tk(canvas, self)
            #toolbar.update()
            #canvas._tkcanvas.pack(side=tk.TOP, fill=None, expand=False)
            canvas._tkcanvas.place(x=10, y=40)

        update_canvas(fig)
        toolbar = NavigationToolbar2Tk(fig.canvas, self)
        toolbar.update()
        def update_canvas_sidepanel(fig_2):
            canvas = FigureCanvasTkAgg(fig_2, self)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.RIGHT, fill=None, expand=False)
            canvas._tkcanvas.pack(side=tk.RIGHT, fill=None, expand=False)
            canvas._tkcanvas.place(x=420, y=240)


        fig_2 = myumap.show_sidepanel()
        update_canvas_sidepanel(fig_2)

        def update_canvas_clickpoint(fig_3):
            canvas = FigureCanvasTkAgg(fig_3, self)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.RIGHT, fill=None, expand=False)
            canvas._tkcanvas.pack(side=tk.RIGHT, fill=None, expand=False)
            canvas._tkcanvas.place(x=600, y=40)

        fig_3 = myumap.show_clickedpoint(point=None)
        update_canvas_clickpoint(fig_3)


        def switch_distortion_class():
            if btn_text.get() == "Show Distortions":
                k = scalevar.get()
                fig = myumap.show_distortion(k)
                update_canvas(fig)
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.close(fig)
                btn_text.set("Show Classes")
            else:
                fig = myumap.show_classes()
                update_canvas(fig)
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.close(fig)
                btn_text.set("Show Distortions")

        def update_k(value=None):
            if btn_text.get() == "Show Classes":
                k = scalevar.get()
                print('k = ',k)
                fig = myumap.show_distortion(k)
                update_canvas(fig)
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.close(fig)

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
            # # updates the plots for closest neighbors
            fig_2 = myumap.show_sidepanel_click(point)
            update_canvas_sidepanel(fig_2)
            plt.close(fig_2)
            fig_3 = myumap.show_clickedpoint(point)
            update_canvas_clickpoint(fig_3)
            plt.close(fig_3)


        fig.canvas.mpl_connect('button_press_event', onclick)



def create_GUI():
    app = NearNeighborVisualization()
    app.geometry("850x470")

    app.mainloop()
