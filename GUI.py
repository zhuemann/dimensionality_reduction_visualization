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

        def update_canvas_sidepanel(fig_2, point):
            canvas = FigureCanvasTkAgg(fig_2, self)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.RIGHT, fill=None, expand=False)
            canvas._tkcanvas.pack(side=tk.RIGHT, fill=None, expand=False)
            canvas._tkcanvas.place(x=420, y=240)

            # will plot the individual nearest neighbors if given a point
            if point != None:
                myumap.show_sidepanel_data(fig_2, point)


        fig_2 = myumap.show_sidepanel()
        update_canvas_sidepanel(fig_2, point=None)

        def switch_distortion_class():
            if btn_text.get() == "Show Distortions":
                k = scalevar.get()
                fig = myumap.show_distortion(k)
                update_canvas(fig)
                plt.close(fig)
                btn_text.set("Show Classes")
            else:
                fig = myumap.show_classes()
                update_canvas(fig)
                plt.close(fig)
                btn_text.set("Show Distortions")

        def update_k(value=None):
            if btn_text.get() == "Show Classes":
                k = scalevar.get()
                print('k = ',k)
                fig = myumap.show_distortion(k)
                update_canvas(fig)
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

            print([x,y])
            embedding_x = myumap.embedding[:, 0]
            embedding_y = myumap.embedding[:, 1]
            # calculates the closest point to the position clicked
            def find_index_of_nearest_xy(y_array, x_array, y_point, x_point):
                distance = (y_array - y_point) ** 2 + (x_array - x_point) ** 2
                idx = np.where(distance == distance.min())
                return idx[0]
            closest_index = find_index_of_nearest_xy(embedding_y, embedding_x, y, x)
            print(closest_index)
            # updates the plots for closest neighbors
            update_canvas_sidepanel(fig_2, point = closest_index)

        print(fig)
        fig.canvas.mpl_connect('button_press_event', onclick)



def create_GUI():
    app = NearNeighborVisualization()
    app.geometry("850x470")

    app.mainloop()
