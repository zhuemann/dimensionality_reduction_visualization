import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from make_umap import make_umap
#from tkinter import *
from tkinter import HORIZONTAL

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
        #button.pack()




class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        #button1.pack()
        button1.place(x=50,y=1)

        fig = Figure(figsize=(4, 4), dpi=100)

        a_sub = fig.add_subplot(111)


        embedding, targets = make_umap()
        a_sub.scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap='Spectral', s=5)


        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        #canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=None, expand=False)
        canvas.get_tk_widget().place(x=10, y=40)

        #toolbar = NavigationToolbar2Tk(canvas, self)
        #toolbar.update()
        #canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #canvas._tkcanvas.pack(side=tk.TOP, fill=None, expand=False)
        canvas._tkcanvas.place(x=10, y=40)

        fig_2 = Figure(figsize=(4, 2), dpi=100)

        a_sub_2 = fig_2.add_subplot(241)
        a_sub_2 = fig_2.add_subplot(242)
        a_sub_2 = fig_2.add_subplot(243)
        a_sub_2 = fig_2.add_subplot(244)
        a_sub_2 = fig_2.add_subplot(245)
        a_sub_2 = fig_2.add_subplot(246)
        a_sub_2 = fig_2.add_subplot(247)
        a_sub_2 = fig_2.add_subplot(248)
        canvas = FigureCanvasTkAgg(fig_2, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=None, expand=False)
        canvas._tkcanvas.pack(side=tk.RIGHT, fill=None, expand=False)
        canvas._tkcanvas.place(x=420, y=240)

        w = tk.Scale(self, label="neighbors", from_=0, to=100, orient=HORIZONTAL) #orient=HORIZONTAL
        w.place(x=450, y=40)

        button2 = ttk.Button(self, text="Show Distortions",
                             command=lambda: controller.show_frame(StartPage))

        button2.place(x=450, y=110)

        #canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)



#app = NearNeighborVisualization()
#app.geometry("850x470")
#app.mainloop()

def create_GUI():
    app = NearNeighborVisualization()
    app.geometry("850x470")
    app.mainloop()