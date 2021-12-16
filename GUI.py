import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from make_umap import MyUmap
from tkinter import HORIZONTAL
import numpy as np


LARGE_FONT = ("Verdana", 12)
DATASET = 1


class NearNeighborVisualization(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "NearNeighborVisualization")


        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.show_frame(StartPage)

    def show_frame(self, cont):
        F=cont
        frame = F(self.container, self)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Welcome", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        options = ["MNIST Digits", "MNIST Fashion", "CIFAR10", "Select Your Own"]
        value_inside = tk.StringVar(self)
        value_inside.set("Select an Option")
        drop = tk.OptionMenu(self, value_inside, *options)
        drop.place(x=350, y=140)

        def select_option():
            global DATASET
            print("Selected Option: {}".format(value_inside.get()))
            if value_inside.get() == options[0]:
                DATASET = 1

            if value_inside.get() == options[1]:
                DATASET = 2

            if value_inside.get() == options[2]:
                DATASET = 3

            if value_inside.get() == options[3]:
                DATASET = browseFiles() # DATASET will be path for custom dataset
            controller.show_frame(PageOne)

            return None
        submit_button = tk.Button(self, text='Load', command=select_option)
        submit_button.place(x=395, y=180)

        def browseFiles():
            current_directory = tk.filedialog.askdirectory(initialdir="/", title="Select a Directory")
            return current_directory

        welcome_message = tk.Label(self, text="Hi welcome to our dimensionality reduction tool. In this tool we will attempt"
                                                    " to help you visualization distortions")
        welcome_message.place(x=50,y=80)
        welcome_message = tk.Label(self, text="in UMAP imparted in 2D!")
        welcome_message.place(x=50,y=100)
        welcome_message_2 = tk.Label(self, text="To get started must select a dataset.")
        welcome_message_2.place(x=50,y=120)

        instruction_1 = tk.Label(self, text="Instructions:")
        instruction_1.place(x=50,y=220)

        instruction_1 = tk.Label(self, text="1. Select your data set and click 'Load'. Please wait while the dataset is being loaded and umap is being generated.")
        instruction_1.place(x=50,y=240)

        instruction_2 = tk.Label(self, text="Note: If you select your own dataset, select a Directory which contains class-wise sub-directories that contains images of that class.")
        instruction_2.place(x=50,y=260)

        instruction_3 = tk.Label(self, text="2. On the next page, select desired number neighbors")
        instruction_3.place(x=50,y=280)

        instruction_4 = tk.Label(self, text="3. Click on any point on the UMAP. You will see its high dimensional neighbors highlighted on the map.")
        instruction_4.place(x=50,y=300)

        instruction_4 = tk.Label(self, text="You will also see images of neighboring data points in the side panel.")
        instruction_4.place(x=50,y=320)

        instruction_4 = tk.Label(self, text="4. Click show distortions. You will see the distortion map for the selected number of neighbors.")
        instruction_4.place(x=50,y=340)

        instruction_4 = tk.Label(self, text="Note: Distortion is the fraction of changed neighbors while reducing dimensions to the number of original neighbors.")
        instruction_4.place(x=50,y=360)

        instruction_5 = tk.Label(self, text="5. Explore! Probe into any data point, see the variation as # neighbors is changed and visualize the neighboring images.")
        instruction_5.place(x=50,y=380)
        instruction_6 = tk.Label(self, text="Protip: Along the bottom are additional zoom and pan functionality to help"
                                            " you explore as you wish!")
        instruction_6.place(x=50,y=400)



class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)        
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.place(x=150,y=5)

        MYUMAP = MyUmap()
        MYUMAP.load_data(DATASET)
        MYUMAP.make_umap()


        fig = plt.figure(figsize=(5, 5),dpi=100)
        MYUMAP.show_classes(fig)
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().place(x=30, y=40)

        toolbar = NavigationToolbar2Tk(fig.canvas, self)
        toolbar.place(x=90, y=420)
        toolbar.update()


        fig_2 = MYUMAP.generate_sidepanel()
        canvas_2 = FigureCanvasTkAgg(fig_2, self)
        canvas_2.draw()
        canvas_2.get_tk_widget().pack(side=tk.RIGHT, fill=None, expand=False)
        canvas_2._tkcanvas.pack(side=tk.RIGHT, fill=None, expand=False)
        canvas_2._tkcanvas.place(x=420, y=227)


        fig_3 = plt.figure(figsize=(2.25, 2.25),dpi=100)
        a_sub_3 = fig_3.add_subplot(111)
        a_sub_3.get_xaxis().set_visible(False)
        a_sub_3.get_yaxis().set_visible(False)
        fig_3.suptitle('Selected Data Point')
        canvas_3 = FigureCanvasTkAgg(fig_3, self)
        canvas_3.draw()
        canvas_3.get_tk_widget().pack(side=tk.RIGHT, fill=None, expand=False)
        canvas_3._tkcanvas.pack(side=tk.RIGHT, fill=None, expand=False)
        canvas_3._tkcanvas.place(x=650, y=40)


        def switch_distortion_class():
            if btn_text.get() == "Show Distortions":
                k = scalevar.get()
                MYUMAP.show_distortion(fig,k)
                canvas.draw()
                # print(fig.axes)
                btn_text.set("Show Classes")
            else:
                MYUMAP.show_classes(fig)
                canvas.draw()
                btn_text.set("Show Distortions")

        def update_k(value=None):
            if btn_text.get() == "Show Classes":
                k = scalevar.get()
                print('k = ',k)
                MYUMAP.show_distortion(fig,k)
                canvas.draw()

        btn_text = tk.StringVar()
        btn_text.set("Show Distortions")

        scalevar = tk.IntVar()
        scalevar.set(1)
        w = tk.Scale(self, label="#Neighbors", from_=1, to=100, orient=HORIZONTAL,
                     variable=scalevar)  # orient=HORIZONTAL
        w.bind("<ButtonRelease-1>", update_k)
        w.place(x=480, y=70)

        button2 = ttk.Button(self, textvariable=btn_text,
                             command=lambda: switch_distortion_class())

        button2.place(x=460, y=140)

        def onclick(event):
            x = event.xdata
            y = event.ydata

            point = np.array([x,y])
            print(point)
            k = scalevar.get()

            MYUMAP.show_click_response(fig,canvas,fig_2,fig_3,k,point)
            canvas_2.draw()
            canvas_3.draw()


        fig.canvas.mpl_connect('button_press_event', onclick)



def create_GUI():
    app = NearNeighborVisualization()
    app.geometry("850x470")

    app.mainloop()
