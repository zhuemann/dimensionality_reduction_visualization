import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import requests

import umap
from matplotlib.widgets import Cursor

import seaborn as sns
#import ipdb


class MyUmap():
    def __init__(self):
        self.data = None
        self.embedding = None
        self.target = None
        self.images = None
        self.DATASET=None
        self.image_res={1:8,2:28,3:32}
        self.kidx2D=None
        self.kidxHD=None

    def load_data(self,DATASET):
        print('DATASET ',DATASET)
        self.DATASET=DATASET
        if DATASET==1:
            print('loading digits dataset')
            from sklearn.datasets import load_digits
            digits = load_digits()
            self.data = digits.data
            self.target = digits.target
            print(self.data.shape,self.target.shape)

        elif DATASET==2:
            print('loading fmnist dataset')
            from sklearn.datasets import fetch_openml
            fmnist = fetch_openml(data_id = 40996)
            print("mnist is loaded!")
            self.data = fmnist.data.to_numpy()[:5000]
            self.target = fmnist.target.to_numpy(int)[:5000]
            print(self.data.shape,self.target.shape)

        elif DATASET==3:
            print('loading cifar10 dataset')
            from sklearn.datasets import fetch_openml
            cifar = fetch_openml(data_id = 40926)
            self.data = cifar.data.to_numpy(int)[:1000]
            self.target = cifar.target.to_numpy(int)[:1000]
            print(self.data.shape,self.target.shape)

        else:
            path = DATASET
            print('loading custom dataset - can handle single csv file, single text file and image folders')
            # path = 'train'
            if path[-3:]=='csv':
              df = pd.read_csv(path, header = 0)
              df = df._get_numeric_data()
              numpy_array = df.to_numpy()
              data = numpy_array[:,:-1]
              target = numpy_array[:,-1]

            elif path[-3:]=='txt':
              df = pd.read_csv(path, header = 0, delimiter = " ")
              df = df._get_numeric_data()
              numpy_array = df.to_numpy()
              data = numpy_array[:,:-1]
              target = numpy_array[:,-1]

            else: # path to directory containing sub directories containing images of each class
              from os import listdir
              from os.path import isfile, join, isdir
              import re
              import matplotlib.pyplot as plt

              category_folders = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
              data = []
              target = []
              for label, category in enumerate(category_folders):
                files = [join(category, f) for f in listdir(join(category)) if isfile(join(category, f))]
                for file in files:
                    img = plt.imread(file)
                    data.append(img)
                    target.append(label)
              from sklearn.utils import shuffle
              data, target = shuffle(np.array(data), np.array(target), random_state=0)
              data = data.reshape(data.shape[0],-1)
            self.data = data
            self.target = target

    def make_umap(self):

        reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
                            force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
                            local_connectivity=1.0, low_memory=False, metric='euclidean',
                            metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
                            n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
                            output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
                            set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
                            target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
                            transform_queue_size=4.0, transform_seed=88, unique=False, verbose=False)

        reducer.fit(self.data)
        self.embedding = reducer.transform(self.data)

        assert (np.all(self.embedding == reducer.embedding_))

        self.precompute_KNN()

    def KNN(self, k, point, embedding):
        return np.argsort(np.linalg.norm(embedding - point[np.newaxis, :], axis=-1))[1:k + 1]

    def precompute_KNN(self):
        k = 100
        allpts2D=[]
        allptsHD=[]
        for query in range(len(self.embedding)):
            kidx2D = self.KNN(k, self.embedding[query], self.embedding)
            kidxHD = self.KNN(k, self.data[query], self.data)
            allpts2D.append(kidx2D)
            allptsHD.append(kidxHD)
        self.kidx2D = np.array(allpts2D)
        self.kidxHD = np.array(allptsHD)
        print('precomputed knn ',self.kidx2D.shape)


    def show_distortion(self, fig, k=1):
        distortion = []
        for query in range(len(self.embedding)):

            set2D = set(self.kidx2D[query,:k])
            setHD = set(self.kidxHD[query,:k])

            i = set2D.intersection(setHD)
            score = 1 - (len(i) / len(set2D))
            distortion.append(score)

        distortion = np.array(distortion)

        fig.clear()
        a_sub = fig.add_subplot(111)
        sp = a_sub.scatter(self.embedding[:, 0], self.embedding[:, 1], c=distortion, cmap='seismic', s=5)
        fig.colorbar(sp)
        a_sub.set_title('Distortion Map of the Digits dataset for neighbors={}'.format(k), fontsize=8);
        # cbar.ax.set_ylabel('Distortion (Fraction of neighbors that are different in 2D compared to high dimensions)',fontsize=6)


    def show_classes(self, fig):

        fig.clear()
        # cursor = Cursor(a_sub, horizOn=True, vertOn=True, useblit=True,
        #                 color='r', linewidth=1)
        a_sub = fig.add_subplot(111)
        sp = a_sub.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.target, cmap='Spectral', s=5)
        # plt.gca().set_aspect('equal', 'datalim')
        fig.colorbar(sp, boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        a_sub.set_title('UMAP projection of the Digits dataset', fontsize=8)

    def generate_sidepanel(self):
        fig_2 = plt.figure(figsize=(4, 2), dpi=100)
        a_sub_2 = fig_2.add_subplot(241)
        a_sub_2 = fig_2.add_subplot(242)
        a_sub_2 = fig_2.add_subplot(243)
        a_sub_2 = fig_2.add_subplot(244)
        a_sub_2 = fig_2.add_subplot(245)
        a_sub_2 = fig_2.add_subplot(246)
        a_sub_2 = fig_2.add_subplot(247)
        a_sub_2 = fig_2.add_subplot(248)
        return fig_2

    # def show_sidepanel_data(self, fig_2, point):

    #     def KNN(k, point, embedding):
    #         return np.argsort(np.linalg.norm(embedding - point[np.newaxis, :], axis=-1))[1:k + 1]

    #     nearest_2d_points = KNN(4, point, self.embedding)
    #     nearest_hd_points = KNN(4, point, self.data)
    #     print(nearest_2d_points)
    #     print(nearest_hd_points)

    #     for i in range(0, 4):
    #         fig_2.axes[i].imshow(self.data[nearest_2d_points[i]].reshape(8, 8))
    #         fig_2.axes[i + 4].imshow(self.data[nearest_hd_points[i]].reshape(8, 8))

    def show_click_response(self, fig, canvas, fig_2, fig_3, k=4, point=None):

        closest_idx = self.KNN(1, point, self.embedding)[0]
        point = self.embedding[closest_idx]
        nearest_2d_points = self.kidx2D[closest_idx,:max(4,k)]
        nearest_hd_points = self.kidxHD[closest_idx,:max(4,k)]

        a_sub = fig.axes[0]
        abc = a_sub.scatter(self.embedding[closest_idx, 0], self.embedding[closest_idx, 1], c='k',marker='*', s=100)
        abc2 = a_sub.scatter(self.embedding[nearest_hd_points[:k], 0], self.embedding[nearest_hd_points[:k], 1], c='k', s=25)

        canvas.draw()
        abc.remove()
        abc2.remove()

        res = self.image_res[self.DATASET]

        for i in range(4):
            fig_2.axes[i].imshow(self.data[nearest_2d_points[i]].reshape(res, res))
            fig_2.axes[i + 4].imshow(self.data[nearest_hd_points[i]].reshape(res, res))

        fig_3.axes[0].imshow(self.data[closest_idx].reshape(res, res))



    # def show_clickedpoint(self, point):
    #     fig = plt.figure(figsize=(2, 2),dpi=100)
    #     sub_plot = fig.add_subplot(111)
    #     # Shows the image for the closest point clicked
    #     if np.any(point != None):
    #         def KNN(k, point, embedding):
    #             return np.argsort(np.linalg.norm(embedding - point[np.newaxis, :], axis=-1))[1:k + 1]
    #         print(point)
    #         closest_idx = KNN(1, point, self.embedding)[0]
    #         sub_plot.imshow(self.data[closest_idx].reshape(8, 8))
    #     return fig







