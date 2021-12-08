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
        self.nclasses = None
        self.images = None

    def make_umap(self):
        digits = load_digits()
        self.data = digits.data
        self.target = digits.target

        self.nclasses = np.unique(self.target).shape[0]

        reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
                            force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
                            local_connectivity=1.0, low_memory=False, metric='euclidean',
                            metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
                            n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
                            output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
                            set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
                            target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
                            transform_queue_size=4.0, transform_seed=88, unique=False, verbose=False)

        reducer.fit(digits.data)
        self.embedding = reducer.transform(digits.data)

        # np.save('embed',self.embedding)
        # self.embedding = np.load('embed.npy')

        assert (np.all(self.embedding == reducer.embedding_))

        return self.embedding, self.target  # digits.target

    def show_distortion(self, k=1):
        def KNN(k, point, embedding):
            return np.argsort(np.linalg.norm(embedding - point[np.newaxis, :], axis=-1))[1:k + 1]

        embedding = self.embedding
        data = self.data
        distortion = []
        for query in range(len(embedding)):
            kidx2D = KNN(k, embedding[query], embedding)
            knn2D = embedding[kidx2D]

            kidxHD = KNN(k, data[query], data)
            knnHD = embedding[kidxHD]

            set2D = set(kidx2D)
            setHD = set(kidxHD)

            i = set2D.intersection(setHD)
            score = 1 - (len(i) / len(kidx2D))
            distortion.append(score)

        distortion = np.array(distortion)

        fig = plt.figure(figsize=(4, 4), dpi=100)
        a_sub = fig.add_subplot(111)
        sp = a_sub.scatter(embedding[:, 0], embedding[:, 1], c=distortion, cmap='seismic', s=5)
        cbar = fig.colorbar(sp)
        plt.title('Distortion Map of the Digits dataset for neighbors={}'.format(k), fontsize=8);
        # cbar.ax.set_ylabel('Distortion (Fraction of neighbors that are different in 2D compared to high dimensions)',fontsize=6)
        return fig

    def show_classes(self):
        embedding = self.embedding
        fig = plt.figure(figsize=(4, 4),
                         dpi=100)  # doesn't this create a new fig each time show class is called so it is a new instance
        a_sub = fig.add_subplot(111)
        cursor = Cursor(a_sub, horizOn=True, vertOn=True, useblit=True,
                        color='r', linewidth=1)
        sp = a_sub.scatter(embedding[:, 0], embedding[:, 1], c=self.target, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(sp, boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        plt.title('UMAP projection of the Digits dataset', fontsize=8);
        return fig

    def show_sidepanel(self):
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

    def show_sidepanel_data(self, fig_2, point):
        print("inside plotings data")

        def KNN(k, point, embedding):
            return np.argsort(np.linalg.norm(embedding - point[np.newaxis, :], axis=-1))[1:k + 1]

        nearest_2d_points = KNN(4, point, self.embedding)
        nearest_hd_points = KNN(4, point, self.data)
        print(nearest_2d_points)
        print(nearest_hd_points)

        for i in range(0, 4):
            fig_2.axes[i].imshow(self.data[nearest_2d_points[i]].reshape(8, 8))
            fig_2.axes[i + 4].imshow(self.data[nearest_hd_points[i]].reshape(8, 8))

    def show_sidepanel_click(self, point=None):

        def KNN(k, point, embedding):
            return np.argsort(np.linalg.norm(embedding - point[np.newaxis, :], axis=-1))[1:k + 1]

        print(point[np.newaxis, :].shape)
        closest_idx = KNN(1, point, self.embedding)[0]
        point = self.embedding[closest_idx]
        nearest_2d_points = KNN(4, point, self.embedding)
        nearest_hd_points = KNN(4, self.data[closest_idx], self.data)
        print(nearest_2d_points)
        print(nearest_hd_points)

        # ipdb.set_trace()
        fig_2 = plt.figure(figsize=(4, 2), dpi=100)

        for i in range(4):
            a_sub_2 = fig_2.add_subplot(241 + i)
            a_sub_2.imshow(self.data[nearest_2d_points[i]].reshape(8, 8))
        for i in range(4):
            a_sub_2 = fig_2.add_subplot(245 + i)
            a_sub_2.imshow(self.data[nearest_hd_points[i]].reshape(8, 8))

        return fig_2







