import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import requests

import umap

import seaborn as sns

#def make_u_map(data, targets):
def make_umap():

    digits = load_digits()
    target = digits.target

    #csv_data = requests.get(
    #    'https://www.openml.org/data/get_csv/18238735/phpnBqZGZ'
    #)
    #with open('C:/Users/zmh001/Downloads/fashion-mnist.csv', 'w') as f:
    #    f.write(csv_data.text)

    #source_df = pd.read_csv('C:/Users/zmh001/Downloads/fashion-mnist.csv')

    #data = source_df.iloc[:, :784].values.astype(np.float32)
    #target = source_df['class'].values

    #reducer = umap.UMAP(random_state=42)
    #embedding = reducer.fit_transform(data)

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
    embedding = reducer.transform(digits.data)

    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert (np.all(embedding == reducer.embedding_))
    print(embedding.shape)
    print(embedding)



    return embedding , target #digits.target
    #plt.figure()
    #plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)

    #plt.gca().set_aspect('equal', 'datalim')
    #plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    #plt.title('UMAP projection of the digits dataset', fontsize=24)
    #plt.show()