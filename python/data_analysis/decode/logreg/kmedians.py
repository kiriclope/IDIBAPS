from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import numpy as np
# Load list of points for cluster analysis.
sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)

sample = np.array(sample)

centers = kmeans_plusplus_initializer(sample, 4).initialize();

# Create instance of K-Medians algorithm.
initial_medians = [[0.0, 0.1], [2.5, 0.7]]
kmedians_instance = kmedians(sample, centers)
# Run cluster analysis and obtain results.
kmedians_instance.process()
clusters = kmedians_instance.get_clusters()
medians = kmedians_instance.get_medians()
