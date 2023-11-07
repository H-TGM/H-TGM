import TensorClustModSparse
import numpy as np
import Cluster_Ensembles.Cluster_Ensembles as CE
from scipy import sparse
from sklearn.utils import check_random_state, check_array
import tensorflow as tf
from coclust.coclustering import CoclustMod
import networkx as nx
import tensorflow as tf
from tensorflow.python.client import device_lib

gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
print('GPU DEVICES:\n  {}'.format(gpus))


def modularity_byhands(adjMatrix, Nclust, vector_clustering):
    n = adjMatrix.shape[0]
    G = nx.from_numpy_matrix(adjMatrix)
    MM = nx.modularity_matrix(G)

    # c = len(np.unique(vector_clustering))
    W = np.zeros((n, Nclust))
    minvector_clustering = np.amin(vector_clustering)
    print('n', n)
    print('vector_clustering ', vector_clustering)
    print('len(vector_clustering)', len(vector_clustering))
    W[np.arange(n), np.asarray(vector_clustering)] = 1
    CC = W.dot(W.T)
    with tf.device('/device:GPU:0'):
        b_sparse = tf.convert_to_tensor(CC, np.float32)
        a_sparse = tf.convert_to_tensor(MM, np.float32)
        res = tf.compat.v1.sparse_matmul(a_sparse, b_sparse)

    resNumpy = res.numpy()
    resMod = np.sum(resNumpy)

    return resMod / np.sum(adjMatrix)


class HTGM():
    """Co-clustering by direct maximization of graph modularity.

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Number of co-clusters to form

    init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column labels

    max_iter : int, optional, default: 20
        Maximum number of iterations

    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs in terms of modularity.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    tol : float, default: 1e-9
        Relative tolerance with regards to modularity to declare convergence

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        Bicluster label of each row


    modularity : float
        Final value of the modularity

    modularities : list
        Record of all computed modularity values for all iterations

    References
    ----------
    * Ailem M., Role F., Nadif M., Co-clustering Document-term Matrices by \
    Direct Maximization of Graph Modularity. CIKM 2015: 1807-1810
    """

    def __init__(self, n_clusters=[5, 4, 3], n_cluster_consensus=10, init=10, max_iter=100, n_init=1,
                 n_init_level1=10, bool_fuzzy=False, fusion_method='Add',
                 tol=1e-9, random_state=None):
        self.n_clusters = n_clusters
        self.n_cluster_consensus = n_cluster_consensus
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_init_level1 = n_init_level1
        self.tol = tol
        self.random_state = random_state
        self.bool_fuzzy = bool_fuzzy
        self.fusion_method = fusion_method
        self.row_labels_ = None
        self.hierarchical_labels_ = None
        self.modularity = None
        self.modularitiesLevel = None
        self.modularitiesIter = None
        self.hierarchicalModularity = None
        self.paritionIteration = None

    def fit(self, X, y=None):
        # first level clusterin
        hierarchical_label = []
        listModularitiesLevel = []
        modularitiesIter = []
        runs = self.n_init_level1
        cluster_runs = np.zeros((runs, (X[0]).shape[0]))
        matrixConsensus = np.zeros(((X[0]).shape[0], (X[0]).shape[0]))
        modularitiesIterLevel = []
        for run in range(runs):
            print('Run level 1 :', run)
            model = TensorClustModSparse.TensorCoclustMod(n_clusters=self.n_clusters[0], init=self.init,
                                                          bool_fuzzy=False, fusion_method='Add')
            model.fit(X)
            modularitiesIterLevel.append(model)
            phiK = np.asarray(model.row_labels_)
            print(np.unique(phiK))
            cluster_runs[run, :] = np.asarray(phiK)
            b = np.zeros((phiK.size, phiK.max() + 1))
            b[np.arange(phiK.size), phiK] = 1
            matrixConsensus = matrixConsensus + b.dot(b.T)

        # modelConsenus = CoclustMod(n_clusters=self.n_clusters[0], max_iter=100)
        # modelConsenus.fit(matrixConsensus)
        # consensus_clustering_labels = modelConsenus.row_labels_
        print('cluster_runs', cluster_runs.shape)
        print('message', self.n_clusters[0])
        consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True,
                                                           nclass=self.n_clusters[0])

        hierarchical_label.append(consensus_clustering_labels)

        modularityC = 0
        for v in range(len(X)):
            adjMatV = X[v]
            modularityC = modularityC + modularity_byhands(adjMatV, self.n_clusters[0], consensus_clustering_labels)

        # consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True, N_clusters_max=self.n_cluster_consensus)

        listModularitiesLevel.append(modularityC)
        modularitiesIter.append(modularitiesIterLevel)

        actuelCLustering = consensus_clustering_labels
        XH = X.copy()
        for k in range(1, len(self.n_clusters)):
            newLevel = []
            listModularitiesClust = []
            modularitiesIterLevel = []
            for clus in np.unique(actuelCLustering):
                modularitiesIterClust = []
                X_clus = []
                for v in range(len(XH)):
                    X_v = XH[v].copy()
                    X_clus_v = X_v[actuelCLustering == clus, :]
                    X_clus_v = X_clus_v[:, actuelCLustering == clus]

                    X_clus.append(X_clus_v)

                cluster_runs = np.zeros((runs, (X_clus[0]).shape[0]))
                matrixConsensus = np.zeros(((X_clus[0]).shape[0], (X_clus[0]).shape[0]))
                for run in range(runs):
                    print('Run level ' + str(k + 1), run)
                    model = TensorClustModSparse.TensorCoclustMod(n_clusters=self.n_clusters[k], bool_fuzzy=False,
                                                                  fusion_method='Add')
                    model.fit(X_clus)
                    modularitiesIterClust.append(model)
                    phiK = np.asarray(model.row_labels_)
                    cluster_runs[run, :] = np.asarray(phiK)
                    b = np.zeros((phiK.size, phiK.max() + 1))
                    b[np.arange(phiK.size), phiK] = 1
                    matrixConsensus = matrixConsensus + b.dot(b.T)

                # modelConsenus = CoclustMod(n_clusters=self.n_clusters[k], max_iter=100)
                # modelConsenus.fit(matrixConsensus)
                # consensus_clustering_labels = modelConsenus.row_labels_
                consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True,
                                                                   nclass=self.n_clusters[k])

                modularityC = 0
                for v in range(len(XH)):
                    adjMatV = X_clus[v]
                    modularityC = modularityC + modularity_byhands(adjMatV, self.n_clusters[k],
                                                                   consensus_clustering_labels)

                # modularityC = model.modularity

                listModularitiesClust.append(modularityC)

                if len(newLevel) > 0:
                    phiK = np.asarray(consensus_clustering_labels) + (np.amax(newLevel) + 1)
                else:
                    phiK = np.asarray(consensus_clustering_labels)

                newLevel = newLevel + phiK.tolist()
            
            print('k ', k)
            print( self.n_clusters[k])
            
            numberCluster = 1 
            for ck in range((k+1)) : 
               numberCluster = numberCluster * self.n_clusters[ck]

            modularityC = 0
            for v in range(len(X)):
                adjMatV = X[v]
                modularityC = modularityC + modularity_byhands(adjMatV,numberCluster,
                                                               newLevel)

            listModularitiesLevel.append(modularityC)

            # listModularitiesLevel.append(np.sum(listModularitiesClust) / len(listModularitiesClust))
            modularitiesIterLevel.append(modularitiesIterClust)
            hierarchical_label.append(newLevel)
            actuelCLustering = np.asarray(newLevel)

        modularitiesIter.append(modularitiesIterLevel)

        self.row_labels_ = newLevel
        self.hierarchical_labels_ = hierarchical_label
        self.modularity = np.sum(listModularitiesLevel) / len(listModularitiesLevel)
        self.modularitiesLevel = listModularitiesLevel
        self.modularitiesIter = modularitiesIter
