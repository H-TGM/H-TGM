# H-TGM 

<p align="justify">
h-TGM (Hierarchical Tensor-based Graph Modularity) is a novel multi-graph clustering approach
that leverages modularity-maximization over a multi-graph. By Combining the different graphs HTGM is fusioning connectivity information from different graphs to provide a more
robust and higher modularity clustering.
It is based on tensor representation and graph modularity and can extract the hierarchical structure of clusters. 
aiming to cluster tensor data.
It is a general, practical, and parameter-free method to learn a consensus clustering from multiple graphs via tensor-based graph modularity. The advantage of the proposed approach is taking 
into account the graphs' properties (degree of nodes, edges density, community structure, etc.)  through the modularity measure. 
We propose to apply the proposed approach to text clustering.

A suitable way To represent multi-graphs, is by combining the different available representations using a 3-way similarity tensor.
To achieve that for a given dataset, we first compute different representation matrices (using BOW,  static embeddings such Skipgram, contextual embeddings such BERT, XLNET and entity-embedding). Then, we compute from each data representation a similarity matrix viewed as the adjacency matrix of a graph that connects the documents (nodes). 
Finally, the similarity matrices are structured as a 3-way tensor (cf. figure below)

It also allows to perform hierarchical tensor clustering through decomposition or tensor learning and tensor algebra. 
HTGMclustering allows easy interaction with other python packages such as NumPy, Tensorly, TensorFlow, or TensorD, and run methods at scale on CPU or GPU.

**It supports major operating systems namely Microsoft Windows, macOS, and Ubuntu**.
</p>

[![N|Solid](https://github.com/TGMclustering/TGMclustering/blob/main/diagram-20231027.pdf?raw=true)]()

- Source-code: https://github.com/H-TGM

- GPU version is available at https://github.com/H-TGM

### Requirements

All needed packages are available in the [requirements file](https://github.com/TGMclustering/TGMclustering/blob/main/requirements.txt)


### Using TGM

To clone TGMclustering project from github
```
# Install git LFS via https://www.atlassian.com/git/tutorials/git-lfs
# initialize Git LFS
git lfs install Git LFS initialized.
git init Initialized
# clone the repository
git clone https://github.com/TGMclustering/TGMclustering.git
cd TGMclustering
```

### License
TGM is released under the MIT License (refer to LISENSE file for details).

### Examples

```python
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import TensorClustModSparse
import matplotlib.pyplot as plt
from scipy import sparse
import hierarchicalTGM
from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import time
import random
from os.path import dirname, exists, expanduser, isdir, join, splitext
from scipy import sparse
import scipy

def random_init(n_clusters, n_cols, random_state=None):
    """Create a random column cluster assignment matrix.
    Each row contains 1 in the column corresponding to the cluster where the
    processed data matrix column belongs, 0 elsewhere.
    Parameters
    ----------
    n_clusters: int
        Number of clusters
    n_cols: int
        Number of columns of the data matrix (i.e. number of rows of the
        matrix returned by this function)
    random_state : int or :class:`numpy.RandomState`, optional
        The generator used to initialize the cluster labels. Defaults to the
        global numpy random number generator.
    Returns
    -------
    matrix
        Matrix of shape (``n_cols``, ``n_clusters``)
    """

    if random_state == None:
        W_a = np.random.randint(n_clusters, size=n_cols)

    else:
        random_state = check_random_state(random_state)
        W_a = random_state.randint(n_clusters, size=n_cols)

    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

bdd_name = ["Amazon", "Dpedia12K", "Amazon_17K"]
path_global = './H-TGM/data/HTGM-Data/'
path_dataframe = './H-TGM/data/HTGM-Data/Data_processed/'
path_data_original = './H-TGM/data/HTGM-Data/Data_processed/'
path_to_save       = './H-TGM/data/HTGM-Data/Resultats/'
nom_dossier_matriceSim = "Sim_sparse"
#Data_processed_binaires/Sim_sparse_XLNET_Github_BIO_IA.npz
nom_couches = ['BOW', 'XLNET', 'SKIPGRAM','entityEmbeding', "sentence_embeddings"]

listKAll = [[3,2,2],[2,2,2],[6,3,3]]

for b, bddName in enumerate(bdd_name):
    print("######################################### "+ bddName + " #############################################")
    listK  = listKAll[b]

    K = listK[0]

    data = []
    for n,nc in enumerate(nom_couches):
       
      #chemin = path_global + bddName +'/'+nom_dossier_matriceSim + '/'+ 'Sim_sparse_'+ nc + '.npz'
      chemin=path_global+'/Data_processed_binaires'+'/'+nom_dossier_matriceSim+'_'+nc+'_'+bddName+'.npz'
      sparse_matrix = scipy.sparse.load_npz(chemin)
      data.append(sparse_matrix)

    #Data_Github_BIO_IA_Processed
    df_original = pd.read_csv(path_dataframe+'Data_'+bddName+'_Processed'+'.csv' )
    labelName= 'Cat'


    h_label = []
    for level in range(len(listK)):
      numericLabel = pd.factorize(df_original[labelName+str(level+1)])[0].astype(np.uint16)
      labels_ = np.asarray(numericLabel)
      h_label.append(labels_)

 
    ##################################################################

    n_new = data[0].shape[0]
    d_new = data[0].shape[1]
    v = len(data)

    ##################################################################
    #                            hyperparameters                     #
    ##################################################################
    #

    nbrIteration = 30
    df_results = pd.DataFrame(columns=["Dataset", "Time", "Clusters", "ACC", "NMI", "ARI", "Purity", "Modularity"],
                              index=np.arange(nbrIteration).tolist())

    print(df_results.shape)




    ##################################################################
    ########################## Version Hard #########################
    ##################################################################


    cpt = 0
    modularities = []

    for it in range(nbrIteration):
        print("iter " + str(it))
        labels=h_label[0]

        random.seed(it)
        np.random.seed(it)

        Z_init = random_init(K, n_new)

        start_time = time.time()
        #model = TensorClustModSparse.TensorCoclustMod(n_clusters=K, init=Z_init, bool_fuzzy=False,
        #                                              fusion_method='Add')

        model = hierarchicalTGM.HTGM(n_clusters=listK, init=Z_init, n_init_level1=10)
        
        model.fit(data)
        end_time = time.time()

        mod = model.modularity
        print('mod ', mod)
        modularities.append(mod)

        print('modularitiesLevel', model.modularitiesLevel)
        modularitiesLevel = model.modularitiesLevel

        phiK= model.hierarchical_labels_
        phiK_= phiK.copy()
        phiK__ = np.asarray(phiK_[0])
        

        level_label = model.hierarchical_labels_
        level_label = np.asarray(level_label)

        np.savez_compressed(path_to_save + "/"+ 'level_label_'+bddName+ '_' +str(it), a=level_label)

        time_ = end_time - start_time
        acc = np.around(accuracy(labels, phiK__.tolist()), 3)
        nmi = np.around(normalized_mutual_info_score(labels, phiK__.tolist()), 3)
        ari = np.around(adjusted_rand_score(labels, phiK__.tolist()), 3)
        purity = np.around(purity_score(labels, phiK__.tolist()), 3)

        print("Accuracy : ", acc)
        print("nmi : ", nmi)
        print("ari : ", ari)
        print("purity : ", purity)

        df_results.Dataset[cpt] = bdd_name
        df_results.Time[cpt] = str(time_)
        df_results.Clusters[cpt] = str(listK)
        df_results.ACC[cpt] = str(acc)
        df_results.NMI[cpt] = str(nmi)
        df_results.ARI[cpt] = str(ari)
        df_results.Purity[cpt] = str(purity)
        df_results.Modularity[cpt] = str(modularitiesLevel)
        cpt = cpt + 1

    df_results.to_csv(path_to_save + "Results_HTGM"+bddName+".csv", index=False)




```
### To reproduce the paper resuls 
```
Step 1: Clone the GitHub repository of H-TGMclustering.
Step 2: Install requirements packages.
Step 3: Run TGM_experiments.py to generate the results of TGM for all configurations.
Step 4: Run ConfigurationAnalysis.py to analyze and compare the configurations.
Step 5: Run ViewDocumentClustering.py, SliceDocumentClustering.py,
TensorDecompDocumentClustering.py, and TSPLBM_DocumentClustering.py to generate the results
of the classical, graph, tensor decomposition, and TSPLBM clustering approaches, respectively. 
Step 6: Run TGMLatexTable.py to generate the latex table with all results.
Step 7: Run objectiveFunctionAnalysis to analyze the evolution of the TGM objective
function according to NMI.
Step 8: Run tgm_ensemble_clustering to compare the results of TGM with and without consensus.
Step 9: Run ClusterNumberSelection.py and ClusterNumberSelectionDraw.py to generate
the results of the cluster number selection using the TGM objective function. 
Step 10: Run TimeAnalysis_TGM_vs_TSPLBM to compare TGM and TSPLBM in terms of time execution.
```