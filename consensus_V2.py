import networkx as nx
import numpy as np
import community
from community import community_louvain
import Function_PLBvem_and_SPLBvem as FP
import Cluster_Ensembles.Cluster_Ensembles as CE
import pandas as pd
import scipy
from scipy import sparse
from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import time
import random
from os.path import dirname, exists, expanduser, isdir, join, splitext
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from coclust.coclustering import CoclustMod, CoclustInfo
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import ward, fcluster, fclusterdata

def  compute_eval_metrics_level(trueLabels, PredictedLabels, level = 0):
  tl = np.asarray(trueLabels[level]).tolist()
  pl = np.asarray(PredictedLabels[level]).tolist()
  acc = np.around(accuracy(tl, pl), 3)
  nmi = np.around(normalized_mutual_info_score(tl, pl), 3)
  ari = np.around(adjusted_rand_score(tl, pl), 3)
  purity = np.around(purity_score(tl, pl), 3)

  return acc,nmi,ari,purity



def compute_consensus(data,nom_couches, clusters_level, algorithm_instance, Z_init, h_label,bddName, it ):
    path_to_save_slice = '/H-TGM/Comparison_Simple_Algo_Slice/'
    hierarchical_labels_ = []
    v = len(data)
    X= data
    """
    X=[]
    for b in range(v):
      X.append(data[b].toarray())
    """
    if algorithm_instance == 'Consensus-AHC' :
      for l, lev in enumerate(clusters_level):
        nclusters = 1
        if l ==0:
          nclusters = clusters_level[0]
        else:
          for j in range(l):
            nclusters= nclusters* clusters_level[j]
        
        cluster_runs = np.zeros((v, (X[0]).shape[0]))
        print(cluster_runs)
        for b in range(v):
          print(b)
          HCclustering = AgglomerativeClustering(n_clusters=nclusters, affinity='euclidean', linkage='single').fit(X[b])
          newLevel   = HCclustering.fit_predict(X[b])
          phiK = np.asarray(newLevel)
          np.savez_compressed(path_to_save_slice + "/" + 'simple_label_' + bddName + '_'+  nom_couches[b] + '_' + algorithm_instance + '_' + str(l)  + '_' + str(it), a=phiK)
          cluster_runs[b, :] = np.asarray(phiK)
        #print("newLevel", np.unique(newLevel))
        consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True, nclass=nclusters)
        hierarchical_labels_.append(consensus_clustering_labels)

    elif algorithm_instance == 'Consensus-NMF' :
      for l, lev in enumerate(clusters_level):
        nclusters = 1
        if l ==0:
          nclusters = clusters_level[0]
        else:
          for j in range(l):
            nclusters= nclusters* clusters_level[j]

        cluster_runs = np.zeros((v, (X[0]).shape[0]))
        for b in range(v):
          model = NMF(n_components=nclusters, init='random', random_state=0)
          W = model.fit_transform(X[b])
          best_clustering_labels = np.argmax(W,axis = 1)
          phiK = np.asarray(best_clustering_labels)
          np.savez_compressed(path_to_save_slice + "/" + 'simple_label_' + bddName + '_'+  nom_couches[b] + '_' + algorithm_instance + '_' + str(l)  + '_' + str(it), a=phiK)

          cluster_runs[b, :] = np.asarray(phiK)
         
        consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True, nclass=nclusters)
        hierarchical_labels_.append(consensus_clustering_labels)

    elif algorithm_instance == 'Consensus-SPLBM' :
      for l, lev in enumerate(clusters_level):
        nclusters = 1
        if l ==0:
          nclusters = clusters_level[0]
        else:
          for j in range(l):
            nclusters= nclusters* clusters_level[j]

        cluster_runs = np.zeros((v, (X[0]).shape[0]))
        for b in range(v):
          [pi_k_hat, rho_l_hat, mukl_hat, part, part2, news, acc_ex, nmi_ex, ari_ex] = FP.SPLBcem(X[b], Z_init, Z_init,h_label[l],nclusters)
          best_clustering_labels = np.asarray(part)
          phiK = np.asarray(best_clustering_labels)
          np.savez_compressed(path_to_save_slice + "/" + 'simple_label_' + bddName + '_'+  nom_couches[b] + '_' + algorithm_instance + '_' + str(l)  + '_' + str(it), a=phiK)
          cluster_runs[b, :] = np.asarray(phiK)

        consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True, nclass=nclusters)
        hierarchical_labels_.append(consensus_clustering_labels)

    elif algorithm_instance == 'Consensus-Coclust' :
      for l, lev in enumerate(clusters_level):
        nclusters = 1
        if l ==0:
          nclusters = clusters_level[0]
        else:
          for j in range(l):
            nclusters= nclusters* clusters_level[j]

        cluster_runs = np.zeros((v, (X[0]).shape[0]))
        for b in range(v):
          model = CoclustMod(n_clusters= nclusters,init=Z_init,max_iter=100)
          model.fit(X[b])
          best_clustering_labels = model.row_labels_
          phiK = np.asarray(best_clustering_labels)
          np.savez_compressed(path_to_save_slice + "/" + 'simple_label_' + bddName + '_'+  nom_couches[b] + '_' + algorithm_instance + '_' + str(l)  + '_' + str(it), a=phiK)
          cluster_runs[b, :] = np.asarray(phiK)

        consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True, nclass=nclusters)
        hierarchical_labels_.append(consensus_clustering_labels)


    elif algorithm_instance == 'Consensus-CoclustInfo' :
      for l, lev in enumerate(clusters_level):
        nclusters = 1
        if l ==0:
          nclusters = clusters_level[0]
        else:
          for j in range(l):
            nclusters= nclusters* clusters_level[j]

        cluster_runs = np.zeros((v, (X[0]).shape[0]))
        for b in range(v):
          model = CoclustInfo(n_row_clusters=nclusters, n_col_clusters=nclusters, init=Z_init,max_iter=100)
          model.fit(X[b])
          best_clustering_labels = model.row_labels_
          phiK = np.asarray(best_clustering_labels)
          np.savez_compressed(path_to_save_slice + "/" + 'simple_label_' + bddName + '_'+  nom_couches[b] + '_' + algorithm_instance + '_' + str(l)  + '_' + str(it), a=phiK)
          cluster_runs[b, :] = np.asarray(phiK)

        consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True, nclass=nclusters)
        hierarchical_labels_.append(consensus_clustering_labels)

    elif algorithm_instance == 'Consensus-Louvain':
      #partition = community_louvain.best_partition(G)

      for l, lev in enumerate(clusters_level):
        nclusters = 1
        if l ==0:
          nclusters = clusters_level[0]
        else:
          for j in range(l):
            nclusters= nclusters* clusters_level[j]

        cluster_runs = np.zeros((v, (X[0]).shape[0]))
        for b in range(v):
          G = nx.from_numpy_array(X[b])
          dendrogram = community_louvain.generate_dendrogram(G)
          print(len(dendrogram))
          try:
            dict_cluster =  community_louvain.partition_at_level(dendrogram, l)
          except:
            print('level non trouvé')
          else:
            dict_cluster =  community_louvain.partition_at_level(dendrogram, l-1)
          partition_level = list(dict_cluster.values())
          phiK = np.asarray(partition_level)
          np.savez_compressed(path_to_save_slice + "/" + 'simple_label_' + bddName + '_'+  nom_couches[b] + '_' + algorithm_instance + '_' + str(l)  + '_' + str(it), a=phiK)
          cluster_runs[b, :] = np.asarray(phiK)

        consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=True, nclass=nclusters)
        hierarchical_labels_.append(consensus_clustering_labels)

    else:
      print("the algorithm is unknown")

    return hierarchical_labels_

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



bdd_name =["Github_BIO_IA", "NYT_33_2K", "Yelp_5K","Amazon", "DBPEDIA"]

path_global = '/home/boutalbk/conda_environnements/H-TGM/data/HTGM-Data/'
path_dataframe = '/home/boutalbk/conda_environnements/H-TGM/data/HTGM-Data/Data_processed/'
path_to_save = '/home/boutalbk/conda_environnements/H-TGM/data/HTGM-Data/Comparison_consensus_Algo'
nom_dossier_matriceSim = "Sim_sparse"

nom_couches = ['BOW', 'XLNET', 'SKIPGRAM','entityEmbeding', "sentence_embeddings"]

listKAll = [[2,2,2],[3,3],[2,2],[3,2,2],[3,2,2]]
#listKAll = [[3,2,2]]

for b, bddName in enumerate(bdd_name):
    print("######################################### "+ bddName + " #############################################")
    listK  = listKAll[b]

    K = listK[0]

    data = []
    data_ = []
    for n,nc in enumerate(nom_couches):

      chemin=path_global+'Data_processed_binaires'+'/'+nom_dossier_matriceSim+'_'+nc+'_'+bddName+'.npz'
      sparse_matrix = scipy.sparse.load_npz(chemin)
      data_.append(np.asarray(sparse_matrix))
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
    #              Execute TSPLBM on the dataset                     #
    ##################################################################
    n_new = data[0].shape[0]
    d_new = data[0].shape[1]
    v = len(data)


    ##################################################################
    ########################## Version Hard ##########################
    ##################################################################
    nbrIteration = 30

    # 'Consensus-AHC', ['Consensus-NMF','Consensus-Louvain','Consensus-Coclust', 'Consensus-CoclustInfo'
    listAlgorithmName =['Consensus-AHC']

    df_results = pd.DataFrame(columns=["Dataset", "Slice", "Algorithm", "Time", "ACC0", "NMI0", "ARI0", "Purity0", "ACC1", "NMI1", "ARI1",
                 "Purity1", "ACC2", "NMI2", "ARI2", "Purity2"],index = np.arange(nbrIteration * len(listAlgorithmName)).tolist())

    cpt = 0
    for a, algorithm in enumerate(listAlgorithmName):
      print("algorithm ", algorithm)
      for it in range(nbrIteration):
          random.seed(it)
          np.random.seed(it)
          print("iter " + str(it))
          Z_init = random_init(K, n_new)

          start_time = time.time()
          PredictedLabels =  compute_consensus(data_, nom_couches, listK, algorithm, Z_init, h_label, bddName, it)

          end_time = time.time()

          np.savez_compressed(path_to_save + "/" + 'level_label_' + bddName + '_' + listAlgorithmName[a] + '_' + str( listK) + '_' + str(it), a=PredictedLabels)
          phiK = np.asarray(PredictedLabels[0])


          algorithm_name = listAlgorithmName[a]
          time_ = end_time - start_time

          print(PredictedLabels)
          acc_l0, nmi_l0, ari_l0, purity_l0 = compute_eval_metrics_level(h_label, PredictedLabels, level=0)
          acc_l1, nmi_l1, ari_l1, purity_l1 = compute_eval_metrics_level(h_label, PredictedLabels, level=1)
          if len(h_label)==3:
            acc_l2, nmi_l2, ari_l2, purity_l2 = compute_eval_metrics_level(h_label, PredictedLabels, level = 2)



          df_results.Dataset[cpt] = bdd_name
          df_results.Slice[cpt] = "Consensus"
          df_results.Algorithm[cpt] = algorithm_name
          df_results.Time[cpt] = str(time_)
          df_results.ACC0[cpt] = str(acc_l0)
          df_results.NMI0[cpt] = str(nmi_l0)
          df_results.ARI0[cpt] = str(ari_l0)
          df_results.Purity0[cpt] = str(purity_l0)
          df_results.ACC1[cpt] = str(acc_l1)
          df_results.NMI1[cpt] = str(nmi_l1)
          df_results.ARI1[cpt] = str(ari_l1)
          df_results.Purity1[cpt] = str(purity_l1)
          if len(h_label)==3:
            df_results.ACC2[cpt] = str(acc_l2)
            df_results.NMI2[cpt] = str(nmi_l2)
            df_results.ARI2[cpt] = str(ari_l2)
            df_results.Purity2[cpt] = str(purity_l2)
          else:
            df_results.ACC2[cpt] = str(0)
            df_results.NMI2[cpt] = str(0)
            df_results.ARI2[cpt] = str(0)
            df_results.Purity2[cpt] = str(0)
          cpt = cpt + 1


    df_results.to_csv(path_to_save+ 'results_'+bddName+'_ConsensusAlgorithms'+'.csv' , index=False)    