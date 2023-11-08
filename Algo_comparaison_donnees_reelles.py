import numpy as np
import tensorly as tl

def flatten(X, mode=0):

        l = np.arange(len(X.shape))
        l = np.concatenate(([mode], np.delete(l,mode)))
        X_flat = X.transpose(*l).reshape((X.shape[mode], -1))

        return X_flat

def calculate_loss(X, Factors, mode=0):

        X_flat = flatten(X, mode)
        H = tl.tenalg.khatri_rao([Factors[i] for i in range(len(Factors)) if i != mode]).T
        return np.linalg.norm(X_flat - Factors[mode] @ H)

def op(X, A, S, w, eps=1e-10):
    return np.multiply(w, np.divide(A.T @ X @ S.T, eps + A.T @ A @ w @ S @ S.T))

def update(X, Factors, Ws, mode=0):

    X_flat = flatten(X, mode)
    H = tl.tenalg.khatri_rao([Factors[i] @ Ws[i] for i in range(len(Factors)) if i != mode]).T
    return op(X_flat, Factors[mode], H, Ws[mode])

def run(X, Factors, r, N=100):

    Ws = [np.random.rand(Factors[0].shape[1],r) for _ in range(len(Factors))]

    losses = []
    for K in range(N):
        for i in range(len(Factors)):
            Ws[i] = update(X, Factors, Ws, mode=i)
        losses.append(calculate_loss(X, [Factors[i] @ Ws[i] for i in range(len(Factors))]))

    return Ws


def update_single(X, Factors, W, mode=0):

    X_flat = flatten(X, mode)
    H = tl.tenalg.khatri_rao([Factors[i] @ W for i in range(len(Factors)) if i != mode]).T
    return op(X_flat, Factors[mode], H, W)

def run_single(X, Factors, r, N=100):

    W = np.random.rand(Factors[0].shape[1],r)

    losses = []
    for K in range(N):
        Ws = []
        for i in range(len(Factors)):
            Ws.append(update_single(X, Factors, W, mode=i))

        Ws.append(W)
        Ws.append(W)
        Ws.append(W)
        W = np.stack(Ws).mean(axis=0)

        losses.append(calculate_loss(X, [Factors[i] @ W for i in range(len(Factors))]))

    return W



def MultiHNTF(X, r, N = 800, k = 3 ):
  # unsupervised case,one layer

  np.random.seed(0)

  error , Factors  = non_negative_parafac(X, rank=r[0])

  list_Factors_level = []
  oldFactor = Factors
  for j in range(1,len(r)):
    W = run_single(X, oldFactor, r[j], N)
    Factors_2 = [oldFactor[i] @ W for i in range(k)]
    list_Factors_level.append(Factors_2[(len(r)-1)])
    oldFactor = Factors_2

  cluster_Factors = np.argmax(Factors[(len(r)-1)], axis = 1)
  list_Clusters_level =[]
  for a in range(len(list_Factors_level)):
    list_Clusters_level.append(np.argmax(list_Factors_level[a],axis = 1))
  list_clusters = [cluster_Factors]+ list_Clusters_level
  return list_clusters

def  compute_eval_metrics_level(trueLabels, PredictedLabels, level = 0):
  tl = np.asarray(trueLabels[level]).tolist()
  pl = np.asarray(PredictedLabels[level]).tolist()
  acc = np.around(accuracy(tl, pl), 3)
  nmi = np.around(normalized_mutual_info_score(tl, pl), 3)
  ari = np.around(adjusted_rand_score(tl, pl), 3)
  purity = np.around(purity_score(tl, pl), 3)

  return acc,nmi,ari,purity

def hierarchize_tensor_algorithm(X, clusters_level, algorithm_instance, Z_init, h_label):

  data = np.zeros((X[0].shape[0],X[0].shape[0],len(X)))
  for v_ in range(len(X)):
        data[:,:,v_]= X[v_].toarray()

  random_state = 12345
  hierarchical_labels_ = []
  if  algorithm_instance == 'H-PARAFAC' :
    error,res_Parafac = parafac(tensor=data, rank=5, init='random', tol=10e-6, random_state=random_state)
    res_Parafac_kmeans = KMeans(n_clusters=clusters_level[0], random_state=0, n_init=1).fit(res_Parafac[0])
    best_clustering_labels = res_Parafac_kmeans.labels_
    hierarchical_labels_.append(best_clustering_labels)
    actuelCLustering = best_clustering_labels
    for k in range(1, len(clusters_level)):
            newLevel = []
            for clus in np.unique(actuelCLustering):
                    X_clus_v = data[actuelCLustering == clus, :,:]
                    X_clus_v = X_clus_v[:, actuelCLustering == clus,:]
                    error,res_Parafac = parafac(tensor=X_clus_v, rank=5, init='random', tol=10e-6, random_state=random_state)
                    res_Parafac_kmeans = KMeans(n_clusters=clusters_level[k], random_state=0, n_init=1).fit(res_Parafac[0])
                    best_clustering_labels_h = res_Parafac_kmeans.labels_
                    if len(newLevel) > 0:
                      phiK = np.asarray(best_clustering_labels_h) + (np.amax(newLevel) + 1)
                    else:
                      phiK = np.asarray(best_clustering_labels_h)

                    newLevel = newLevel + phiK.tolist()

            hierarchical_labels_.append(newLevel)
            actuelCLustering = np.asarray(newLevel)

  elif algorithm_instance == 'H-TUCKER' :
    # a completer
    tucker_rank = [5, 5, 2]
    core, tucker_factors = tucker(data, rank=tucker_rank, init='random', tol=10e-5, random_state=random_state)
    res_Tucker_kmeans = KMeans(n_clusters=clusters_level[0], random_state=0, n_init=1).fit(tucker_factors[0])
    best_clustering_labels = res_Tucker_kmeans.labels_
    hierarchical_labels_.append(best_clustering_labels)
    actuelCLustering = best_clustering_labels
    for k in range(1, len(clusters_level)):
            newLevel = []
            for clus in np.unique(actuelCLustering):
                    X_clus_v = data[actuelCLustering == clus, :,:]
                    X_clus_v = X_clus_v[:, actuelCLustering == clus,:]
                    core, tucker_factors = tucker(X_clus_v, rank=tucker_rank, init='random', tol=10e-5, random_state=random_state)
                    res_Tucker_kmeans = KMeans(n_clusters=clusters_level[k], random_state=0, n_init=1).fit(tucker_factors[0])
                    best_clustering_labels_h = res_Tucker_kmeans.labels_
                    if len(newLevel) > 0:
                      phiK = np.asarray(best_clustering_labels_h) + (np.amax(newLevel) + 1)
                    else:
                      phiK = np.asarray(best_clustering_labels_h)

                    newLevel = newLevel + phiK.tolist()

            hierarchical_labels_.append(newLevel)
            actuelCLustering = np.asarray(newLevel)

  elif algorithm_instance == 'H-NTF' :

    hierarchical_labels_ = MultiHNTF(data, clusters_level,N = 800, k = 3 )

  elif algorithm_instance == 'H-TSPLBM' :
    model = sparseTensorCoclustering.SparseTensorCoclusteringPoisson(n_clusters=clusters_level[0],  fuzzy = False, init_row=Z_init, init_col=Z_init, max_iter=50)
    model.fit(data)
    best_clustering_labels = model.row_labels_
    hierarchical_labels_.append(best_clustering_labels)
    actuelCLustering = best_clustering_labels
    for k in range(1, len(clusters_level)):
            newLevel = []
            for clus in np.unique(actuelCLustering):
                    X_clus_v = data[actuelCLustering == clus, :,:]
                    X_clus_v = X_clus_v[:, actuelCLustering == clus,:]
                    print("X_clus_v", X_clus_v.shape)
                    Z_init_h = random_init(clusters_level[k], X_clus_v.shape[0])
                    model_h =  sparseTensorCoclustering.SparseTensorCoclusteringPoisson(n_clusters=clusters_level[k],  fuzzy = False, init_row=Z_init_h, init_col=Z_init_h, max_iter=50)
                    model_h.fit(X_clus_v)
                    best_clustering_labels_h = model_h.row_labels_
                    if len(newLevel) > 0:
                      phiK = np.asarray(best_clustering_labels_h) + (np.amax(newLevel) + 1)
                    else:
                      phiK = np.asarray(best_clustering_labels_h)

                    newLevel = newLevel + phiK.tolist()

            hierarchical_labels_.append(newLevel)
            actuelCLustering = np.asarray(newLevel)

  else:
    print("the algorithm is unknown")

  return hierarchical_labels_

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
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
import sparseTensorCoclustering
from tensorly.decomposition import parafac, non_negative_parafac, tucker


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





bdd_name = ["DBPEDIA","Github_BIO_IA", "NYT_33_2K","Yelp_5K"]
path_global = '/home/boutalbk/conda_environnements/H-TGM/data/HTGM-Data/'
path_dataframe = '/home/boutalbk/conda_environnements/H-TGM/data/HTGM-Data/Data_processed/'
path_to_save = '/home/boutalbk/conda_environnements/H-TGM/data/HTGM-Data/Comparison_Tensor_Algo/'
nom_dossier_matriceSim = "Sim_sparse"

nom_couches = ['BOW', 'XLNET', 'SKIPGRAM','entityEmbeding', "sentence_embeddings"]

listKAll = [[3,2,2],[2,2,2],[3,3],[2,2]]

for b, bddName in enumerate(bdd_name):
    print("######################################### "+ bddName + " #############################################")
    listK  = listKAll[b]

    K = listK[0]

    data = []
    for n,nc in enumerate(nom_couches):

      #chemin = path_global + bddName +'/'+nom_dossier_matriceSim + '/'+ 'Sim_sparse_'+ nc + '.npz'
      chemin=path_global+'Data_processed_binaires'+'/'+nom_dossier_matriceSim+'_'+nc+'_'+bddName+'.npz'
      sparse_matrix = scipy.sparse.load_npz(chemin)
      data.append(sparse_matrix)

    #Data_Github_BIO_IA_Processed
    df_original = pd.read_csv(path_dataframe+'Data_'+bddName+'_Processed'+'.csv' )
    labelName= 'Cat'

    h_label = []
    for level in range(len(listK)):
      numericLabel = pd.factorize(df_original[labelName+str(level+1)])[0].astype(np.uint16)
      #print("numericLabel ", numericLabel)
      labels_ = np.asarray(numericLabel)
      h_label.append(labels_)


    #data=[ data_v2[:,:,0], data_v2[:,:,1], data_v2[:,:,2]]
    ##################################################################
    #              Execute TSPLBM on the dataset                     #
    ##################################################################
    n_new = data[0].shape[0]
    d_new = data[0].shape[1]
    v = len(data)

    ##################################################################
    #                    Loading SMARTask dataset                    #
    ##################################################################


    ##################################################################
    ########################## Version Hard #########################
    ##################################################################
    nbrIteration = 30


    #"H-PARAFAC",'H-TUCKER',
    listAlgorithmName =['H-TSPLBM']#

    df_results = pd.DataFrame(columns=["Dataset", "Slice", "Algorithm", "Time", "ACC0", "NMI0", "ARI0", "Purity0", "ACC1", "NMI1", "ARI1", "Purity1", "ACC2", "NMI2", "ARI2", "Purity2"],
                              index=np.arange(nbrIteration*len(listAlgorithmName)).tolist())
    cpt = 0


    for a, algorithm in enumerate(listAlgorithmName):
      print("algorithm ", algorithm)
      for it in range(10,nbrIteration):  #
          random.seed(it)
          np.random.seed(it)
          print("iter " + str(it))
          Z_init = random_init(K, n_new)


          start_time = time.time()

          PredictedLabels =  hierarchize_tensor_algorithm(data, listK, algorithm, Z_init, h_label)

          end_time = time.time()

          np.savez_compressed(path_to_save + "/"+ bddName+ '_'+ listAlgorithmName[a]+'_'+str(listK) + '_' +str(it), a=PredictedLabels)

          phiK = np.asarray(PredictedLabels[0])


          algorithm_name = listAlgorithmName[a]
          time_ = end_time - start_time

          acc_l0, nmi_l0, ari_l0, purity_l0 = compute_eval_metrics_level(h_label, PredictedLabels, level = 0)
          acc_l1, nmi_l1, ari_l1, purity_l1 = compute_eval_metrics_level(h_label, PredictedLabels, level = 1)
          if len(h_label)==3:
            acc_l2, nmi_l2, ari_l2, purity_l2 = compute_eval_metrics_level(h_label, PredictedLabels, level = 2)

          print("Accuracy : ", acc_l0)
          print("nmi : ", nmi_l0)
          print("ari : ", ari_l0)
          print("purity : ", purity_l0)

          df_results.Dataset[cpt] = bdd_name
          df_results.Slice[cpt] = "All"
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


    df_results.to_csv(path_to_save+ 'results_'+bddName+'_tensorAlgorithms'+'.csv' , index=False)    


