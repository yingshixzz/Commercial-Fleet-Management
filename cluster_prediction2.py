import CTMC3 as ctmc    #ctmc: continuous-time markov chain
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import copy as copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from random import sample
from numpy.random import uniform
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import kelbow_visualizer
from sklearn.metrics import silhouette_score
import matplotlib.colors
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from scipy import optimize

def n_make_region_plot(dataset):
    # makes = dataset['VehicleMake'].unique()
    makes = dataset['VehicleMake'].value_counts().index
    regions = dataset['Region'].unique()
    regions = np.sort(regions)
    n_makes = len(makes)
    n_regions = len(regions)
    matrix = np.zeros((n_regions, n_makes))
    for i, region in enumerate(regions):
        for j, make in enumerate(makes):
            print('VehicleMake: ', make, ', Region: ', region)
            subdataset = dataset[(dataset.loc[:, 'VehicleMake'] == make) & (dataset.loc[:, 'Region'] == region)]
            print('number of dataset: ', subdataset.shape[0])
            matrix[i, j] = subdataset.shape[0]
    ax = sns.heatmap(matrix, xticklabels=makes, yticklabels=regions, cmap="YlGnBu", annot=True, fmt=".0f")
    plt.title('The Number of Vehicles with Different Makes and Regions', fontsize=10)  # title with fontsize 10
    plt.xlabel('Vehicle Make', fontsize=10)  # x-axis label with fontsize 10
    plt.ylabel('Region', fontsize=10)  # y-axis label with fontsize 10
    plt.show()

def dataset_generation(dataset, feature_names):
    # dataset generation
    dataset_remain = pd.DataFrame(columns=dataset.columns)
    makes = dataset['VehicleMake'].unique()
    regions = dataset['Region'].unique()
    train_dataset = copy.deepcopy(dataset)
    dicts_testdata = []
    for make in makes:
        for region in regions:
            print('VehicleMake: ', make, ', Region: ', region)
            subdataset = dataset[(dataset.loc[:, 'VehicleMake'] == make) & (dataset.loc[:, 'Region'] == region)]
            print('number of dataset: ', subdataset.shape[0])
            if subdataset.shape[0] >= 5:
                n_test = int(subdataset.shape[0]*0.2)
            elif (subdataset.shape[0] < 5) & (subdataset.shape[0] >= 3):
                n_test = 1
            else:
                n_test = 0
                continue
            test_data = subdataset[:n_test]
            # test_data = test_data.loc[:, feature_names]
            train_dataset = train_dataset.drop(test_data.index)
            dicts_testdata.append({'make': make, 'region': region, 'test_dataset': test_data})
    return train_dataset, dicts_testdata

# function to compute hopkins's statistic for the dataframe X
def hopkins_statistic(X):
    X = X.values  # convert dataframe to a numpy array
    sample_size = int(X.shape[0] * 0.05)  # 0.05 (5%) based on paper by Lawson and Jures

    # a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0), (sample_size, X.shape[1]))

    # a random sample of size sample_size from the original data X
    random_indices = sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]

    # initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)

    # u_distances = nearest neighbour distances from uniform random sample
    u_distances, u_indices = nbrs.kneighbors(X_uniform_random_sample, n_neighbors=2)
    u_distances = u_distances[:, 0]  # distance to the first (nearest) neighbour

    # w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances, w_indices = nbrs.kneighbors(X_sample, n_neighbors=2)
    # distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[:, 1]

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)

    # compute and return hopkins' statistic
    H = u_sum / (u_sum + w_sum)
    return H

class determine_k:
    def __init__(self, dataset):
        self.dataset = dataset

    def elbow(self):
        kelbow_visualizer(KMeans(random_state=1), self.dataset, k=(2, 10))

    def silhouette(self):
        # sil = []
        # kmax = 10
        # # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        # for k in range(2, kmax + 1):
        #     kmeans = KMeans(n_clusters=k).fit(self.dataset)
        #     labels = kmeans.labels_
        #     sil.append(silhouette_score(self.dataset, labels, metric='euclidean'))
        # plt.plot(range(2, kmax + 1), sil)
        # plt.show()
        # Instantiate a scikit-learn K-Means model
        model = KMeans(random_state=4)
        # Instantiate the KElbowVisualizer with the number of clusters and the metric
        visualizer = kelbow_visualizer(model, self.dataset, k=(2, 10), metric='silhouette', timings=False)

def plot_clusters(train_data, train_kmeans):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    customcmap = matplotlib.colors.ListedColormap(["crimson", "mediumblue", "darkmagenta", "green"])
    plt.scatter(train_data.loc[:, 'OldState'], train_data.loc[:, 'MileageBetweenInspections'], train_data.loc[:, 'CurrentState'],
                c=train_kmeans.labels_.astype(float),
                edgecolor="k", cmap=customcmap)
    ax.set_xlabel('OldState', fontsize=12)
    ax.set_ylabel('MileageBetweenInspections', fontsize=12)
    ax.set_zlabel('CurrentState', fontsize=12)
    ax.set_title("K-Means Clusters for the Train Dataset", fontsize=12)
    plt.show()

def Pijt_dt(dataset,weights, label_min, label_max, mileage_max, dt=100):
    # dt = 100    # the accuracy increases when decrease dt
    pijt = np.zeros((label_max+1, label_max+1, mileage_max+1))
    T = range(int(mileage_max+1))
    for i in range(label_min,label_max+1):
        index1= np.where(dataset[:,4]==i)
        dataset_i = dataset[index1]
        weights1 = weights[index1[0]]
        for l in range(int(dt/2), int(mileage_max+1-dt/2)):
            index2= np.where((dataset_i[:,0]<=(l+dt/2)) & (dataset_i[:,0]>=(l-dt/2)))
            dataset_il = dataset_i[index2]
            n_datal = np.sum(weights[index2[0]])
            weights2=weights1[index2[0]]
            if n_datal ==0:
                pijt[i, :, l] = pijt[i, :, l-1]
                continue
            for j in range(label_min,label_max+1):
                pijt[i,j,l] =np.sum((dataset_il[:,3]==j).astype(int).reshape((1,-1))*weights2.reshape((1,-1))) /n_datal
            k = l
            while np.sum(pijt[i,:,k-1])==0:
                pijt[i, :, k - 1] = pijt[i,:,k]
                k = k-1
        for l in range(int(dt/2)):
            pijt[i,:,l] = pijt[i,:,int(dt/2)]
        for l in range(int(mileage_max+1-dt/2), int(mileage_max+1)):
            pijt[i, :, l] = pijt[i, :, int(mileage_max-dt/2)]
    return pijt

def predict_state(test_i, test_mileage, pijt, label_min, label_max):
    n_test = len(test_mileage)
    y_pred = np.zeros(n_test)
    for n in range(n_test):
        pij = pijt[int(test_i[n]), :, int(test_mileage[n])]
        # y_pred[n] = np.argmax(pij)
        # y_pred[n] = np.random.choice(range(label_max + 1), 1, p=pij)[0]
        if np.sum(pij)==0:
            y_pred[n] = int(test_i[n])
        else:
            y_pred[n] = np.random.choice(range(label_max+1),1, p=pij/np.sum(pij))[0]
    return y_pred

def predict_prob(test_i, test_mileage, k, pijt_k, probs, label_min, label_max):
    n_test = len(test_mileage)
    y_pred = np.zeros(n_test)
    for n in range(n_test):
        pij_m_average = np.zeros(label_max+1)
        for m in range(k):
            pij_m = pijt_k[m][int(test_i[n]), :, int(test_mileage[n])]
            pij_m_average = pij_m_average + pij_m * probs[m]
        if np.sum(pij_m_average)==0:
            y_pred[n] = int(test_i[n])
        else:
            y_pred[n] = np.random.choice(range(label_max+1),1, p=pij_m_average/np.sum(pij_m_average))[0]
    return y_pred

def evaluate(y_pred, y_true, label_max):
    accuracy = accuracy_score(y_true, y_pred)
    cmatrix = confusion_matrix(y_true, y_pred, labels=range(0,label_max+1))
    # a= np.where(np.abs(y_true-y_pred)<=1)
    accuracy_fuzzy= np.size(np.where(np.abs(y_true-y_pred)<=1)[0])/np.size(y_pred)
    return accuracy, cmatrix, accuracy_fuzzy

def standardize(X):
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    # scaler = RobustScaler()
    # scaler = Normalizer()
    X =scaler.fit_transform(X)
    return X, scaler

def inversestandardize(X, scaler):
    X = scaler.inverse_transform(X)
    return X

# def cal_coefficient(dataset):
#     x = dataset['OldState']
#     y = dataset['MileageBetweenInspections']
#     z = dataset['DR']
#     plsq = optimize.leastsq(residuals, np.array([0, 0, 0, 0, 0, 0]), args=(z, x, y))
#     coeff1, coeff2, coeff3, coeff4, coeff5, coeff6 = plsq[0]
#     return coeff1, coeff2, coeff3, coeff4, coeff5, coeff6
# def func(x, y, p):
# 	""" 数据拟合所用的函数：z=ax+by
# 	:param x: 自变量 x
# 	:param y: 自变量 y
# 	:param p: 拟合参数 a, b
# 	"""
# 	coeff1, coeff2, coeff3, coeff4, coeff5, coeff6 = p
# 	return coeff1 * x**2 + coeff2 * x + coeff3 * y**2 + coeff4 * y + coeff5 * x* y + coeff6
def cal_coefficient(dataset):
    x = dataset['OldState']
    y = dataset['MileageBetweenInspections']
    z = dataset['DR']
    plsq = optimize.leastsq(residuals, np.array([0, 0, 0]), args=(z, x, y))
    coeff1, coeff2, coeff3 = plsq[0]
    return coeff1, coeff2, coeff3
def func(x, y, p):
	""" 数据拟合所用的函数：z=ax+by
	:param x: 自变量 x
	:param y: 自变量 y
	:param p: 拟合参数 a, b
	"""
	coeff1, coeff2, coeff3= p
	return coeff1 * x + coeff2 * y + coeff3
def residuals(p, z, x, y):
	""" 得到数据 z 和拟合函数之间的差
	"""
	return z - func(x, y, p)

def plot_clusters_DR(train_dataset, k):
    for m in range(k):
        data_m = train_dataset[(train_dataset.loc[:,'label']==m)]
        n_data_m = data_m.shape[0]
        drs = data_m.loc[:,'DR']
        hist, bins = np.histogram(drs, bins=100, range=(np.min(drs), np.max(drs)), normed=True)
        hist = hist * np.diff(bins)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, width=width)
        plt.xlabel('Deterioration Rates')
        plt.ylabel('Percentage')
        plt.title('The distribution of Deterioration Rates (cluster: '+str(m)+'# data: ' + str(n_data_m) + ')')
        plt.show()
        plt.clf()



if __name__ == "__main__":
    # component = 'FTire'
    # component = 'RTire'
    # component = 'FBrake'
    # component = 'RBrake'
    component = 'FBrake_RBrake'
    # component = 'FTire_RTire'
    # dataset = pd.read_csv('D:/PycharmProject/Mobility/dataset/' + component + 'data.csv')
    # dataset = pd.read_csv('D:/PycharmProject/Mobility/dataset/' + component + 'dataheavy.csv')    #0col: mileage between inspections; 1col: current odometer; 2col: old odometer; 3col: current state; 4col: old state
    # dataset = pd.read_csv('D:/PycharmProject/Mobility/dataset/Brakedataheavy_info.csv')
    dataset = pd.read_csv('D:/PycharmProject/Mobility/dataset/Brakedataheavy_info_filter.csv')
    dataset.columns = ['LastInspectiondate', 'Inspectiondate', 'VIN', 'VehicleYear', 'VehicleMake', 'VehicleModel',
                       'VehicleBody', 'Zip', 'CurrentOdometer', 'OldOdometer',
                       'CurrentState', 'OldState', 'MileageBetweenInspections', 'MileagePerYear', 'DR',
                       'ComponentLocation', 'VehicleAge', 'Region']

    ## check dataset (brake)
    label_min = 2
    label_max = 17
    # dr_max = 0.007
    dr_max = 0.05
    mileage_max = 20000

    dataset = dataset[dataset.loc[:, 'CurrentState'] < dataset.loc[:, 'OldState']]
    dataset = dataset[(dataset.loc[:, 'DR'] > 0) & (dataset.loc[:, 'DR'] < dr_max)]
    # dataset = dataset[(dataset.loc[:, 'DR'] > 0) & (dataset.loc[:, 'DR'] <= 0.0005)]
    # dataset = dataset[(dataset.loc[:, 'DR'] > 0.0005) & (dataset.loc[:, 'DR'] <= 0.02)]
    dataset = dataset[(dataset.loc[:, 'MileageBetweenInspections'] > 0) & (dataset.loc[:, 'MileageBetweenInspections'] <= mileage_max)]
    dataset = dataset[(dataset.loc[:, 'CurrentState'] >= label_min) & (dataset.loc[:, 'OldState'] >= label_min)]
    dataset = dataset[(dataset.loc[:, 'CurrentState'] <= label_max) & (dataset.loc[:, 'OldState'] <= label_max)]
    dataset = dataset[~dataset['DR'].isin([np.nan, np.inf, -np.inf])]
    dataset = dataset[dataset['Region'].astype(str)!='nan']
    dataset['Region'] = pd.to_numeric(dataset['Region']).astype(int)
    print('number of dataset:', np.shape(dataset)[0])

    # #dataset visualization
    # n_make_region_plot(dataset)

    # coeff1, coeff2, coeff3, coeff4, coeff5, coeff6= cal_coefficient(dataset)
    # dataset['coeff1'] = coeff1
    # dataset['coeff2'] = coeff2
    # dataset['coeff3'] = coeff3
    # dataset['coeff4'] = coeff4
    # dataset['coeff5'] = coeff5
    # dataset['coeff6'] = coeff6
    # cluster_features = ['OldState', 'MileageBetweenInspections', 'CurrentState', 'DR', 'coeff1', 'coeff2', 'coeff3', 'coeff4', 'coeff5', 'coeff6']
    # coeff1, coeff2, coeff3 = cal_coefficient(dataset)
    # dataset['coeff1'] = coeff1
    # dataset['coeff2'] = coeff2
    # dataset['coeff3'] = coeff3
    # cluster_features = ['OldState', 'MileageBetweenInspections', 'CurrentState', 'DR', 'coeff1', 'coeff2', 'coeff3']
    # cluster_features = ['DR', 'coeff1', 'coeff2', 'coeff3']
    # cluster_features = ['DR']
    # cluster_features = ['OldOdometer', 'OldState', 'MileageBetweenInspections', 'VehicleAge', 'CurrentState', 'DR']
    cluster_features = ['OldState', 'MileageBetweenInspections', 'CurrentState', 'DR']
    train_dataset, dicts_testdata = dataset_generation(dataset, cluster_features)
    train_datacluster = train_dataset.loc[:, cluster_features]

    # normalize train data
    train_datacluster = pd.DataFrame(standardize(train_datacluster)[0], columns=cluster_features)
    # test_X = scaler.fit_transform(test_X)
    #
    # ##### clustering_main
    # ## calculate Hopkins statistic
    # l = []  # list to hold values for each call
    # for i in range(100):
    #     H = hopkins_statistic(train_datacluster)
    #     l.append(H)
    # # print average value:
    # hopkins_value = np.mean(l)
    # print(hopkins_value)
    #
    # ## determine k
    # determinek = determine_k(train_datacluster)
    # determinek.elbow()
    # # determinek.silhouette()
    #
    k = 4
    model = KMeans(n_clusters=k, random_state=4)
    train_kmeans = model.fit(train_datacluster)
    labels = train_kmeans.labels_.astype(float)
    train_dataset['label'] = labels
    train_dataset.to_csv('D:/PycharmProject/Mobility/results/train_dataset_labeled_OMCDR.csv', index=True, sep=',')

    ####################
    train_dataset = pd.read_csv('D:/PycharmProject/Mobility/results/train_dataset_labeled_OMCDR.csv')
    # train_dataset = pd.read_csv('D:/PycharmProject/Mobility/results/train_dataset_labeled_DR.csv')
    plot_clusters_DR(train_dataset, k)

    # ## train the pattern classification method
    train_dataset = pd.read_csv('D:/PycharmProject/Mobility/results/train_dataset_labeled.csv')
    classify_features = ['VehicleMake', 'Region', 'OldOdometer', 'OldState', 'MileageBetweenInspections', 'VehicleAge']
    train_dataclassify_X = train_dataset.loc[:, classify_features]
    train_dataclassify_y = train_dataset.loc[:, 'label']
    categorical_cols = ['VehicleMake', 'Region']
    train_dataclassify_X = pd.get_dummies(train_dataclassify_X, columns=categorical_cols)
    train_dataclassify = pd.concat([train_dataclassify_X, train_dataclassify_y], axis=1)

    classifier = KNeighborsClassifier()
    classifier.fit(train_dataclassify_X, train_dataclassify_y)

    #######################################################################
    ### subdataset training and testing -only use samples from each cluster
    ## training
    pijt_k = []
    pred_features = ['OldState', 'MileageBetweenInspections', 'CurrentState']
    for i in range(k):
        train_data_i = train_dataset[train_dataset['label']==i]
        weights = np.ones(train_data_i.shape[0])
        pijt = Pijt_dt(
            np.array(train_data_i.loc[:,
                     ['MileageBetweenInspections', 'CurrentOdometer', 'OldOdometer', 'CurrentState', 'OldState']]),
            weights,
            label_min, label_max, mileage_max)
        pijt_k.append(pijt)
    output = open('pijt_kbrake2.pkl', 'wb')
    pickle.dump(pijt_k, output)
    output.close()

    ## testing
    pkl_file = open(
        'D:/PycharmProject/Mobility/pijt_kbrake2.pkl',
        'rb')
    pijt_k = pickle.load(pkl_file)
    pkl_file.close()
    dicts = []
    accu_sum = 0
    accu_fuzzy_sum = 0
    n_all_testdata = 0
    for index in range(len(dicts_testdata)):
        dict_testdata = dicts_testdata[index]
        test_dataset = dict_testdata['test_dataset']
        test_dataclassify_X = test_dataset.loc[:, classify_features]
        test_dataclassify_X = pd.get_dummies(test_dataclassify_X, columns=categorical_cols)
        # Get missing columns in the training test
        missing_cols = set(train_dataclassify_X.columns) - set(test_dataclassify_X.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            test_dataclassify_X[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        test_dataclassify_X = test_dataclassify_X[train_dataclassify_X.columns]
        cluster = classifier.predict(test_dataclassify_X)
        all_y_pred = []
        n_testdata = np.shape(test_dataset)[0]
        for j in range(n_testdata):
            pijt = pijt_k[int(cluster[j])]
            test_sample = test_dataset.iloc[j,:]
            a = test_sample['OldState']
            b = test_sample['MileageBetweenInspections']
            y_pred = predict_state([test_sample['OldState']], [test_sample['MileageBetweenInspections']], pijt, label_min, label_max)
            all_y_pred.append(int(y_pred[0]))
        accuracy, cmatrix, accuracy_fuzzy = evaluate(all_y_pred, test_dataset['CurrentState'].astype(int), label_max)
        # print('accuracy', accuracy)
        # print('accuracy_fuzzy', accuracy_fuzzy)
        dicts.append({'accuracy': accuracy, 'accuracy_fuzzy': accuracy_fuzzy})
        accu_sum = accu_sum + accuracy * n_testdata
        accu_fuzzy_sum = accu_fuzzy_sum + accuracy_fuzzy * n_testdata
        n_all_testdata = n_all_testdata + n_testdata
    print('Accuracy using specific model')
    print(accu_sum / n_all_testdata, accu_fuzzy_sum / n_all_testdata)


    ####### subdataset training and testing -use samples from the whole dataset to calibrate
    ## training
    pijt_k = []
    # pred_features = ['OldState', 'MileageBetweenInspections', 'CurrentState']
    for m in range(k):
        weights = np.ones(train_dataset.shape[0])
        index = (train_dataclassify_y== m)
        weights[(train_dataclassify_y == m)] = 10000
        pijt = Pijt_dt(np.array(train_dataset.loc[:,
                             ['MileageBetweenInspections', 'CurrentOdometer', 'OldOdometer', 'CurrentState', 'OldState']]),
                    weights,
                    label_min, label_max, mileage_max)
        pijt_k.append(pijt)
    output = open('pijt_kbrake_calibrate2.pkl', 'wb')
    pickle.dump(pijt_k, output)
    output.close()
    ## testing
    pkl_file = open(
        'D:/PycharmProject/Mobility/pijt_kbrake_calibrate2.pkl',
        'rb')
    pijt_k = pickle.load(pkl_file)
    pkl_file.close()
    dicts = []
    accu_sum = 0
    accu_fuzzy_sum = 0
    n_all_testdata = 0
    for index in range(len(dicts_testdata)):
        dict_testdata = dicts_testdata[index]
        test_dataset = dict_testdata['test_dataset']
        test_dataclassify_X = test_dataset.loc[:, classify_features]
        test_dataclassify_X = pd.get_dummies(test_dataclassify_X, columns=categorical_cols)
        # Get missing columns in the training test
        missing_cols = set(train_dataclassify_X.columns) - set(test_dataclassify_X.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            test_dataclassify_X[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        test_dataclassify_X = test_dataclassify_X[train_dataclassify_X.columns]
        cluster = classifier.predict(test_dataclassify_X)
        all_y_pred = []
        n_testdata = np.shape(test_dataset)[0]
        for j in range(n_testdata):
            pijt = pijt_k[int(cluster[j])]
            test_sample = test_dataset.iloc[j,:]
            a = test_sample['OldState']
            b = test_sample['MileageBetweenInspections']
            y_pred = predict_state([test_sample['OldState']], [test_sample['MileageBetweenInspections']], pijt, label_min, label_max)
            all_y_pred.append(int(y_pred[0]))
        accuracy, cmatrix, accuracy_fuzzy = evaluate(all_y_pred, test_dataset['CurrentState'].astype(int), label_max)
        # print('accuracy', accuracy)
        # print('accuracy_fuzzy', accuracy_fuzzy)
        dicts.append({'accuracy': accuracy, 'accuracy_fuzzy': accuracy_fuzzy})
        accu_sum = accu_sum + accuracy * n_testdata
        accu_fuzzy_sum = accu_fuzzy_sum + accuracy_fuzzy * n_testdata
        n_all_testdata = n_all_testdata + n_testdata
    print('Accuracy using specific model-calibrated')
    print(accu_sum / n_all_testdata, accu_fuzzy_sum / n_all_testdata)
    #

    ####### predict using the whole dataset - whole dataset prediction and training
    test_dataset = dicts_testdata[0]['test_dataset']
    for i in range(1, len(dicts_testdata)):
        d = dicts_testdata[i]
        test_dataset = pd.concat([test_dataset, d['test_dataset']], axis=0)
    pijt = ctmc.Pijt_dt(np.array(train_dataset.loc[:,['MileageBetweenInspections', 'CurrentOdometer', 'OldOdometer','CurrentState','OldState']]), label_min, label_max, mileage_max)
    test_data = test_dataset.loc[:,['MileageBetweenInspections', 'CurrentOdometer', 'OldOdometer','CurrentState','OldState']]
    y_pred = predict_state(np.array(test_data['OldState']), np.array(test_data['MileageBetweenInspections']), pijt, label_min, label_max)
    accuracy, cmatrix, accuracy_fuzzy = evaluate(y_pred.astype(int), test_dataset['CurrentState'].astype(int), label_max)
    print('Error using generalized model')
    print(accuracy, accuracy_fuzzy)
