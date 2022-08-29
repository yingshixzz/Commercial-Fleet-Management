import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import itertools
from functools import reduce
import copy
import pickle

def exponential_distribution(t, lamb):
    return 1-np.exp(-lamb*t)

def weibull_CDF_distribution(t, b,k):
    return 1-np.exp(-b*(t**k))

def weibull_distribution(t, b,k):
    return b*k*(t**(k-1))*np.exp(-b*(t**k))

def Pijt_dt(dataset,label_min, label_max, mileage_max, dt=100):
    # dt = 100    # the accuracy increases when decrease dt
    pijt = np.zeros((label_max+1, label_max+1, mileage_max+1))
    T = range(int(mileage_max+1))
    for i in range(label_min,label_max+1):
        dataset_i = dataset[np.where(dataset[:,4]==i)]
        for l in range(int(dt/2), int(mileage_max+1-dt/2)):
            dataset_il = dataset_i[np.where((dataset_i[:,0]<=(l+dt/2)) & (dataset_i[:,0]>=(l-dt/2)))]
            n_datal = np.shape(dataset_il)[0]
            if n_datal ==0:
                pijt[i, :, l] = pijt[i, :, l-1]
                continue
            for j in range(label_min,label_max+1):
                pijt[i,j,l] = np.count_nonzero(dataset_il[:,3]==j)/n_datal
            k = l
            while np.sum(pijt[i,:,k-1])==0:
                pijt[i, :, k - 1] = pijt[i,:,k]
                k = k-1
        for l in range(int(dt/2)):
            pijt[i,:,l] = pijt[i,:,int(dt/2)]
        for l in range(int(mileage_max+1-dt/2), int(mileage_max+1)):
            pijt[i, :, l] = pijt[i, :, int(mileage_max-dt/2)]

        # ##plot
        # plt.figure(figsize=(14, 8))
        # for j in range(label_min,i+1):
        #     plt.plot(T, pijt[i,j,:], label=str(i) + ' to ' + str(j))
        # plt.xlabel("Mileage between inspections")
        # plt.ylabel("Probability of transition from state i to j after certain mileage")
        # plt.title("transition from state " + str(i))
        # plt.legend()
        # plt.savefig("D:/PycharmProject/Mobility/results/pijt/Probability of transition from state" + str(i) + ".png")
        # # plt.show()
        # plt.clf()
    return pijt

def Pijt_cluster(dataset, label_min, label_max, mileage_max, fill_thre=1, n_fill=20):
    # mbis = dataset[:, 0]
    # state_origin = dataset[:, 4]
    pijt = np.zeros((label_max+1, label_max+1, mileage_max+1))
    T = range(int(mileage_max+1))
    n_data = np.shape(dataset)[0]
    drlist = (dataset[:,4]-dataset[:,3])/dataset[:,0]
    # n_fill = 50
    # fill_thre = 1
    for i in range(label_min,label_max+1):
        dataset_i = dataset[np.where(dataset[:,4] == i)]
        for l in range(mileage_max+1):
            dataset_il = dataset_i[np.where(dataset_i[:, 0] == l)]
            n_data_il = np.shape(dataset_il)[0]
            if n_data_il < fill_thre:
            # if n_data_il == 0:
                # distance = np.power((dataset[:,4]-np.ones(n_data)*i), 2) + np.power((dataset[:,0]-np.ones(n_data)*l),2)
                distance = np.power((dataset[:, 4] - np.ones(n_data) * i)/(label_max-label_min+1), 2) + np.power((dataset[:, 0] - np.ones(n_data) * l)/(mileage_max), 2)
                dr_sele = drlist[np.argsort(distance)][:n_fill]
                dataset_il = np.zeros((n_fill, 5))
                dataset_il[:, 0] = l
                dataset_il[:, 4] = i
                dataset_il[:, 3] = ((np.ones(n_fill)*i) - (np.ones(n_fill)*l) * dr_sele).astype(int)
                dataset_il[np.where(dataset_il[:, 3] < 2)[0], 3] = 2
                # dataset_il = np.vstack((dataset_il, data_fill))
            for j in range(label_min, label_max + 1):
                pijt[i, j, l] = np.count_nonzero(dataset_il[:,3] == j)/np.shape(dataset_il)[0]
            # print(np.sum(pijt[i,:,l]))
        ##plot
        plt.figure(figsize=(14, 8))
        for j in range(label_min, i + 1):
            plt.plot(T[1:], pijt[i, j, 1:], label=str(i) + ' to ' + str(j))
        plt.xlabel("Mileage between inspections")
        plt.ylabel("Probability of transition from state i to j after certain mileage")
        plt.title("transition from state " + str(i))
        plt.legend()
        plt.savefig(
            "D:/PycharmProject/Mobility/results/pijt_cluster20000/Probability of transition from state" + str(i) + ".png")
        # plt.show()
        plt.clf()
    return pijt

def evaluate(test_dataset, Fijt_bk):
    n_testdataset = np.shape(test_dataset)[0]
    y_pred = np.zeros(n_testdataset)
    for n in range(n_testdataset):
        test_data = test_dataset[n, :]
        pij = np.zeros(19)
        i = int(test_data[4])
        for j in range(2,18):
            if i ==j :
                pij[j] = 1 - weibull_CDF_distribution(test_data[0], Fijt_bk[i, j, 0], Fijt_bk[i, j, 1])
            else:
                pij[j] = weibull_CDF_distribution(test_data[0], Fijt_bk[i,j,0],Fijt_bk[i,j,1])
        y_pred[n] = np.argmax(pij)
    y_true = test_dataset[:,3]
    print(y_pred)
    print(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    cmatrix = confusion_matrix(y_true, y_pred, labels=range(3,18))
    return accuracy, cmatrix

def tranmatrix(Fii1t_bk, i, mileage, label_min, label_max):
    pij = np.zeros(label_max)
    for j in range(label_min,label_max):
        pij[j] = weibull_CDF_distribution(mileage, Fii1t_bk[i, j, 0], Fii1t_bk[i, j, 1])
    # if (Fii1t_bk[i, i, 0]==0) & (Fii1t_bk[i, i, 1]==0):
    #     pij[i] = 1
    return pij

def predict(test_i, test_mileage, pijt, label_min, label_max):
    n_test = len(test_mileage)
    y_pred = np.zeros(n_test)
    for n in range(n_test):
        pij = pijt[int(test_i[n]), :, int(test_mileage[n])]
        # y_pred[n] = np.argmax(pij)
        # y_pred[n] = np.random.choice(range(label_max + 1), 1, p=pij)[0]
        if np.sum(pij)==0:
            y_pred[n] = int(test_i[n])
        else:
            y_pred[n] = np.random.choice(range(label_max+1),1, p=pij)[0]
    return y_pred

def predict_lower(test_i, test_mileage, pijt, label_min, label_max):
    n_test = len(test_mileage)
    y_pred = np.zeros(n_test)
    for n in range(n_test):
        pij = pijt[int(test_i[n]), :, int(test_mileage[n])]
        # y_pred[n] = np.argmax(pij)
        # y_pred[n] = np.random.choice(range(label_max + 1), 1, p=pij)[0]
        if np.sum(pij)==0:
            y_pred[n] = int(test_i[n])
        else:
            y_pred[n] = (pij!=0).argmax(axis=0)
    return y_pred

def predict_upper(test_i, test_mileage, pijt, label_min, label_max):
    n_test = len(test_mileage)
    y_pred = np.zeros(n_test)
    for n in range(n_test):
        pij = pijt[int(test_i[n]), :, int(test_mileage[n])]
        # y_pred[n] = np.argmax(pij)
        # y_pred[n] = np.random.choice(range(label_max + 1), 1, p=pij)[0]
        if np.sum(pij)==0:
            y_pred[n] = int(test_i[n])
        else:
            y_pred[n] =label_max - (np.array(list(reversed(np.array(pij))))!=0).argmax(axis=0)
    return y_pred

def riskdistribution(test_i, test_mileage, pijt, label_min, label_max):
    n_test = len(test_mileage)
    pi2 = np.zeros(n_test)
    for n in range(n_test):
        pij = pijt[int(test_i[n]), :, int(test_mileage[n])]
        if np.sum(pij)==0:
            if int(test_i[n])==2:
                pi2[n] = 1
        else:
            pi2[n] = pij[2]   #label risk is 2
    pi2 [pi2 <0.00001] =0
    pi2_ = np.ones(n_test) - pi2
    list_vehicles = np.array(np.nonzero(pi2)[0]).tolist()
    p2 = np.zeros(n_test+1)
    for n_risk in range(len(list_vehicles)+1):
        # index_comb = list(itertools.combinations(list_vehicles, n_risk))
        # for index in index_comb:
        for index in itertools.combinations(list_vehicles, n_risk):
            p_risk = copy.deepcopy(pi2_)
            p_risk[list(index)] = pi2[list(index)]
            p2[n_risk] = p2[n_risk] + reduce(prod,p_risk)
    return p2

def prod(x,y):
    return x*y

def evaluate(y_pred, y_true, label_max):
    accuracy = accuracy_score(y_true, y_pred)
    cmatrix = confusion_matrix(y_true, y_pred, labels=range(0,label_max+1))
    # a= np.where(np.abs(y_true-y_pred)<=1)
    accuracy_fuzzy= np.size(np.where(np.abs(y_true-y_pred)<=1)[0])/np.size(y_pred)
    return accuracy, cmatrix, accuracy_fuzzy

def histogram_generation(data, n_generateddata):
    hist, bins = np.histogram(data, bins=100, range=(np.min(data), np.max(data)), density=True, normed=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    # generate data with double random()
    # n_generateddata = 500
    generatedData = np.zeros(n_generateddata)
    maxData = np.max(data)
    minData = np.min(data)
    i = 0
    while i < n_generateddata:
        randNo = np.random.rand(1) * (maxData - minData) - np.absolute(minData)
        if np.random.rand(1) <= hist[np.argmax(randNo < (center + (bins[1] - bins[0]) / 2)) - 1]:
            generatedData[i] = randNo
            i += 1
    # normalized histogram of generatedData
    hist2, bins2 = np.histogram(generatedData, bins=100, range=(np.min(data), np.max(data)), normed=True)
    width2 = 0.7 * (bins2[1] - bins2[0])
    center2 = (bins2[:-1] + bins2[1:]) / 2

    # plot both histograms
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(14, 8), sharex=True, sharey=True, )
    ax1.set_title('Original data')
    ax1.bar(center, hist, align='center', width=width, fc='#AAAAFF')
    ax2.set_title('Generated data')
    ax2.bar(center2, hist2, align='center', width=width2, fc='#AAAAFF')
    # fig.suptitle(f"{name} Deterioration Rate Histogram Generation")
    fig.suptitle("Mileage between inspections")
    plt.show()
    return generatedData

def state_generation(data, n_generateddata):
    mu = np.mean(data)
    sigma = np.std(data)
    max_state = np.max(data)
    min_state = np.min(data)
    num_bins = int(max_state - min_state +1)
    # n, bins, patches = plt.hist(data, num_bins, density=1, alpha=0.75)
    generatedData = np.random.normal(mu, sigma, n_generateddata*4)
    generatedData = np.round(generatedData, 0)
    generatedData = generatedData[np.where((generatedData>=min_state) & (generatedData<=max_state))]
    generatedData = generatedData[:n_generateddata]

    # plot both histograms
    # normalized histogram of generatedData
    hist, bins = np.histogram(data, bins=num_bins, range=(min_state, max_state), density=True, normed=True)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    hist2, bins2 = np.histogram(generatedData, bins=num_bins, range=(min_state, max_state), density=True, normed=True)
    width2 = 0.7 * (bins2[1] - bins2[0])
    center2 = (bins2[:-1] + bins2[1:]) / 2
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(14, 8), sharex=True, sharey=True, )
    ax1.set_title('Original data')
    ax1.bar(center, hist, align='center', width=width, fc='#AAAAFF')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Probability')
    ax2.set_title('Generated data')
    ax2.bar(center2, hist2, align='center', width=width2, fc='#AAAAFF')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Probability')
    # fig.suptitle(f"{name} Deterioration Rate Histogram Generation")
    fig.suptitle("State distribution for simulation")
    # fig.suptitle("Deterioration Rate")
    plt.show()

    # #plot
    # plt.grid(True)
    # y = norm.pdf(bins, mu, sigma)
    # plt.plot(bins, y, 'r--')
    # plt.xlabel('State')
    # plt.ylabel('Probability')
    # plt.title('State distribution: $\mu$=' + str(round(mu,2))+' $\sigma=$' +str(round(sigma,2)))
    # plt.show()
    return generatedData

def plot_hist(data,name, xlabel, ylabel):
    # hist, bins = np.histogram(data, bins=100, range=(np.min(data), np.max(data)), density=True, normed=True)
    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # # plot both histograms
    # fig, ax= plt.subplots(ncols=2, nrows=1, figsize=(14, 8), sharex=True, sharey=True )
    # # ax.title('Original data')
    # ax.bar(center, hist, align='center', width=width, fc='#AAAAFF')
    # # fig.suptitle(f"{name} Deterioration Rate Histogram Generation")
    # # ax.set_xlabel(xlabel)
    # # ax.set_ylabel(ylabel)
    # fig.title(name)
    # # ax.set_title(name)
    # plt.show()
    plt.hist(data,bins=range(int(np.min(data)), int(np.max(data)+1),1), density=True)
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()

def prank(test_i, test_mileage, pijt, risk_label):
    pirt = np.zeros(len(test_i))
    for r in range(2, risk_label+1):
        for i in range(len(test_i)):
            pirt[i] = pirt[i] + pijt[int(test_i[i]), r, int(test_mileage[i])]
    rank_id = sorted(range(len(pirt)), key=lambda k: pirt[k], reverse=True)
    return pirt, rank_id



if __name__ == "__main__":
    # component = 'FTire'
    # component = 'RTire'
    # component = 'FBrake'
    # component = 'RBrake'
    component = 'FBrake_RBrake'
    # component = 'FTire_RTire'
    # dataset = pd.read_csv('D:/PycharmProject/Mobility/dataset/' + component + 'data.csv')
    dataset = pd.read_csv('D:/PycharmProject/Mobility/dataset/' + component + 'dataheavy.csv')    #0col: mileage between inspections; 1col: current odometer; 2col: old odometer; 3col: current state; 4col: old state

    #hyper-parameter
    dt = 100    # the accuracy increases when decrease dt

    ## check dataset (brake)
    label_min = 2
    label_max = 17
    dr_max = 0.007
    mileage_max = 20000
    # # check dataset (tire)
    # label_min = 2
    # label_max = 30
    # dr_max = 0.01
    # mileage_max = 40000

    # label_min = int(min(dataset.iloc[:, 3]))
    # label_max = int(max(dataset.iloc[:, 4]))


    n_test = 1000
    n_labels = int(label_max - label_min + 1)
    dataset = dataset[(dataset['MileageBetweenInspections'] > 0) & (dataset['MileageBetweenInspections'] <= mileage_max)]
    dataset = dataset[dataset.iloc[:, 3] <= dataset.iloc[:, 4]]
    dataset = dataset[(dataset.iloc[:, 3] >= label_min) & (dataset.iloc[:, 4] >= label_min)]
    dataset = dataset[(dataset.iloc[:, 3] <= label_max) & (dataset.iloc[:, 4] <= label_max)]
    dataset = np.array(dataset)
    dataset = dataset[np.where(((dataset[:,3] - dataset[:,4])/dataset[:,0])>=-dr_max)]
    train_dataset = dataset[n_test:, :]
    # train_dataset = dataset[8000:9000, :]


    # pijt = Pijt_dt(train_dataset, label_min, label_max, mileage_max, dt)
    pijt = Pijt_cluster(train_dataset, label_min, label_max, mileage_max)

    output = open('pijtbrake' +str(mileage_max) +'.pkl', 'wb')
    pickle.dump(pijt, output)
    output.close()

    test_dataset = pd.read_csv('D:/PycharmProject/Mobility/dataset/' + component + 'dataheavy.csv')
    test_dataset = test_dataset[(test_dataset['MileageBetweenInspections'] > 0) & (test_dataset['MileageBetweenInspections'] <= mileage_max)]
    test_dataset = test_dataset[test_dataset.iloc[:, 3] <= test_dataset.iloc[:, 4]]
    test_dataset = test_dataset[(test_dataset.iloc[:, 3] >= label_min) & (test_dataset.iloc[:, 4] >= label_min)]
    test_dataset = test_dataset[(test_dataset.iloc[:, 3] <= label_max) & (test_dataset.iloc[:, 4] <= label_max)]
    test_dataset = np.array(test_dataset)
    # test_dataset = test_dataset[np.where(((test_dataset[:,3] - test_dataset[:,4])/test_dataset[:,0]) >= -dr_max)]
    test_dataset = test_dataset[:n_test, :]
    # test_dataset = np.vstack((test_dataset[:8000, :], test_dataset[9000:, :]))
    test_i = test_dataset[:,4]

    # rank -given last states, given mileages, predict the next states
    pirt, rank_id=prank(test_i, test_dataset[:,0], pijt, 2)
    ranked = np.vstack((rank_id, pirt[rank_id]))
    ranked = pd.DataFrame(ranked)
    ranked.to_csv("D:/PycharmProject/Mobility/dataset/ranked1000_prediction.csv", index=False, sep=',')
    # prediction
    y_pred = predict(test_i,  test_dataset[:,0], pijt, label_min, label_max)
    accuracy, cmatrix, accuracy_fuzzy = evaluate(y_pred, test_dataset[:,3], label_max)
    print(accuracy, cmatrix, accuracy_fuzzy)

    # ##### cross validation
    # n_vehicles = 1000
    # n_iter = int(np.shape(dataset)[0] / n_vehicles)
    # cross_accuracy = 0
    # cross_accuracy_fuzzy = 0
    # for i in range(n_iter):
    #     test_dataset = dataset[i * 1000:(i + 1) * 1000, :]
    #     train_dataset = np.vstack((dataset[:i * 1000, :], dataset[(i + 1) * 1000:, :]))
    #     # pijt = Pijt_dt(train_dataset, label_min, label_max, mileage_max, dt)
    #     pijt = Pijt_cluster(train_dataset, label_min, label_max, mileage_max)
    #     y_pred = predict(test_dataset[:,4], test_dataset[:, 0], pijt, label_min, label_max)
    #     accuracy, cmatrix, accuracy_fuzzy = evaluate(y_pred, test_dataset[:, 3], label_max)
    #     cross_accuracy += accuracy
    #     cross_accuracy_fuzzy += accuracy_fuzzy
    # cross_accuracy = cross_accuracy/n_iter
    # cross_accuracy_fuzzy = cross_accuracy_fuzzy/n_iter
    # print(cross_accuracy, cross_accuracy_fuzzy)

    # plot_hist(test_dataset[:, 4],'distribution of old fleet '+component +' state',component, 'density')
    # plot_hist(test_dataset[:,3], 'distribution of true fleet '+component +' state after one year',component, 'density')
    # plot_hist(y_pred, 'distribution of predicted fleet '+component +' state after one year',component, 'density')
    #
    # # # Probability of number of cases in risk(prediction)
    # # p2 = riskdistribution(test_i, test_dataset[:,0], pijt, label_min, label_max)
    # # print(p2)
    # # X= np.array(np.nonzero(p2)[0]).tolist()
    # # Y = p2[np.array(np.nonzero(p2)[0]).tolist()]
    # # plt.bar(np.array(np.nonzero(p2)[0]).tolist(), p2[np.array(np.nonzero(p2)[0]).tolist()])
    # # plt.xlabel('Number of cases in risk')
    # # plt.ylabel('Probability')
    # # plt.title('Probability of number of cases in risk(prediction)')
    # # for a,b in zip(X,Y):
    # #     plt.text(a, b+0.01, '%.6f'%(b*100) +'%', ha='center', va='bottom', fontsize=8)
    # # plt.show()
    #
    # print("start simulation")
    # #simulation
    # # test_i = state_generation(train_dataset[:, 4], n_test)    #simulate the original state
    # test_mileage = histogram_generation(train_dataset[:,0], n_test)
    # # prediction
    # y_pred_simu = predict(test_i, test_mileage, pijt, label_min, label_max)
    # accuracy_simu, cmatrix_simu, accuracy_fuzzy_simu = evaluate(y_pred_simu, test_dataset[:, 3], label_max)
    # print(accuracy_simu, cmatrix_simu, accuracy_fuzzy_simu)
    # plot_hist(y_pred_simu, 'distribution of predicted fleet '+component +' state after one year -with simulation',component, 'density')
    #
    # # rank -given last states, simulate mileages, predict the next states
    # pirt, rank_id=prank(test_i, test_mileage, pijt, 2)
    # ranked = np.vstack((rank_id, pirt[rank_id]))
    # ranked = pd.DataFrame(ranked)
    # ranked.to_csv("D:/PycharmProject/Mobility/dataset/ranked1000_simuprediction.csv", index=False, sep=',')
    #
    # p2_simu = riskdistribution(test_i, test_mileage, pijt, label_min, label_max)
    # print(p2_simu)
    # # print(np.sum(p2))
    # X= np.array(np.nonzero(p2_simu)[0]).tolist()
    # Y = p2_simu[np.array(np.nonzero(p2_simu)[0]).tolist()]
    # plt.bar(np.array(np.nonzero(p2_simu)[0]).tolist(), p2_simu[np.array(np.nonzero(p2_simu)[0]).tolist()])
    # plt.xlabel('Number of cases in risk')
    # plt.ylabel('Probability')
    # plt.title('Probability of number of cases in risk(prediction&simulation)')
    # for a,b in zip(X,Y):
    #     plt.text(a, b+0.01, '%.6f'%(b*100) +'%', ha='center', va='bottom', fontsize=8)
    # plt.show()
    #
    # ## cumulated state distribution for a fleet with 1000 vehicles
    # # percent_matrix = np.zeros((4,label_max+1))
    # # for i in range(label_max+1):
    # #     percent_matrix[0,i] =np.size(np.where(test_dataset[:,4] <= i)[0])
    # #     percent_matrix[1, i] = np.size(np.where(test_dataset[:, 3] <= i)[0])
    # #     percent_matrix[2, i] = np.size(np.where(y_pred <= i)[0])
    # #     percent_matrix[3, i] = np.size(np.where(y_pred_simu <= i)[0])
    # # percent_matrix = percent_matrix/np.size(y_pred)
    # # print(percent_matrix)
    #
    # # lower and upper bound state distribution for a fleet with 1000 vehicles
    # y_pred_simu_low = predict_lower(test_i, test_mileage, pijt, label_min, label_max)
    # y_pred_simu_up = predict_upper(test_i, test_mileage, pijt, label_min, label_max)
    #
    # # state distribution for a fleet with 1000 vehicles
    # percent_matrix = np.zeros((6,label_max+1))
    # for i in range(label_max+1):
    #     percent_matrix[0,i] = np.size(np.where(test_dataset[:,4] == i)[0])
    #     percent_matrix[1, i] = np.size(np.where(test_dataset[:, 3] == i)[0])
    #     percent_matrix[2, i] = np.size(np.where(y_pred == i)[0])
    #     percent_matrix[3, i] = np.size(np.where(y_pred_simu == i)[0])
    #     percent_matrix[4, i] = np.size(np.where(y_pred_simu_low == i)[0])
    #     percent_matrix[5, i] = np.size(np.where(y_pred_simu_up == i)[0])
    # percent_matrix = percent_matrix/np.size(y_pred)
    # print(percent_matrix)
    #
    #
    #
    #
    #
