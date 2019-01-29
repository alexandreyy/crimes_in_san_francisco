'''
Created on 28/09/2015

@author: Alexandre Yukio Yamashita
'''
from sklearn.neighbors.classification import KNeighborsClassifier

from crimes_data_parsed import CrimeData
from normalize import z_norm_by_feature
import numpy as np


def predict(classifier, X):
    labels = []

    for x_item in X:
        labels.append(clf.predict(x_item))

    labels = np.array(labels)

    return labels.astype(int)

if __name__ == '__main__':
    '''
    Train and test logistic regression.
    '''

    # Load data.
    path_weights = "resources/knn_weights.bin"
    # path_train = "resources/crimes_training_ones.bin"
    path_train = "resources/crimes_samples_training.bin"
    # path_tests = "resources/crimes_testing_ones.bin"
    path_tests = "resources/crimes_samples_testing.bin"

    print "Normalizing train"
    crime_train = CrimeData(path_train)
    crime_train.data[:, 22:24], mean_x_y, std_x_y = z_norm_by_feature(crime_train.data[:, 22:24])
    crime_train.data[:, 1:5], mean_time, std_time = z_norm_by_feature(crime_train.data[:, 1:5])

    print "Normalizing test"
    crime_test = CrimeData(path_tests)
    crime_test.data[:, 22:24] = z_norm_by_feature(crime_test.data[:, 22:24], mean_x_y, std_x_y)
    crime_test.data[:, 1:5] = z_norm_by_feature(crime_test.data[:, 1:5], mean_time, std_time)



    for i in range(100000):
        for i in range(30):
    #         cost = train(trX, trY)
            batch_size = 300000

            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                cost = train(trX[start:end], trY[start:end])
            print cost

        print "Saving"
        save(path_weights, w.get_value())
        print i, np.mean(np.argmax(teY, axis = 1) == predict(teX))

    n = 0.1
    for i in range(1, 10):
        n *= 10
        n = int(n)
        clf = KNeighborsClassifier(n_neighbors = n)
        print "Fitting"
        clf.fit(crime_train.data, crime_train.y)
        print "Testing"
        preds = clf.predict(crime_test.data[0:10000])
        print n, np.mean(crime_test.y[0:10000] == preds[0:10000])
