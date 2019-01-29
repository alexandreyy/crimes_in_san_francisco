'''
Created on 28/09/2015

@author: Alexandre Yukio Yamashita
'''
from theano import tensor as T
import theano

from crimes_data_parsed import CrimeData
from normalize import z_norm_by_feature
import numpy as np


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

def floatX(X):
    return np.asarray(X, dtype = theano.config.floatX)  # @UndefinedVariable

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.001))

def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

def save(path, w):
    f = file(path, "wb")
    np.save(f, w)
    f.close()

def load(path):
    f = file(path, "rb")
    w = np.load(f)
    f.close()
    return theano.shared(floatX(w))

if __name__ == '__main__':
    '''
    Train and test logistic regression.
    '''

    # Load data.
    path_weights = "resources/lr_weights.bin"
    path_train = "resources/crimes_training_ones.bin"
    # path_train = "resources/crimes_samples_training.bin"
    path_tests = "resources/crimes_testing_ones.bin"
    # path_tests = "resources/crimes_samples_testing.bin"

    print "Normalizing train"
    crime_train = CrimeData(path_train)
    crime_train.data[:, 22:24], mean_x_y, std_x_y = z_norm_by_feature(crime_train.data[:, 22:24])
    crime_train.data[:, 1:5], mean_time, std_time = z_norm_by_feature(crime_train.data[:, 1:5])

    print "Normalizing test"
    crime_test = CrimeData(path_tests)
    crime_test.data[:, 22:24] = z_norm_by_feature(crime_test.data[:, 22:24], mean_x_y, std_x_y)
    crime_test.data[:, 1:5] = z_norm_by_feature(crime_test.data[:, 1:5], mean_time, std_time)
    n = np.max(np.hstack((crime_test.y, crime_train.y))) + 1

    print "One hot"
    trY = one_hot(crime_train.y, n)
    teY = one_hot(crime_test.y, n)

    print "Tex"
    teX = crime_test.data

    print "Trx"
    trX = crime_train.data

    X = T.fmatrix()
    Y = T.fmatrix()

    # w = init_weights((len(trX[0]), n))
    w = load(path_weights)

    py_x = model(X, w)
    y_pred = T.argmax(py_x, axis = 1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    gradient = T.grad(cost = cost, wrt = w)
    update = [[w, w - gradient * 9.0]]

    train = theano.function(inputs = [X, Y], outputs = cost, updates = update, allow_input_downcast = True)
    predict = theano.function(inputs = [X], outputs = y_pred, allow_input_downcast = True)

    precision = 0
    recall = 0

    for i in range(len(teY[0])):
        print i
        where = np.where(teY[:, i] == 1)
        where2 = np.where(teY[:, i] == 0)

        TP_FN = len(where[0])

        TP = sum(i == predict(teX[where]))
        FP = sum(i == predict(teX[where2]))
        TP_FP = TP + FP

        if TP_FP == 0:
            TP_FP = 1.0

        if TP_FN == 0:
            TP_FN = 1.0

        recall += TP * 1.0 / (TP_FN * len(teY[0]))
        precision += TP * 1.0 / (TP_FP * len(teY[0]))

    print "Precision", precision * 1.0
    print "Recall", recall * 1.0
    print "Accuracy", np.mean(np.argmax(teY, axis = 1) == predict(teX))

#     for i in range(100000):
#         for i in range(30):
#     #         cost = train(trX, trY)
#             batch_size = 300000
#
#             for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
#                 cost = train(trX[start:end], trY[start:end])
#             print cost
#
#         print "Saving"
#         save(path_weights, w.get_value())
#         print i, np.mean(np.argmax(teY, axis = 1) == predict(teX))
