'''
Created on 27/09/2015

@author: Alexandre Yukio Yamashita
'''

import numpy as np


class CrimeData:
    '''
    Read parsed crime data.
    '''

    def __init__(self, path_data):
        f = file(path_data, "rb")
        self.data = np.load(f)
        self.data_labels = np.load(f)
        self.y = np.load(f)
        self.y_labels = np.load(f)
        f.close()


    def save(self, path_data):
        '''
        Save data.
        '''

        f = file(path_data, "wb")
        np.save(f, self.data)
        np.save(f, self.data_labels)
        np.save(f, self.y)
        np.save(f, self.y_labels)
        f.close()


if __name__ == '__main__':
    '''
    Load crime data.
    '''

    path_data = "resources/crimes.bin"
    crime_data = CrimeData(path_data)
    crime_data.data = crime_data.data[300000:0]
    crime_data.y = crime_data.y[300000:0]
    crime_data.save("resources/crimes_samples_testing.bin")

#     print crime_data.data.shape
#
#     for i in range(len(crime_data.data[0])):
#         print crime_data.data_labels[i], type(crime_data.data[0][i]), crime_data.data[0][i]
