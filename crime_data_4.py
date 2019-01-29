'''
Created on 21/09/2015

@author: Alexandre Yukio Yamashita
'''
from numpy import int16, int8, float32, uint16, uint8
import seaborn as sns
from matplotlib import pyplot as plt

from k_means import kmeans
import numpy as np
import pandas as pd


class CrimeData:
    '''
    Read crime data and transform in features.
    '''

    def __init__(self, path = ""):
        if path != "":
            self.data = pd.read_csv(path, quotechar = '"', skipinitialspace = True)
            self.data = self.data.as_matrix()

            labels = ["year", "month", "day", "time"]
            years = []
            months = []
            days = []
            times = []

            for data in self.data[:, 0]:
                splitted_timestamp = data.split("-")
                years.append(int(splitted_timestamp[0]))
                months.append(int(splitted_timestamp[1]))
                day_time = splitted_timestamp[2].split()
                days.append(int(day_time[0]))
                splitted_time = day_time[1].split(":")

                tempo = int(float(splitted_time[0]) + float(splitted_time[1]) / 60.0 + 0.5)
                if tempo < 3:
                    tempo = "0-3"
                elif tempo < 6:
                    tempo = "3-6"
                elif tempo < 9:
                    tempo = "6-9"
                elif tempo < 12:
                    tempo = "9-12"
                elif tempo < 15:
                    tempo = "12-15"
                elif tempo < 6:
                    tempo = "18-21"
                else:
                    tempo = "21-24"

                times.append(tempo)

            data = self.data[:, 1]

            times = np.array(times)
            print times.shape
            data = data[np.where(times == "12-15")]
            labels = np.unique(data)

            print labels.shape

            data = data.tolist()
            frequencies = [data.count(i) for i in labels]
            frequencies, labels = (list(t) for t in zip(*sorted(zip(frequencies, labels))))
            frequencies = frequencies[::-1]
            labels = labels[::-1]

            frequencies = frequencies[0:10]
            labels = labels[0:10]
            frequencies = frequencies[::-1]
            labels = labels[::-1]

            y_pos = np.arange(len(labels))

            cmap = plt.get_cmap('gray')
            indices = np.linspace(0, cmap.N, len(labels))
            my_colors = [cmap(int(i)) for i in indices]

            plt.barh(y_pos, frequencies, align = 'center', color = my_colors)
            plt.yticks(y_pos, labels)
            plt.xlabel('Quantidade de crimes')
            plt.ylabel('Categorias')
            plt.title('Crimes mais comuns')
            plt.tight_layout()

            plt.show()

            exit()


            address_labels, address_features = self._parse_address()

            print "Description."
            description_labels, description_features = self._parse_description()
            print "Time."
            time_stamp_labels, time_stamp_features = self._parse_time_stamps()
            days_of_week_labels, days_of_week_features = self._parse_days_of_week()
            district_labels, district_features = self._parse_districts()
            resolution_labels, resolution_features = self._parse_resolutions()

            print "Stacking labels."
            self.data_labels = np.hstack((["bias"], time_stamp_labels, days_of_week_labels, district_labels, ['x', 'y'], \
                                          resolution_labels, address_labels, description_labels))
            self.y_labels, self.y = np.unique(self.data[:, 1], return_inverse = True)

            print "Stacking features."
            x_y = np.array(self.data[:, 7:9])
            data_ones = np.ones((len(self.data), 1))
            data_ones = data_ones.astype(bool)

            self.data = np.hstack((data_ones, time_stamp_features, days_of_week_features))
            print self.data.shape

            self.data = np.hstack((self.data, district_features))
            print self.data.shape

            self.data = np.hstack((self.data, x_y))
            print self.data.shape

            self.data = np.hstack((self.data, resolution_features))
            print self.data.shape

            self.data = np.hstack((self.data, address_features))
            print self.data.shape

            self.data = np.hstack((self.data, description_features))
            print self.data.shape

            self.data[:, 0] = self.data[:, 0].astype(bool)
            self.data[:, 1] = self.data[:, 1].astype(uint16)
            self.data[:, 2] = self.data[:, 2].astype(uint8)
            self.data[:, 3] = self.data[:, 3].astype(uint8)

            for i in range(5, len(district_labels) + len(days_of_week_labels) + 5):
                self.data[:, i] = self.data[:, i].astype(bool)



    def _parse_description(self):
        '''
        Convert description string to array.
        '''
        self.description_labels, self.description = np.unique(self.data[:, 2], return_inverse = True)
        dictionary = self._create_dictionary()

        data = np.array([])
        data.shape = (0, len(self.data[:, 2]))
        print dictionary.shape
        for label in dictionary:
            data = np.vstack((data, np.array([label in i for i in self.data[:, 2]])))

        data = data.astype(bool)

        return dictionary, data.transpose()


    def _create_dictionary(self):
        '''
        Create dictionary for description.
        '''

        descriptions = np.unique(self.data[:, 2])
        return descriptions

        categories = np.unique(self.data[:, 2])
        print "Creating dictionary."

        dictionary = np.hstack((\
            ["PROSTITUTION", "JUVENILE", "DRIVING WHILE", "ACCESS CARD INFORMATION",
             "ACCIDENTAL", "AGGRAVATED ASSAULT", "AIDED CASE"
             "ALCOHOL", "ANIMAL", "RAPE", "THEFT", "HOMICIDE", "KIDNAP", "MAYHEM",
             "ROBBERY", "BATTERY", "BURGLARY", "CARJACKING", "WEAPON", "ABUSE",
             "CREDIT CARD", "DAMAGE", "DEATH", "DEFRAUDING", "DESTROY", "DESTRUCT",
             "DISTURB", "DOG, FIGHTING", "DOG", "DRUG", "MINOR", "COCAINE", "MARIJUANA",
             "FALSE", "FALSIF", "FIREARM", "FORGE", "FRAUDULENT", "HAZARDOUS", "ILLEGAL",
             "INDECENT", "INJURY", "INTERFERRING", "LICENSE PLATE", "LOITERING",
             "LOST PROPERTY", "MALICIOUS MISCHIEF", "MAYHEM", "MISCELLANEOUS",
             "OBSCENE", "OBSTRUCT", "COPULATION", "PERMIT VIOLATION", "PEYOTE",
             "BOMB", "GUN", "BARBITUATES", "HEROIN", "GAMBLING", "METHADONE",
             "AMPHETAMINE", "OPIUM", "OPIATES", "NARCOTICS", "HALLUCINOGENIC",
             "PROBATION", "ROBBERY", "VEHICLE", "BANK", "SEXUAL", "VIOLATION",
             "STOLEN", "TRANSPORTATION", "TRESPASSING", "VANDALISM", "WARRANT"], \
            categories))

        related_word = np.hstack((\
            ["PROSTITUTION", "JUVENILE", "DRIVING WHILE", "ACCESS CARD INFORMATION",
             "ACCIDENTAL", "ASSAULT", "AIDED CASE",
             "ALCOHOL", "ANIMAL", "RAPE", "THEFT", "HOMICIDE", "KIDNAP", "MAYHEM",
             "ROBBERY", "BATTERY", "BURGLARY", "CARJACKING", "WEAPON", "ABUSE",
             "CREDIT CARD", "DAMAGE", "DEATH", "DEFRAUDING", "DESTROY", "DESTRUCT",
             "DISTURB", "DFIGHTING", "DOG", "DRUG", "MINOR", "COCAINE", "MARIJUANA",
             "FALSE", "FALSIF", "FIREARM", "FORGE", "FRAUDULENT", "HAZARDOUS", "ILLEGAL",
             "INDECENT", "INJURY", "INTERFERRING", "LICENSE PLATE", "LOITERING",
             "LOST PROPERTY", "MALICIOUS MISCHIEF", "MAYHEM", "MISCELLANEOUS",
             "OBSCENE", "OBSTRUCT", "COPULATION", "PERMIT VIOLATION", "PEYOTE",
             "BOMB", "GUN", "BARBITUATES", "HEROIN", "GAMBLING", "METHADONE",
             "AMPHETAMINE", "OPIUM", "OPIATES", "NARCOTICS", "HALLUCINOGENIC",
             "PROBATION", "ROBBERY", "VEHICLE", "BANK", "SEXUAL", "VIOLATION",
             "STOLEN", "TRANSPORTATION", "TRESPASSING", "VANDALISM", "WARRANT"], \
            categories))

        print "Finding words."
        for index_description in range(len(descriptions)):
            for index_word in range(len(dictionary)):
                if dictionary[index_word] in descriptions[index_description]:
                    descriptions[index_description] = related_word[index_word]

        print "Created dictionary."
        del categories
        del related_word
        del dictionary
        dictionary = np.unique(descriptions)
        del descriptions

        return dictionary


    def _parse_time_stamps(self):
        '''
        Convert time stamp string to array.
        '''

        labels = ["year", "month", "day", "time"]
        years = []
        months = []
        days = []
        times = []

        for data in self.data[:, 0]:
            splitted_timestamp = data.split("-")
            years.append(int(splitted_timestamp[0]))
            months.append(int(splitted_timestamp[1]))
            day_time = splitted_timestamp[2].split()
            days.append(int(day_time[0]))
            splitted_time = day_time[1].split(":")
            times.append(float(splitted_time[0]) + float(splitted_time[1]) / 60.0)

        del splitted_timestamp

        years = np.array(years)
        years = years.astype(int16)
        months = np.array(months)
        months = months.astype(int8)
        days = np.array(days)
        days = days.astype(int8)
        times = np.array(times)
        times = times.astype(float32)

        return labels, np.vstack((years, months, days, times)).transpose()


    def _binarize_feature(self, original_data, labels):
        '''
        Binarize categorical features.
        '''

        data = np.array([])
        data.shape = (0, len(original_data))

        for label in labels:
            data = np.vstack((data, original_data == label))

        data = data.astype(bool)

        return data.transpose()


    def _parse_days_of_week(self):
        '''
        Convert days of week to binary arrays.
        '''

        labels = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        data = self._binarize_feature(self.data[:, 3], labels)

        return labels, data


    def _parse_resolutions(self):
        '''
        Convert resolutions to binary arrays.
        '''

        labels = np.unique(self.data[:, 5])
        data = self._binarize_feature(self.data[:, 5], labels)

        return labels, data


    def _parse_districts(self):
        '''
        Convert districts to binary arrays.
        '''

        labels = np.unique(self.data[:, 4])
        data = self._binarize_feature(self.data[:, 4], labels)

        return labels, data


    def _parse_address(self):
        '''
        Convert address to binary arrays.
        '''

        data = []
        print "Parsing address."
        print len(np.unique(self.data[:, 6]))

        for index in range(len(self.data[:, 6])):
            splitted_address = self.data[:, 6][index].split(' ', 1)

            if self._is_int(splitted_address[0]):
                data.append(splitted_address[1])
            else:
                data.append(self.data[:, 6][index])

            self.data[index][6] = data[index]

        labels = np.unique(data)
        data = np.array(data)
        means_address = []

        for index_label in range(len(labels)):
            address_data = self.data[np.where(data == labels[index_label])]
            means_address.append([np.mean(address_data[:, 7]), np.mean(address_data[:, 8])])

        means_address = np.float32(np.array(means_address))

        cond_1 = means_address[np.where(means_address[:, 1] > 85)]
        centroid1 = cond_1[0]
        cond_2 = means_address[np.where(np.logical_and(means_address[:, 1] > 68, means_address[:, 1] < 80))]
        centroid2 = cond_2[0]
        cond_3 = means_address[np.where(np.logical_and(means_address[:, 1] > 50, means_address[:, 1] < 70))]
        centroid3 = cond_3[0]
        total = len(means_address)

#         centroid1 = means_address[np.random.randint(total)]
#         centroid2 = means_address[np.random.randint(total)]
#         centroid3 = means_address[np.random.randint(total)]

        rets = []

        # for n in range(10, 11):
        for n in range(100, 101):
            center = None
            label = None
            ret = 9999999

            for _ in range(5):
                centroids = []

                if n == 2:
                    centroids.append(centroid2)
                    centroids.append(means_address[np.random.randint(total)])
                    centroids = np.float32(centroids)
                elif n == 3:
                    centroids.append(centroid1)
                    centroids.append(centroid2)
                    centroids.append(means_address[np.random.randint(total)])
                    centroids = np.float32(centroids)
                elif n == 4:
                    centroids.append(centroid1)
                    centroids.append(centroid2)
                    centroids.append(centroid3)
                    centroids.append(means_address[np.random.randint(total)])
                else:
                    centroids.append(centroid1)
                    centroids.append(centroid2)
                    centroids.append(centroid3)
                    centroids = np.float32(centroids)
                    centroids = np.vstack((centroids, means_address[np.random.choice(range(len(means_address)), n - 3)]))

                    while len(np.unique(centroids)) < n:
                        centroids = np.unique(centroids)
                        x = np.random.rand() * np.mean(centroids[:, 0])
                        y = np.random.rand() * np.mean(centroids[:, 1])
                        centroids = np.vstack((centroids, [x, y]))

                c_ret, c_label, c_center = kmeans(means_address, k = n, \
                                                  centroids = centroids, steps = 1000)

                if c_ret < ret:
                    ret = c_ret
                    label = c_label
                    center = c_center

                print ret

            rets.append(ret)

#         plt.plot(rets)
#         plt.show()

        # Now separate the data, Note the flatten()
        # Plot the data
#         plt.scatter(means_address[:, 0], means_address[:, 1])
#
#         plt.scatter(center[:, 0], center[:, 1], s = 80, c = 'y', marker = 's')
#         plt.xlabel('Height'), plt.ylabel('Weight')
#         plt.show()


        for index_label in range(len(labels)):
            data[np.where(data == labels[index_label])] = "A" + str(label[index_label])

        labels = np.unique(data)
        data = self._binarize_feature(data, labels)

        del means_address
        del splitted_address
        del address_data
        del rets
        del center
        del label
        del c_label
        del c_center
        del centroids

        return labels, data


    def _is_int(self, s):
        '''
        Check if string is integer.
        '''

        try:
            int(s)
            return True
        except ValueError:
            return False


    def save(self, path_data, path_data_labels, path_y, path_y_labels):
        '''
        Save parsed data.
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
#     path = 'resources/crimes_testing_samples.csv'
#     path_data = 'resources/samples_data'
#     path_data_labels = 'resources/samples_data_labels'
#     path_y = 'resources/samples_y'
#     path_y_labels = 'resources/samples_y_labels'

    path = 'resources/crimes.csv'
    path_data = 'resources/data.bin'
    path_data_labels = 'resources/data_labels'
    path_y = 'resources/y'
    path_y_labels = 'resources/y_labels'
    # path = 'resources/crimes_testing_samples.csv'

    crime_data = CrimeData(path)

    for i in range(len(crime_data.data[0])):
        print crime_data.data_labels[i], type(crime_data.data[0][i]), crime_data.data[0][i]

    crime_data.save(path_data, path_data_labels, path_y, path_y_labels)
