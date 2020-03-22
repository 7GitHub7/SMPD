import pandas as pd  # analysis of tabular/json data
from operator import itemgetter  # get each rows from pandas data type
from numpy import math  # root
from sklearn.model_selection import train_test_split  # dividing data into groups


class Knn:
    """
    Implementation of knn algorithm

    ...

    Attributes
    ----------
    k = 3  : int
        k parameter in algorithm
    feature_first : int
        first leaf feature
    feature_second : int
        second leaf feature
    first_class : float
        leaf class from instruction
    second_class : float
        leaf class from instruction
    new_types_list : list
        list of list that contain: distance between test and train point, class of predict class
    true_counter : int
        count positive predictions

    Methods
    -------
    prepare_data(sound=None)
        load and split data
    calculate_euclides_distance(sound=None)
        calculate distance between points
    prepare_data(sound=None)
        calculate distance between points
    calculate_euclides_distance(sound=None)
        calculate distance between points
    prepare_data(sound=None)
        calculate distance between points
    calculate_euclides_distance(sound=None)
        calculate distance between points
    """

    k = 3
    feature_first = 2
    feature_second = 6
    first_class = 3.0
    second_class = 5.0
    new_types_list = []
    true_counter = 0
    path_to_data_set = ""

    def __init__(self, k=3, feature_first=2, feature_second=9, first_class=3.0, second_class=5.0, path_to_data_set="leaf.csv"):
        self.k = k
        self.feature_first = feature_first
        self.feature_second = feature_second
        self.first_class = first_class
        self.second_class = second_class
        self.path_to_data_set = path_to_data_set

    def _prepare_data(self, path_to_csv):
        """import data"""
        df = pd.read_csv(path_to_csv, header=None)

        """choose class 3 i 5"""
        return df[(df[0] == 3) | (df[0] == 5)]

    def _calculate_euclides_distance(self, traning_element, test_element, feature_first, feature_second):
        return math.sqrt((traning_element[feature_first] - test_element[feature_second]) ** 2 + (
                traning_element[feature_first] - test_element[feature_second]) ** 2)

    def _get_k_nearest_neighbours(self, k_nearest_neighbours, first_class, k):
        counter_type_first = 0
        counter_type_second = 0
        for i in range(k):
            nn_leaf_class = k_nearest_neighbours[0]
            if nn_leaf_class[0] == first_class:
                counter_type_first += 1
            else:
                counter_type_second += 1
        return [counter_type_first, counter_type_second]

    def _predict_class_of_test_element(self, first_class, second_class, k_nearest_neighbours, k):
        counter_list = self._get_k_nearest_neighbours(k_nearest_neighbours,first_class, k)
        counter_type_first = counter_list[0]
        counter_type_second = counter_list[1]
        if counter_type_second > counter_type_first:
            checked_case_class = second_class
        else:
            checked_case_class = first_class
        return checked_case_class

    def check_algorithm_efficiency(self):
        data = self._prepare_data(self.path_to_data_set)

        train2, test2 = train_test_split(data.copy(), test_size=0.2, random_state=111)
        """
              Split data into two groups names test and train
        
               Parameters
               ----------
               test_size : float
                   train size in percent  
               random_state : random parameter(seed)     
        """

        for z, a in test2.iterrows():
            """loop iterating in test elements"""
            for o, l in train2.iterrows():
                """loop iterating in training elements"""

                dist = self._calculate_euclides_distance(traning_element=l, test_element=a,
                                                         feature_first=self.feature_first,
                                                         feature_second=self.feature_second)
                self.new_types_list.append([l[0], dist])

            k_nearest_neighbours = sorted(self.new_types_list, key=itemgetter(1))
            checked_case_class = self._predict_class_of_test_element(self.first_class, self.second_class, k_nearest_neighbours, self.k)

            if checked_case_class == a[0]:
                self.true_counter += 1

        print(self.true_counter)
        print(test2)
        print("Wielkosc testowego zbioru: " + str(len(test2)))
        result_in_percent = (self.true_counter / len(test2)) * 100
        print("Skuteczność: ")
        print(result_in_percent)


# usage
knn = Knn()
knn.check_algorithm_efficiency()



