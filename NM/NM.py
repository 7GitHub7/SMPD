import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import train_test_split  # dividing data into groups

class Class:
    name = 0
    feature_list=[]
    

def _prepare_data(path_to_csv,class_list,feature_list):
        """import data"""
        df = pd.read_csv(path_to_csv, header=None)
        """choose class"""
        k = df[(df[0].isin(class_list))]
        """return class with specific features"""
        return k[feature_list]
 
data = _prepare_data(path_to_csv="leaf.csv",class_list = [3, 5],feature_list = [0, 1, 2,4])

# create list with class names, without duplicates 
class_names_list = list(dict.fromkeys(data[0].to_list()))
print(class_names_list)

# create objects for class exists in collection
# example
# class.name = 3
# class.feature_list[1, 1, 1],
            #     [1, 2, 1],
            #     [1, 3, 2],
            #     [1, 4, 3],
            #     [2, 4, 2],
            #     [2, 3, 3],
            #     [2, 2, 1],
            #     [1, 4, 2]
# ])
class_object_list = []
i = 0
for c in class_names_list:
    class_object_list.append(Class())
    class_object_list[i].name = c
    i+=1
    
for z, test_item in data.iterrows():
#    get only features from list, element zero is class name 
   buff_list = test_item[1:]
#    find class and append feature list
   for obj in class_object_list:
        if test_item[0] == obj.name:
            obj.feature_list.append(buff_list)
# cast to numpy.array
arr = np.array(class_object_list[0].feature_list) 


# for z, test_item in data.iterrows():
#     print(test_item[0])
    



# train2, test2 = train_test_split(
#     data.copy(), test_size=0.2, random_state=111)

