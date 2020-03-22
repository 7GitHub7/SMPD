# import numpy as np # tablice wielowymiarowe
import numpy as np

import pandas as pd  # analiza danych tabularycznych/json
import matplotlib as mpl  # wizualizacja danych, jakieś wykresy itp
import matplotlib.pyplot as mpl

# scikit -AI

df = pd.read_csv("leaf.csv", header=None)
# print(df)

# sqrt[(x2-x1)^2 + (y2 - y1)^2]


# dzielenie na dwa zbiory, wykorzystanie gotowej funkcji
from sklearn.model_selection import train_test_split

train, test = train_test_split(df.copy(), test_size=0.2, random_state=123)

# print("TRAIN")
# print(train)
# print("TEST")
# print(test)

# wybranie klas 3 i 5
data = df[(df[0] == 3) | (df[0] == 5)]
# data= df[df[0] == 5]
# print(data)

# ponowne splitowanie danych po wybraniu klas 3 i 5
train2, test2 = train_test_split(data.copy(), test_size=0.2, random_state=111)

# print("TRAIN1")
# print(train2)
# print("TEST")
# print(test2)
# print(test2[0][0])
# print(type(train2))


mpl.scatter(x=train2[1], y=train2[6])
mpl.show()
# train2.loc[1, 1]

# mpl.scatter(x = , y= )

# wyświetlic kolorami
