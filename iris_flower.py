from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# read Data and fill a nan to None
df = pd.read_csv("~/myproject/Model/IRIS model/iris_data.csv")
df.fillna(None)

# set x and y
X = df.drop(columns=["species", "species_id"])
Y = df["species"]

# train and test split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=42)

# create model
for k in range(1,5):
    print("n_neighbors:",k)
    model_knn = KNeighborsClassifier(n_neighbors= k)
    model_knn.fit(train_x, train_y)

    yhat = model_knn.predict(test_x)

    print("accuracy_score:",metrics.accuracy_score(test_y, yhat))
    print("----------------------------------")

col = ["sepal_length","sepal_width","petal_length","petal_width"]
len_break = len(col)
new_x_data = []
n = 0
while True:
    try:
        value = float(input(f"Enter {col[n]}:"))
        if col[n] == "sepal_length":
            if value > df["sepal_length"].max() or value < df["sepal_length"].min():
                print("the data is more or less please Enter a normal data!")
                continue
                
        if col[n] == "sepal_width":
            if value > df["sepal_width"].max() or value < df["sepal_width"].min():
                print("the data is more or less please Enter a normal data!")
                continue

        if col[n] == "petal_length":
            if value > df["petal_length"].max() or value < df["petal_length"].min():
                print("the data is more or less please Enter a normal data!")
                continue
            
        if col[n] == "petal_width":
            if value > df["petal_width"].max() or value < df["petal_width"].min():
                print("the data is more or less please Enter a normal data!")
                continue
        
        new_x_data.append(value)
    except ValueError as e:
        print("pls Enter a float or int data not string!")
        continue
        
    if n == 3:
        break
    n+=1

new_x_data_array = np.array([new_x_data])
df_new_x_data = pd.DataFrame([{
    "sepal_length":new_x_data_array[0][0],
    "sepal_width":new_x_data_array[0][1],
    "petal_length":new_x_data_array[0][2],
    "petal_width":new_x_data_array[0][3]
}])

predication = model_knn.predict(df_new_x_data)
print("model answer:",predication[0])