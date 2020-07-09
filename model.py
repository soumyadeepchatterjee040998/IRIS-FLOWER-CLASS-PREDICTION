import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import pickle

file = load_iris()
x_df = pd.DataFrame(file.data,columns=file.feature_names)
y_df = pd.DataFrame(file.target,columns=['species'])


clf = DecisionTreeClassifier()
model = clf.fit(x_df,y_df)


file = open("model.pkl","wb")
pickle.dump(model,file)
file.close()