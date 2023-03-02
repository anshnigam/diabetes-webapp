import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('diabetes.csv')
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
model = RandomForestClassifier()
model.fit(x,y)
pickle.dump(model, open('model.pkl','wb'))