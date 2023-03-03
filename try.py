import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes.csv')
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.25)

model = pickle.load(open('model.pkl', 'rb'))
y_pred = model.predict(x_test)
print(accuracy_score(y_pred,y_test))