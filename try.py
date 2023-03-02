import pickle
import pandas as pd
md = pickle.load(open('model.pkl', 'rb'))
print(md.predict([[1,0,1,26,0,0,0,1,0,1,0,1,0,3,5,30,0,1,4,6,8]]))