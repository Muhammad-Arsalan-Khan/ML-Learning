# Feature scalling
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

data = {
    "studyHours" : [2, 4, 6, 8, 10, 12, 14, 15, 16, 20],
    "TestScore" : [50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
}

df = pd.DataFrame(data)

StandardScaled = StandardScaler().fit_transform(df)
print("Standerd scaler output")
print(pd.DataFrame(StandardScaled, columns=['studyHours', 'TestScore']))

# StandardScaler formula 
# z = X - U       X = actual value , U = mean of value, stdCol stnader deviation of column 
#     stdCol

MinMaxScaled = MinMaxScaler().fit_transform(df)
print("MinMax scaler output")
print(pd.DataFrame(MinMaxScaled, columns=['studyHours', 'TestScore']))

# MinMaxScaler fromula
# scale =    X   - X(min)
#         X(min) - X(max)

# hum datafram input dena hain warna ho serios data samjay ga yaha data split hoga
X = df[['studyHours']]
y = df[['TestScore']]

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)

# xtrain or ytrain se model train hoga or xtest or ytest se check hoga train main 80% data hoga or test main 20% data hoga random state 42 is leay takay rh bar same rwsult aye 

print("\ntraining ky leya ya data use hoga")
print(f'{X_train},\n{y_train}')

print("\ntesting ky leya ya data use hoga")
print(f'{X_test},\n{y_test}')

