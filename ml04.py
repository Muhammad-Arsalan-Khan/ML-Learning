# Supervised Machine Learning
# logistic_Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
X = [[3], [4], [5], [6], [7]]
y = [0, 0, 0, 1, 1]


model.fit(X, y)

status = float(input('how many hours you studies '))    
predicted_status = model.predict([[status]])
if predicted_status[0] == 1:
    print(f"you are likely pass {predicted_status[0]}")
else:
    print(f"you are likely fail {predicted_status[0]}")
