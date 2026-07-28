# Supervised Machine Learning
# LinearRegression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
 
X = [[1], [2], [3], [4], [5], [6], [7]]
y = [50, 55, 65, 80, 85, 95, 110]

model.fit(X, y)

hours = float(input('how many hours you studies '))

predicted_marks = model.predict([[hours]])

print(f"you marks around {predicted_marks[0]:.2f}")