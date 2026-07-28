from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.model_selection import cross_val_score


data = pd.read_csv('Salary Data.csv')


data = data.dropna()

education_map = {
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}
data["Education Level"] = data["Education Level"].map(education_map)

# print(data.head())


X = data[['Years of Experience', 'Education Level']]
y = data['Salary']

model = LinearRegression()

model.fit(X, y)
y_pred = model.predict(X)


mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# scores = cross_val_score(model, X, y, cv=5, scoring='r2')
# print(scores)
# cross vale score data ko bina split kiye hi 5 fold me divide karke model ko train aur test karta hai.

#graph
plt.figure(figsize=(7,5))
plt.scatter(data['Years of Experience'], data['Salary'], color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    data['Years of Experience'],
    data['Education Level'],
    data['Salary'],
    color='blue'
)


x_surf = np.linspace(data['Years of Experience'].min(),
                     data['Years of Experience'].max(), 20)
y_surf = np.linspace(data['Education Level'].min(),
                     data['Education Level'].max(), 20)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = model.predict(np.c_[x_surf.ravel(), y_surf.ravel()])
z_surf = z_surf.reshape(x_surf.shape)
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)


ax.set_xlabel("Experience")
ax.set_ylabel("Education Level")
ax.set_zlabel("Salary")
ax.set_title("3D Experience + Education vs Salary")

plt.show()

experience = int(input("Enter years of experience "))
education_level = int(input("Enter education level (Bachelor's= 1, Master's = 2, PhD = 3) "))

if education_level not in [1, 2, 3]:
    print("Invalid education level. Please enter 1 for Bachelor's, 2 for Master's, or 3 for PhD.")
    exit()
elif experience < 0:
    print("Invalid years of experience. Please enter a non-negative value.")
    exit()
    
input_data = pd.DataFrame({
    "Years of Experience": [experience],
    "Education Level": [education_level]
})


predicted_salary = model.predict(input_data)

if education_level == 1:
    Edu_lable = "Bachelor's"
elif education_level == 2:
    Edu_lable = "Master's"
elif education_level == 3:
    Edu_lable = "PhD"


print(f'The predicted salary for {experience} years of experience and education level {Edu_lable} is: {round(predicted_salary[0], 2)}')