import pandas as pd 

df = pd.read_csv('sample_data.csv')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df_lable = df.copy()

df_lable.columns = df_lable.columns.str.strip()
# agar column ky nam mai  space hoga tu sapce deni hogi warna column get ni #  hoga lakin agar strip laga degay tu space remove ho jayega aur column get # ho jayega

le = LabelEncoder()

# label encoding
df_lable['gender_encoded'] = le.fit_transform(df_lable['Gender'])
df_lable['Passed_encoded'] = le.fit_transform(df_lable['Passed'])

# print('\n lable encoded data')
# print(df_lable)

# one hot encoding
df_one_hot = pd.get_dummies(df_lable, columns=['City'])
# df_one_hot = pd.get_dummies(df_lable, columns=['City'], drop_first=True)
# drop_first=True ka matlab hai ke ek column drop kar do
# Agar mujhe pata hai:
# Karachi = 0
# Lahore = 1
# To automatically Islamabad = 0 hoga
# 👉 matlab ek column extra / redundant hai

# ab in true false ko  0 aur 1 main convert karna ky 2 metho hain 
# 1st method
df_one_hot = pd.get_dummies(df_lable, columns=['City'],  dtype=int)

# 2nd method
# df_one_hot = pd.get_dummies(df_lable, columns=['City'])
# for col in df_one_hot.columns:
#     if df_one_hot[col].dtype == 'bool':
#         df_one_hot[col] = df_one_hot[col].astype(int)
# print('\n one hot encoded data')
print(df_one_hot)



ohe = OneHotEncoder(sparse_output=False)
# # sparse=False ka matlab hai ke output ko dense array mai convert kar do
encoded = ohe.fit_transform(df_lable[['City']])
# # raw matrix data 
# print(encoded)
# # encoded data ko dataframe mai convert karna
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['City']))
# print(encoded_df)

# Ordinal Encoding
# Ordinal Encoding mein har category ko ek integer (0,1,2,3...) assign kiya jata hai.
# Ye tab use hota hai jab categories ka natural order ho.
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    'Education': ['Bachelor', 'Master', 'PhD', 'High School', 'Master']
})

encoder = OrdinalEncoder(categories=[
    ['High School', 'Bachelor', 'Master', 'PhD']
])

df['Education_Encoded'] = encoder.fit_transform(df[['Education']])

print(df)

# Frequency Encoding
# Har category ko uski dataset mein aane ki frequency (count) se replace kiya jata hai.
# notebook main hain 

# Binary Encoding
# Binary Encoding mein pehle categories ko numbers diye jaate hain, phir un numbers ko binary form mein convert kiya jata hai.
# notebook main hain