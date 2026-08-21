import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Fake Data taiyar karna (Bank Customer Data)
# Features: Income (Monthly), Credit_Score, Existing_Loan (0 = No, 1 = Yes)
# Target: Loan_Approved (0 = No, 1 = Yes)

data = {
    "Income": [
        25000,
        35000,
        60000,
        80000,
        20000,
        55000,
        90000,
        40000,
        75000,
        120000,
    ],
    "Credit_Score": [600, 650, 720, 780, 580, 710, 800, 630, 740, 820],
    "Existing_Loan": [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    "Loan_Approved": [
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
    ],  # 0 = Reject, 1 = Approve
}

# Data ko DataFrame (table) mein convert karna
df = pd.DataFrame(data)

# Features (X) aur Target (y) ko alag karna
X = df[["Income", "Credit_Score", "Existing_Loan"]]
y = df["Loan_Approved"]

# Data ko Train aur Test sets mein split karna (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Decision Tree Classifier Model banana aur train karna
# 'criterion="entropy"' ya "gini" yeh decide karta hai ki tree split kaise hogi
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("--- Model Training Complete! ---")

# 3. Naye Customer ke liye Prediction (Faisla) karna
# Maan lete hain ek naya banda aaya jiski Income=65000, Credit Score=730 hai, aur koi purana loan nahi hai (0)
Income = input("Enter Income (Monthly): ")
Credit_Score = input("Enter Credit Score: ")
Existing_Loan = input("Do you have an existing loan? (0 = No, 1 = Yes): ")
new_customer = [[float(Income), float(Credit_Score), int(Existing_Loan)]]
prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nResult: Loan APPROVED!")
else:
    print("\nResult: Loan REJECTED!")

# 4. Tree ko Visualize karna (Dekhna ki AI ne kaise socha)
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True,
)
plt.title("Decision Tree Kaise Faisle Le Raha Hai:")
plt.show()



# [ gini/information-gain ya decide karta hain root node kya hoga ]

#                           Root Node (tree)
#                              /   \
#                     branch  /     \  branch
#                            /       \
#                [ decision node ]     [ decision node ] (sub_tree)
#                                             /   \
#                                     branch /     \  branch
#                                           /       \
#  (sub_tree)[ sub_decision_node/internal node ]    [ sub_decision_node/internalnode ]
#                              /   \
#                     branch  /     \ branch
#                            /       \
#                      [ leaf ]      [ leaf ]  leaf wahi hoga jaha decision ka outcome aye ga


# jb model overfit ho jate hain tu hum kuch sub_tree ko hata detay hain tu is process ko hum pruning bolte hain 

# Root Node:
# root node wo node hoga jis impurity/gini_impurity zeyda hogi or information gain kam hoga like koi asa sawal hain jis base pr hum direct result ni e sakte tu woh root sawal hoga mean root node jis ki impiurity zeyad hain or information gain kam hain jasy jasy sawal pochy gay mean tree bane gay tu information gain zeyad hoga or impurity kam hogi 

# Entropy:
# Entropy batati hai ke data mein kitni uncertainty (confusion) hai.
# Entropy=−∑pi​log2​(pi​)
# p matlab os tree main se kitne sawal nikal rahe hain 
# Entropy zeyda hogi sawal utna root node kareeb higa ya rooot node he hoga 

# Gini Impurity:
# Gini = 1−∑pi2​
# Gini bhi yahi batata hai ke data kitna mixed hai, lekin formula alag use karta hai.

# Socho ek basket mein balls hain.
# 10 Red, 0 Blue → Pure → Entropy = 0, Gini = 0
# 5 Red, 🔵 5 Blue → Sab se zyada mixed → Entropy aur Gini dono high honge.

# 10 0r 0 ya 5 or 5 p ki value hain 10 hain or 10 main se 10 he red mean sb ak he class se hain entropy zero or jb b entropy zero aye tu samaj jao wo leaf node waha information gain zeyda hain or entropy or gini kam hain 

# Information Gain = Parent Entropy − Weighted Child Entropy

# Parent Entropy = Split se pehle data ki confusion.
# Child Entropy = Split ke baad har group ki confusion.
# Weighted Child Entropy = Har child group ki entropy ko us group ke size ke hisaab se weight dena.

# Weighted" ka matlab kya hai?
# Weighted ka matlab hai har child node ko uske records ki tadaad ke mutabiq importance dena.
# Agar ek child mein 8 records hain aur doosre mein 2, to 8 wala group zyada weight lega.

# Maan lo total 10 students hain.

# Split karne ke baad:
# Left Child
# 8 students
# Entropy = 0.5

# Right Child
# 2 students
# Entropy = 1

# Ab Weighted Child Entropy nikalo:
# 8/10 * 0.5 + 2/10 * 1

# Step by step:
# Left weight = 8/10 = 0.8
# Right weight = 2/10 = 0.2

# Ab:
# 0.8×0.5=0.4
# 0.2×1=0.2

# Dono ko add karo:
# 0.4+0.2=0.6

# Weighted Child Entropy = 0.6

# Ab Information Gain
# Agar Parent Entropy = 0.9 thi, to:
# Information Gain= 0.9−0.6 = 0.3

# Yani is split ne 0.3 confusion kam kar di.
# Yaad rakhne ka shortcut
# Weighted Child Entropy =
# (Child 1 ka size ÷ Total size × Child 1 ki Entropy) + (Child 2 ka size ÷ Total size × #Child 2 ki Entropy) + ...












