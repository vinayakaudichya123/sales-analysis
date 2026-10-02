import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import rfft2

df = pd.read_csv(r"C:\Users\pc\Downloads\sales_data.csv")

print(df.head())
print(df.tail())
print(df.dropna())
print(df.describe())
print(df.isnull().sum())
print(df.columns)


print(df.fillna(df['Rating'].mean()))

print(df.duplicated().sum())
print(df.dtypes)
print(df.drop_duplicates())

df['Order_Date'] = pd.to_datetime(df['Order_Date'])
print(df['Order_Date'])

print(df['Order_Date'].dt.date)


# df['Order_ID'] = df['Order_Date'].astype(str)  this for converting numerical value into string format

print(df.select_dtypes(include="object").columns)

print(df['Customer_ID'].unique())
print(df['Customer_Name'].unique())
print(df['Gender'].unique())
print(df['City'].unique())
print(df['State'].unique())
print(df['Category'].unique())

df['Payment_Method'] = df['Payment_Method'].str.title()
print(df['Payment_Method'])

print(df.select_dtypes(include="number").columns)

sns.boxplot(x=df['Sales'])
plt.show()

# numeric_cols = df.select_dtypes(include="number").columns
#
# for col in numeric_cols:
#     sns.boxplot(x=df['Sales'])
#     plt.show()


Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)

IQR = Q3 -Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("lower",lower)
print("upper",upper)


outliers = df[(df['Sales']<lower) | (df['Sales']>upper)]

print(outliers)


# Univariate analysis : ek column check karna

print("length of outliers",len(outliers))

print("Category")
print(df['Category'].value_counts())

print("Payment Method")
print(df['Payment_Method'].value_counts())

print("Order_Status ")
print(df['Order_Status'].value_counts())

print("Gender")
print(df['Gender'].value_counts())

print("sales stastical")
print(df['Sales'].describe())

# bivaritae analysis = 2 columns ke bich relation



print("Category vs Sales:")
print(df.groupby("Category")["Sales"].sum())

print("\nCity vs Sales:")
print(df.groupby("City")["Sales"].sum())

print("\nPayment Method vs Sales:")
print(df.groupby("Payment_Method")["Sales"].sum())

print("\nOrder Status vs Sales:")
print(df.groupby("Order_Status")["Sales"].sum())

print("\nQuantity vs Sales:")
print(df.groupby("Quantity")["Sales"].sum())

# Step 10.3 — Multivariate Analysis karte hain.
#
# Yahan hum 3 ya usse zyada columns ko ek saath analyze karenge.


# Step 10.3 - Multivariate Analysis

print("Category + Payment Method + Sales:")
print(df.groupby(["Category", "Payment_Method"])["Sales"].sum())

print("\nCategory + Gender + Sales:")
print(df.groupby(["Category", "Gender"])["Sales"].sum())

print("\nCity + Category + Sales:")
print(df.groupby(["City", "Category"])["Sales"].sum())

print("\nCorrelation:")
print(df.corr(numeric_only=True))


#visulize Category

plt.figure(figsize=(8,5))

sns.countplot(x="Category", data=df)

plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Orders")

plt.show()

plt.figure(figsize=(8,5))

#visulize order_status
sns.countplot(x="Order_Status",data=df)
plt.title("Order Status Distribution")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")

plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="Payment_Method",data=df)

plt.title("Payment Method Distribution")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")

plt.show()


plt.figure(figsize=(8,5))

sns.histplot(df["Sales"], bins=20)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()

# Quantity and sales

plt.figure(figsize=(8,5))

sns.scatterplot(x="Quantity", y="Sales", data=df)

plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")

plt.show()

 # unit price and Sales
plt.figure(figsize=(8,5))

sns.scatterplot(x="Unit_Price", y="Sales", data=df)

plt.title("Unit Price vs Sales")
plt.xlabel("Unit Price")
plt.ylabel("Sales")

plt.show()

# category and sales
plt.figure(figsize=(8,5))

sns.barplot(x="Category", y="Sales", data=df, estimator="sum")

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.show()

# Correlation Heatmap

plt.figure(figsize=(8,5))

sns.heatmap(df.corr(numeric_only=True),annot=True)

plt.title("Correlation Heatmap")

plt.show()

plt.figure(figsize=(8,5))

# sales by order status
#barplot isliye use kiya kyunki hum ek categorical column ko ek numerical column ke saath compare kar rahe hain.

sns.barplot(
    x="Order_Status",
    y="Sales",
    data=df,
    estimator="sum"
)

plt.title("Total Sales by Order Status")
plt.xlabel("Order Status")
plt.ylabel("Total Sales")

plt.show()


# top category by sales

category_sales = df.groupby("Category")["Sales"].sum()
print(category_sales)
print("\nTop Category:")
print(category_sales.idxmax())

city_sales = df.groupby("City")["Sales"].sum()
print(city_sales)

print("\nTop City:")
print(city_sales.idxmax())


payment_counts = df["Payment_Method"].value_counts()
print(payment_counts)

print("\nTop payment:")
print(payment_counts.idxmax())

status = df["Order_Status"].value_counts()
print(status)

print("\nTop status:")
print(status.idxmax())


print("mean :",df["Sales"].mean())
print("max :",df["Sales"].max())
print("min :",df["Sales"].min())


# feature engnineering

# Date se features banana
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Day"] = df["Order_Date"].dt.day
df["DayOfWeek"] = df["Order_Date"].dt.dayofweek

# Weekend check
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

# Check
print(df[["Order_Date", "Year", "Month", "Day", "DayOfWeek", "IsWeekend"]].head())


# ML portion
df = df.dropna()
print(df)

X = df[
    [
        "Age",
        "Quantity",
        "Unit_Price",
        "Discount",
        "Category",
        "Payment_Method",
        "City",
        "State",
        "Year",
        "Month",
        "DayOfWeek",
        "IsWeekend"
    ]
]


y=df["Sales"]

print(X.isnull().sum())

print(X.head(10))
print(y.head(10))


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = [
    "Category",
    "Payment_Method",
    "City",
    "State"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    ],
    remainder="passthrough"

)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# model training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE",mae)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print("R2",r2)


from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=42)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

tree_r2 = r2_score(y_test, tree_pred)

print("Decision Tree R²:", tree_r2)


# random forest

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# rf_model.fit(X_train,y_train)
print(rf_model.fit(X_train,y_train))


rf_pred = rf_model.predict(X_test)
print(rf_pred)


from sklearn.metrics import r2_score
rf_r2 = r2_score(y_test,rf_pred)
print("rf_r2",rf_r2)


print("Linear Regression : ",r2)
print("Decision Tree Regressor : ",tree_r2)
print("Random Forest Regressor : ",rf_r2)

feature_names = preprocessor.get_feature_names_out()

importances = rf_model.feature_importances_

feature_importances = pd.Series(
    importances,
    index=feature_names

).sort_values(ascending=False)

print(feature_importances)

plt.figure(figsize=(10,6))

feature_importances.head(10).plot(kind="bar")

plt.title("Top 10 Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()


new_data = pd.DataFrame({
    "Age": [25],
    "Quantity": [2],
    "Unit_Price": [1000],
    "Discount": [0.10],
    "Category": ["Electronics"],
    "Payment_Method": ["UPI"],
    "City": ["Jaipur"],
    "State": ["Rajasthan"],
    "Year": [2026],
    "Month": [9],
    "DayOfWeek": [2],
    "IsWeekend": [0]
})


new_data = preprocessor.transform(new_data)

prediction = rf_model.predict(new_data)
print("prediction",prediction)


import joblib

joblib.dump(
    (preprocessor,rf_model),
    "sales_model.pkl"
)
print("model saved")

preprocessor,rf_model = joblib.load("sales_model.pkl")
print("model loaded")