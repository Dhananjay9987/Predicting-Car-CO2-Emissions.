# step 1: import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#step 2: Load the dataset
data = pd.read_csv("CO2_emissions.csv")
print(data.head())

#step3: Explore the data
print(data.info())
print(data.describe())
print(data.isnull().sum())

#Step 4: Visualise
sns.scatterplot(x="ENGINESIZE", y="CO2EMISSIONS", data=data)
plt.title("Engine Size vs CO2 Emissions")
plt.show()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Step 5: Select features and target
X = data[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]]
y = data["CO2EMISSIONS"]

#Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 7: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Step 8: Make predictions
y_pred = model.predict(X_test)

#Step 9: Check accuracy
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

#Step 10: Visualise predictions
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted")
plt.show()
