import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Generate dataset
np.random.seed(42)
data = {
    "Date": pd.date_range(start="2025-01-01", periods=100, freq="D"),
    "Product": np.random.choice(["Laptop", "Mobile", "Tablet", "Headphones"], 100),
    "Quantity": np.random.randint(1, 5, size=100),
    "Price_per_unit": np.random.randint(5000, 50000, size=100),
}

df = pd.DataFrame(data)
df["Total_Sales"] = df["Quantity"] * df["Price_per_unit"]

# Calculate total sales per product
sales_per_product = df.groupby("Product")["Total_Sales"].sum()

# Plot total sales per product
plt.figure(figsize=(8, 5))
sns.barplot(x=sales_per_product.index.astype(str), y=sales_per_product.values)
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Total Sales per Product")
plt.xticks(rotation=45)
plt.show()

print("Total Sales per Product:")
print(sales_per_product)

# Convert Date to numeric format
df["Day_Index"] = np.arange(len(df))

# Prepare data for Linear Regression
X = df[["Day_Index"]].values
y = df["Total_Sales"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict next 90 days
future_days = np.arange(len(df), len(df) + 90).reshape(-1, 1)
future_sales = model.predict(future_days)

# Plot predictions
plt.figure(figsize=(10, 5))
sns.lineplot(x=df["Day_Index"], y=df["Total_Sales"], label="Actual Sales")
sns.lineplot(x=future_days.flatten(), y=future_sales, label="Predicted Sales", linestyle="dashed")

plt.xlabel("Days")
plt.ylabel("Total Sales")
plt.title("Sales Prediction for Next 90 Days")
plt.legend()
plt.show()

# Save the DataFrame to a CSV file
df.to_csv("sales_data.csv", index=False)

# Save model details for future reference
model_info = {
    "coefficients": model.coef_,
    "intercept": model.intercept_,
}
