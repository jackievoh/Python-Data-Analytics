import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load dataset - chose to do california housing dataset
data = pd.read_csv("california_housing.csv")

# Import median_house_value column into a separate variable
response = data['median_house_value']
features = data.drop('median_house_value', axis=1)

# Split the data into train and test parts
X_train, X_test, y_train, y_test = train_test_split(features, response, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit train data into linear regression model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate linear regression model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

# Export model
joblib.dump(model, 'linear_regression_model.pkl')

# Create scatter plot
loaded_model = joblib.load('linear_regression_model.pkl')
plt.scatter(y_test, y_pred, alpha=0.5)
y_pred = loaded_model.predict(X_test)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('California Housing Linear Regression Model')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.show()