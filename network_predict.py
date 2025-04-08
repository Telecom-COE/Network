import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('network_traffic.csv')
# Encoding categorical variable 'Connection Type' using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Splitting the data into train and test sets
X = data_encoded.drop('Traffic Volume (GB)', axis=1)
y = data_encoded['Traffic Volume (GB)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the mod
model = LinearRegression()
model.fit(X_train, y_train)

# After model training, save the model and scaler
joblib.dump(model, 'network_predict.pkl')
joblib.dump(scaler, 'scaler.pkl')
print('Model saved!')

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Testing the Model on New Data

#new_data = np.array([[-50, 30, 0.5, 1, 0.7, 50, 500, 99.7, 20, 50, 25, 12, 2, 4]])  # Example data
#new_data_encoded = pd.get_dummies(pd.DataFrame(new_data, columns=X.columns), drop_first=True)

# Make prediction for new data
#new_traffic_volume = model.predict(new_data_encoded)
#print(f"Predicted Traffic Volume (GB): {new_traffic_volume[0]}")
