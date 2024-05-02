import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv('my_timestamps.csv', index_col='Timestamps', parse_dates=True)

features = ['Bytes_Sent (TCP)', 'Bytes_Sent (UDP)', 'Bytes_Sent (Other)',
            'Bytes_Received (TCP)', 'Bytes_Received (UDP)', 'Bytes_Received (Other)',
            'Packets_Sent (TCP)', 'Packets_Sent (UDP)', 'Packets_Sent (Other)',
            'Packets_Received (TCP)', 'Packets_Received (UDP)', 'Packets_Received (Other)']

from scipy.ndimage import gaussian_filter1d

def get_model_evaluation(df,target):
    print('Evaluating the model for : ',end='')
    print(target)
    print('-'*50)
    smoothed_data = gaussian_filter1d(df[features], sigma=1)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(smoothed_data if 'smoothed_data' in locals() else df[features], df[target], test_size=0.2, random_state=42)
    # Create the HistGradientBoostingRegressor model
    model = HistGradientBoostingRegressor(random_state=42)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    # Evaluate model performance (e.g., calculate mean squared error)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2}")
    from sklearn.metrics import explained_variance_score
    explained_variance = explained_variance_score(y_test, y_pred)
    print(f"Explained Variance Score: {explained_variance}")
    print('-'*50)
    print('Using the trained model to make predictions on new data......')
    y_pred = model.predict(X_test)
    print(y_pred)


for each_feature in features:
    get_model_evaluation(df,each_feature)

