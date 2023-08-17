#These are the necessary import statements for the libraries you'll be using.
#NumPy is used for numerical operations, pandas for data manipulation, Matplotlib for data visualization, and scikit-learn for machine learning functionalities.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#conditions to choose the stock you want to see.
print("choose a stock to predict\n")
choose=input(" 1.Apple\n 2.Tesla\n 3.Amazon\n :")
if(choose=="1"):
    #saving the path to the data file in this variable
    csv_file_path = '/Users/oranefroni/Desktop/AAPL.csv'  
elif(choose=="2"):
    #saving the path to the data file in this variable
    csv_file_path = '/Users/oranefroni/Desktop/TSLA.csv'
elif(choose=="3"):
    #saving the path to the data file in this variable
    csv_file_path = '/Users/oranefroni/Desktop/AMZN.csv'
else:
    print("Invalid choice")
    exit()

#This line reads the information from the file and saving it into the 'data' container  
data = pd.read_csv(csv_file_path)

#convert the Date column to datetime format using the function: pd.to_datetime
data['Date'] = pd.to_datetime(data['Date'])

#sorts the data base on the date
data = data.sort_values('Date')

#This line calculates the number of days between each date in the dataset and the earliest date,
#creating a new column called 'Days' in the data.
#This conversion helps represent dates as numerical values, which can be used for analysis or modeling.
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

#the x variable saves the column of the days between each date, the y variable saves the column of the price in each window
X = data[['Days']]
y = data['Close']

#This line splits the data into training and testing sets. It assigns 80% of the data to the training and 20% to the testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating the model of the linear regression
model = LinearRegression()
#train the model base on the train data
model.fit(X_train, y_train)

#asking the model to make a prediction using the data we saves to test and saves its prediction in a varible calls: y_pred
y_pred = model.predict(X_test)

#this line calculate the mse (to see how close to the reality our prediction) using the prediction data - y_pred and the real data using - y_test
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#This code segment creates a scatter plot of the actual prices (X_test and y_test) in blue points and overlays the predicted prices (X_test and y_pred) as a red line.
#It also adds labels, a title, and a legend to the plot. Finally, plt.show() displays the plot on the screen.
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Stock Price Prediction using Linear Regression')
plt.legend()
plt.show()
