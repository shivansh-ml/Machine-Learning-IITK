from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import datasets
import matplotlib.pyplot as plt


#The data pertains to the houses found in a given California district 
#and some summary stats about them based on the 1990 census data.
#Total number of points = 20640 
#The 8 features are

#MedInc
#HouseAge
#AveRooms
#AveBedrms
#Population
#AveOccup
#Latitude
#Longitude

# The target is
#MedHouseVal


california = datasets.fetch_california_housing()
print(california.feature_names)
print(california.target_names)
X = california.data
y = california.target 

plt.figure()
plt.scatter(X[:,6], X[:,7],  c=y, cmap="jet", alpha=0.6)
plt.colorbar()
plt.title('Scatter Plot of Housing Prices') 
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid()
plt.show()

Xtrain, Xtest, ytrain, ytest =\
train_test_split(X, y, test_size=0.2, random_state=5)
reg = LinearRegression()
reg.fit(Xtrain, ytrain)

ytrainpredict = reg.predict(Xtrain)
mse_train = mean_squared_error(ytrain, ytrainpredict)
r2_train = r2_score(ytrain, ytrainpredict)

print('Train MSE =', mse_train)
print('Train R2 score =', r2_train)
print("\n")

ytestpredict = reg.predict(Xtest)
mse_test = mean_squared_error(ytest, ytestpredict)
r2_test = r2_score(ytest, ytestpredict)

print('Test MSE =', mse_test)
print('Test R2 score =', r2_test)
print("\n")

plt.figure()
plt.scatter(ytest, ytestpredict, color='green', alpha=0.6)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='green', linestyle='--')
plt.title('Actual vs Predicted Values') 
plt.xlabel('Actual Med House Val')
plt.ylabel('Predicted Med House Val')
plt.grid()
plt.show()

"""
print('Train RMSE =', mse)
print('Train R2 score =', r2)
print("\n")

y_test_predict = reg.predict(X_test)
mse = mean_squared_error(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)
print('Test RMSE =', mse)
print('Test R2 score =', r2)


plt.figure()
plt.scatter(y_test, y_test_predict, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', linestyle='--')
plt.title('Actual vs Predicted Values') 
plt.xlabel('Actual Med House Val')
plt.ylabel('Predicted Med House Val')
plt.grid()
plt.show()
"""
