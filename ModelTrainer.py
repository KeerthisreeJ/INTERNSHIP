from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    :param X_train: Training features
    :param y_train: Training targets
    :return: Trained Linear Regression model
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    
    :param X_train: Training features
    :param y_train: Training targets
    :return: Trained Random Forest Regressor model
    """
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.
    
    :param model: Trained model
    :param X_test: Test features
    :param y_test: Test targets
    :return: Mean Squared Error and R² score of the model on the test data
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2
