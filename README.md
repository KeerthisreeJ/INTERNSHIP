# INTERNSHIP
The code is meant to explore and compare the effectiveness of different machine learning models in predicting important process variables in blast furnace operations. The use of time-shifted data helps in understanding how well these models can predict future states of the system based on historical data.

This code is designed to predict the CO/CO2 ratio in blast furnace operations using machine learning models. It involves several key steps:

Data Loading:
The data from an Excel file is loaded using a custom function load_data. The data contains various sensor readings and parameters from blast furnace operations.

Initial Data Splitting:
The data is initially split into training, validation, and test sets for different target variables, including CO, CO2, and their ratio.

Shifted Target Creation:
To account for time delays in the process, the data is further processed to create shifted targets (1-hour, 2-hour, 3-hour, and 4-hour shifts). This helps in predicting future values based on past data.

Model Training and Evaluation:
Two machine learning models are used: Linear Regression and Random Forest. These models are trained on the shifted data and evaluated using Mean Squared Error (MSE) and R2 score.
The training and evaluation are done separately for each time shift (1-hour, 2-hour, 3-hour, and 4-hour).

Visualization:
The results (MSE and R2 scores) are visualized using bar and line graphs. These graphs compare the performance of the Linear Regression and Random Forest models across different time shifts.
