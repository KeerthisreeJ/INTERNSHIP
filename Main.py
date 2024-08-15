from DataLoader import load_data
from DataSplitter import split_initial_data, create_shifted_targets, split_shifted_data
from ModelTrainer import train_linear_regression, train_random_forest, evaluate_model
import matplotlib.pyplot as plt

# Load data
file_path = 'bf3_data_2022_01_07.xlsx'
data = load_data(file_path)

# Split initial data
X_train, X_val, X_test, y_train_CO, y_val_CO, y_test_CO, y_train_CO2, y_val_CO2, y_test_CO2, y_train_ratio, y_val_ratio, y_test_ratio = split_initial_data(data)

# Create shifted targets and split the data
shifted_data = create_shifted_targets(data)
X_train_1hr, X_val_1hr, X_test_1hr, y_train_1hr, y_val_1hr, y_test_1hr, \
X_train_2hr, X_val_2hr, X_test_2hr, y_train_2hr, y_val_2hr, y_test_2hr, \
X_train_3hr, X_val_3hr, X_test_3hr, y_train_3hr, y_val_3hr, y_test_3hr, \
X_train_4hr, X_val_4hr, X_test_4hr, y_train_4hr, y_val_4hr, y_test_4hr = split_shifted_data(shifted_data)

# Initialize dictionaries to store results
results = {
    '1hr': {'MSE_LR': 0, 'R2_LR': 0, 'MSE_RF': 0, 'R2_RF': 0},
    '2hr': {'MSE_LR': 0, 'R2_LR': 0, 'MSE_RF': 0, 'R2_RF': 0},
    '3hr': {'MSE_LR': 0, 'R2_LR': 0, 'MSE_RF': 0, 'R2_RF': 0},
    '4hr': {'MSE_LR': 0, 'R2_LR': 0, 'MSE_RF': 0, 'R2_RF': 0},
}

# Train and evaluate models for the 1-hour shift
model_1hr_lr = train_linear_regression(X_train_1hr, y_train_1hr)
mse_lr, r2_lr = evaluate_model(model_1hr_lr, X_test_1hr, y_test_1hr)
results['1hr']['MSE_LR'] = mse_lr
results['1hr']['R2_LR'] = r2_lr
print(f'Linear Regression - 1hr Shift: MSE={mse_lr}, R2={r2_lr}')

model_1hr_rf = train_random_forest(X_train_1hr, y_train_1hr)
mse_rf, r2_rf = evaluate_model(model_1hr_rf, X_test_1hr, y_test_1hr)
results['1hr']['MSE_RF'] = mse_rf
results['1hr']['R2_RF'] = r2_rf
print(f'Random Forest - 1hr Shift: MSE={mse_rf}, R2={r2_rf}')

# Train and evaluate models for the 2-hour shift
model_2hr_lr = train_linear_regression(X_train_2hr, y_train_2hr)
mse_lr_2hr, r2_lr_2hr = evaluate_model(model_2hr_lr, X_test_2hr, y_test_2hr)
results['2hr']['MSE_LR'] = mse_lr_2hr
results['2hr']['R2_LR'] = r2_lr_2hr
print(f'Linear Regression - 2hr Shift: MSE={mse_lr_2hr}, R2={r2_lr_2hr}')

model_2hr_rf = train_random_forest(X_train_2hr, y_train_2hr)
mse_rf_2hr, r2_rf_2hr = evaluate_model(model_2hr_rf, X_test_2hr, y_test_2hr)
results['2hr']['MSE_RF'] = mse_rf_2hr
results['2hr']['R2_RF'] = r2_rf_2hr
print(f'Random Forest - 2hr Shift: MSE={mse_rf_2hr}, R2={r2_rf_2hr}')

# Train and evaluate models for the 3-hour shift
model_3hr_lr = train_linear_regression(X_train_3hr, y_train_3hr)
mse_lr_3hr, r2_lr_3hr = evaluate_model(model_3hr_lr, X_test_3hr, y_test_3hr)
results['3hr']['MSE_LR'] = mse_lr_3hr
results['3hr']['R2_LR'] = r2_lr_3hr
print(f'Linear Regression - 3hr Shift: MSE={mse_lr_3hr}, R2={r2_lr_3hr}')

model_3hr_rf = train_random_forest(X_train_3hr, y_train_3hr)
mse_rf_3hr, r2_rf_3hr = evaluate_model(model_3hr_rf, X_test_3hr, y_test_3hr)
results['3hr']['MSE_RF'] = mse_rf_3hr
results['3hr']['R2_RF'] = r2_rf_3hr
print(f'Random Forest - 3hr Shift: MSE={mse_rf_3hr}, R2={r2_rf_3hr}')

# Train and evaluate models for the 4-hour shift
model_4hr_lr = train_linear_regression(X_train_4hr, y_train_4hr)
mse_lr_4hr, r2_lr_4hr = evaluate_model(model_4hr_lr, X_test_4hr, y_test_4hr)
results['4hr']['MSE_LR'] = mse_lr_4hr
results['4hr']['R2_LR'] = r2_lr_4hr
print(f'Linear Regression - 4hr Shift: MSE={mse_lr_4hr}, R2={r2_lr_4hr}')

model_4hr_rf = train_random_forest(X_train_4hr, y_train_4hr)
mse_rf_4hr, r2_rf_4hr = evaluate_model(model_4hr_rf, X_test_4hr, y_test_4hr)
results['4hr']['MSE_RF'] = mse_rf_4hr
results['4hr']['R2_RF'] = r2_rf_4hr
print(f'Random Forest - 4hr Shift: MSE={mse_rf_4hr}, R2={r2_rf_4hr}')

# Plot the results
shifts = ['1hr', '2hr', '3hr', '4hr']

# Bar Graph: MSE Comparison
mse_lr_values = [results[shift]['MSE_LR'] for shift in shifts]
mse_rf_values = [results[shift]['MSE_RF'] for shift in shifts]

plt.figure(figsize=(10, 5))
plt.bar(shifts, mse_lr_values, width=0.4, label='Linear Regression MSE')
plt.bar(shifts, mse_rf_values, width=0.4, label='Random Forest MSE', alpha=0.7)
plt.xlabel('Shift')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison for Different Shifts')
plt.legend()
plt.show()

# Line Graph: R2 Comparison
r2_lr_values = [results[shift]['R2_LR'] for shift in shifts]
r2_rf_values = [results[shift]['R2_RF'] for shift in shifts]

plt.figure(figsize=(10, 5))
plt.plot(shifts, r2_lr_values, marker='o', label='Linear Regression R2')
plt.plot(shifts, r2_rf_values, marker='o', label='Random Forest R2')
plt.xlabel('Shift')
plt.ylabel('R2 Score')
plt.title('R2 Score Comparison for Different Shifts')
plt.legend()
plt.show()

