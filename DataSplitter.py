from sklearn.model_selection import train_test_split

def split_initial_data(data):
    """
    Split the initial data into training, validation, and test sets.
    
    :param data: Pandas DataFrame with the input data
    :return: Splitted datasets for CO, CO2, and RATIO_CO_CO2
    """
    X = data.drop(['CO', 'CO2', 'RATIO_CO_CO2'], axis=1)
    y_CO = data['CO']
    y_CO2 = data['CO2']
    y_ratio = data['RATIO_CO_CO2']

    X_train, X_temp, y_train_CO, y_temp_CO = train_test_split(X, y_CO, test_size=0.4, random_state=42)
    X_val, X_test, y_val_CO, y_test_CO = train_test_split(X_temp, y_temp_CO, test_size=0.5, random_state=42)
    
    y_train_CO2, y_temp_CO2 = train_test_split(y_CO2, test_size=0.4, random_state=42)
    y_val_CO2, y_test_CO2 = train_test_split(y_temp_CO2, test_size=0.5, random_state=42)
    
    y_train_ratio, y_temp_ratio = train_test_split(y_ratio, test_size=0.4, random_state=42)
    y_val_ratio, y_test_ratio = train_test_split(y_temp_ratio, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train_CO, y_val_CO, y_test_CO, y_train_CO2, y_val_CO2, y_test_CO2, y_train_ratio, y_val_ratio, y_test_ratio

def create_shifted_targets(data):
    """
    Create shifted targets for future predictions (1hr, 2hr, 3hr, 4hr).
    
    :param data: Pandas DataFrame with the input data
    :return: Pandas DataFrame with shifted target columns
    """
    for i in range(1, 5):
        data[f'RATIO_CO_CO2_shift_{i}hr'] = data['RATIO_CO_CO2'].shift(-i)
    data = data.dropna()
    return data

def split_shifted_data(shifted_data):
    """
    Split the shifted data into training, validation, and test sets for each shifted target.
    
    :param shifted_data: Pandas DataFrame with shifted target columns
    :return: Splitted datasets for each shifted target (1hr, 2hr, 3hr, 4hr)
    """
    X = shifted_data.drop([f'RATIO_CO_CO2_shift_{i}hr' for i in range(1, 5)], axis=1)
    
    X_train, X_temp = train_test_split(X, test_size=0.4, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

    y_train_1hr, y_temp_1hr = train_test_split(shifted_data['RATIO_CO_CO2_shift_1hr'], test_size=0.4, random_state=42)
    y_val_1hr, y_test_1hr = train_test_split(y_temp_1hr, test_size=0.5, random_state=42)
    
    y_train_2hr, y_temp_2hr = train_test_split(shifted_data['RATIO_CO_CO2_shift_2hr'], test_size=0.4, random_state=42)
    y_val_2hr, y_test_2hr = train_test_split(y_temp_2hr, test_size=0.5, random_state=42)
    
    y_train_3hr, y_temp_3hr = train_test_split(shifted_data['RATIO_CO_CO2_shift_3hr'], test_size=0.4, random_state=42)
    y_val_3hr, y_test_3hr = train_test_split(y_temp_3hr, test_size=0.5, random_state=42)
    
    y_train_4hr, y_temp_4hr = train_test_split(shifted_data['RATIO_CO_CO2_shift_4hr'], test_size=0.4, random_state=42)
    y_val_4hr, y_test_4hr = train_test_split(y_temp_4hr, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train_1hr, y_val_1hr, y_test_1hr, y_train_2hr, y_val_2hr, y_test_2hr, y_train_3hr, y_val_3hr, y_test_3hr, y_train_4hr, y_val_4hr, y_test_4hr
