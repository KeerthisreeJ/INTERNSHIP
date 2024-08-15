import pandas as pd

def load_data(file_path):
    """
    Load data from an Excel file.
    
    :param file_path: Path to the Excel file
    :return: Pandas DataFrame with the loaded data
    """
    data = pd.read_excel(file_path)
    return data
