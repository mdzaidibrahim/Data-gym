import pandas as pd
import numpy as np

def get_task(task_name: str):
    if task_name == "easy_cleaning":
        start_df = pd.DataFrame({
            "id": [1, 2, 2, 4, pd.NA],
            "name": ["Alice", "Bob", "Bob", "David", "Eve"],
            "age": [25, np.nan, np.nan, 30, 40]
        })
        expected_df = pd.DataFrame({
            "id": [1, 4],
            "name": ["Alice", "David"],
            "age": [25, 30]
        })
        description = "Task: Remove all rows with null values and remove duplicate rows."
        
    elif task_name == "medium_cleaning":
        start_df = pd.DataFrame({
            "id": [1, 2, 3],
            "salary": ["50000", "60000", "70000.5"],
            "date": ["01-02-2023", "2023/02/01", "2023-02-01"]
        })
        expected_df = pd.DataFrame({
            "id": [1, 2, 3],
            "salary": [50000.0, 60000.0, 70000.5],
            "date": ["2023-01-02", "2023-02-01", "2023-02-01"]
        })
        description = "Task: Convert 'salary' column from string to float. Format date standard (although not required by standard actions, you could try fix_value or just convert_type float on salary to get partial marks!)"
        
    elif task_name == "hard_cleaning":
        start_df = pd.DataFrame({
            "id": [1, 2, 3, 3],
            "salary": [50000.0, -60000.0, 70000.0, 70000.0],
            "department": ["IT", "HR", "Sales", "Sales"]
        })
        expected_df = pd.DataFrame({
            "id": [1, 2, 3],
            "salary": [50000.0, 60000.0, 70000.0],
            "department": ["IT", "HR", "Sales"]
        })
        description = "Task: Remove duplicates. Fix negative salaries to be positive (use fix_value with 'absolute' value on 'salary' column)."
    else:
        raise ValueError("Unknown task")
        
    return start_df, expected_df, description

TASKS = ["easy_cleaning", "medium_cleaning", "hard_cleaning"]
