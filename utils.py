import os
import pandas as pd
# Function to format inner circle values
# def format_num_as_str(value):
#     """Format numbers as strings."""
#     if value.is_integer():
#         return str(int(value))
#     else:
#         return str(value).replace('.', '_')

def safe_divide(a, b):
    if a == 0 and b == 0:
        return 0
    return a / b

def join_as_list(var1, var2):
    """
    Join two variables as lists.
    """
    # Ensure both variables are lists (if they are strings, convert them into single-item lists)
    list1 = var1 if isinstance(var1, list) else [var1]
    list2 = var2 if isinstance(var2, list) else [var2]
    
    # Combine both lists
    return list1 + list2

## Create folder for saving if not exists
def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it does not exist.

    Args:
        folder_path: Path of the folder to create.
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")
    except OSError as e:
        print(f"Error creating folder: {e}")

## Used for cell-cell interactions
def has_consecutive_true(condition, time_threshold):
    """
    Checks if a condition holds for at least 'time_threshold' consecutive rows in a group.

    Parameters:
        group (pd.DataFrame): The subset (group) of the DataFrame.
        condition (pd.Series): A boolean Series indicating where the condition is True.
        time_threshold (int): The minimum number of consecutive True values required.

    Returns:
        bool: True if there are at least 'time_threshold' consecutive True values, otherwise False.
    """
    # Convert boolean condition to a list of 1s and 0s
    true_values = condition.astype(int).values

    # Count consecutive True values
    max_consecutive = 0
    current_consecutive = 0

    for val in true_values:
        if val:  # If True (1), increase counter
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:  # Reset counter when False (0) is encountered
            current_consecutive = 0
    
    return max_consecutive >= time_threshold

# Format plot title by adding a new line after every max_length characters
def create_plot_title(df_group, group_var=None, title_vars=None, max_line_length=30):
    """
    Create a plot title based on group variables.
    Args:
        df_group: DataFrame corresponding to the group.
        group_var: Variable(s) used for grouping.
        title_vars: Variable(s) to include in the title.
        max_line_length: Maximum length of each line in the title.
    Returns:
        Formatted plot title as a string.
    """
    # Format according to title_vars
    if group_var is None:
        return ""
    if title_vars is None:
        # print(f"Formatting title according to group_var {group_var}.")
        if isinstance(group_var, str):
            title_vars = [group_var]
        elif isinstance(group_var, list):
            title_vars = group_var
        else:
            raise Exception("group_var must be either a string or a list of strings.")    
    if isinstance(title_vars, str):
        title_vars = [title_vars]
    # else:
        # print(f"Formatting title according to title_vars {title_vars}.")
    
    title_lines = []
    for var in title_vars:
        value = df_group.iloc[0][var]
        if not pd.isna(value):
            title_lines.append(f"{var}: {value}")
    title = ", ".join(title_lines)
    title = format_plot_title(title, max_line_length=max_line_length)
    
    return title

def format_plot_title(title, max_line_length=30):
    words = title.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_line_length:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

# Function to find position columns in the DataFrame
def find_position_columns(df):
    pos_x_results = df.columns[df.columns.str.startswith('Position X')]
    if len(pos_x_results)==1:
        pos_x_var = pos_x_results[0]
    elif len(pos_x_results)==0:
        print(f"No column starting with 'Position X' found. Setting pos_x_var to None.")
        pos_x_var = None  # If no column starting with 'Position X' is found, set pos_x_var to None
    else:
        raise ValueError(f"Expected exactly one column starting with 'Position X', but found {len(pos_x_results)}: {pos_x_results.tolist()}")
    pos_y_results = df.columns[df.columns.str.startswith('Position Y')]
    if len(pos_y_results)==1:
        pos_y_var = pos_y_results[0]
    elif len(pos_y_results)==0:
        print(f"No column starting with 'Position Y' found. Setting pos_y_var to None.")
        pos_y_var = None  # If no column starting with 'Position Y' is found, set pos_y_var to None
    else:
        raise ValueError(f"Expected exactly one column starting with 'Position Y', but found {len(pos_y_results)}: {pos_y_results.tolist()}")
    pos_z_results = df.columns[df.columns.str.startswith('Position Z')]
    if len(pos_z_results)==1:
        pos_z_var = pos_z_results[0]
    elif len(pos_z_results)==0:
        print(f"No column starting with 'Position Z' found. Setting pos_z_var to None.")
        pos_z_var = None  # If no column starting with 'Position Z' is found, set pos_z_var to None
    else:
        raise ValueError(f"Expected exactly one column starting with 'Position Z', but found {len(pos_z_results)}: {pos_z_results.tolist()}")
    
    return pos_x_var, pos_y_var, pos_z_var