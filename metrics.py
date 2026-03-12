## Load packages
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.stats import f

## Compute time average data
def time_average_data(data, output_var, group_vars, metadata_vars, time_var='T_rel'):
    """
    Compute the time average of a data set.

    Args:
        data: Input data formatted as pandas dataframe.
        output_var: Name of the column of 'data' to compute the time average of.
        group_vars: List of columns to group the data by.
        metadata_vars: List of metadata columns to keep.
        time_var: Name of the column of 'data' to use as the time variable.
    
    Raises:
        Exception: one or more of the metadata_vars is not defined.
    """
    for var in group_vars:
        if var not in data.columns:
            raise Exception(f'Group variable {var} is not a column of the input data!')

    for var in metadata_vars:
        if var not in data.columns:
            raise Exception(f'Metadata variable {var} is not a column of the input data!')

    # Construct aggregation dictionary
    agg_dict = {
        **{var: (var, 'first') for var in metadata_vars},  # Metadata columns
        output_var+'_Mean': (output_var, 'mean'),   # Mean
        output_var+'_Std': (output_var, 'std'),     # Standard Deviation
        output_var+'_SEM': (output_var, 'sem')      # Standard Error of Mean
    }

    # Group by and aggregate
    df_time_avg = data.groupby(group_vars + [time_var], as_index=False).agg(**agg_dict)
    
    # Coerce to numeric types
    df_time_avg[output_var+'_Mean'] = pd.to_numeric(df_time_avg[output_var+'_Mean'], errors='coerce').astype(float)
    df_time_avg[output_var+'_Std'] = pd.to_numeric(df_time_avg[output_var+'_Std'], errors='coerce').astype(float)
    df_time_avg[output_var+'_SEM'] = pd.to_numeric(df_time_avg[output_var+'_SEM'], errors='coerce').astype(float)

    return df_time_avg

## Compute infiltration counts
def compute_infiltration_counts(data, df_boundary_all, wells, inner_circles, id_vars, time_variables=['Time_frame', 'Time [hr]'],decimals=1):
    """
    Compute the number and fraction of cells within specified inner circles of droplets.
    
    Parameters:
    - data: DataFrame containing cell data with positions and well information.
    - df_boundary_all: DataFrame containing droplet boundary information including radius.
    - wells: List of well identifiers to process.
    - inner_circles: List or array of fractions (0-1) representing inner circle radii relative to droplet radius.
    - id_vars: List of columns to retain in the final aggregated DataFrame.
    - time_variables: List of time-related columns to include in grouping.
    - decimals: Number of decimal places to round inner circle fractions for consistent naming.
    
    Returns:
    - df_count_cells: Aggregated DataFrame with counts and fractions of cells within each inner circle.
    """

    # Initialize columns for droplet area and inner circle flags
    data = data.copy()
    data['total_n_cells'] = np.nan
    data['droplet_area'] = np.nan
    
    for inner_circ in inner_circles:
        inner_circ = np.round(inner_circ, decimals=decimals) # ensure consistent rounding
        data['in_droplet_'+str(inner_circ)] = np.nan

    for row_i in range(len(wells)):
        # Determine droplet radius
        filt = (df_boundary_all['Well']==wells[row_i])
        droplet_radius = df_boundary_all.loc[filt, 'Droplet_radius'].iloc[0]
        
        # Filter data
        row_filt = (data['Well']==wells[row_i])

        # Total number of cells (inside + outside droplet)
        data.loc[row_filt, 'total_n_cells'] = data.loc[row_filt].shape[0]

        # droplet area
        data.loc[row_filt, 'droplet_area'] = np.pi*droplet_radius**2

        # Inner circles
        for inner_circ in inner_circles:
            inner_circ = np.round(inner_circ, decimals=decimals) # ensure consistent rounding
            data.loc[row_filt, 'in_droplet_'+str(inner_circ)] = data.loc[row_filt, 'Position r_rel'] < inner_circ*droplet_radius

    ## Count cells in each region
    # Exclude grouping variables from aggregation
    grouping_vars_fixed = ['Well', 'Channel']+time_variables
    id_vars = [x for x in id_vars if x not in grouping_vars_fixed]
    
    # Build dynamic aggregation dictionary
    agg_dict = {}
    for id_var in id_vars:
        if id_var not in data.columns:
            raise ValueError(f"Variable {id_var} not found in data columns.")
        agg_dict[id_var] = 'first'  # Use 'first' to keep the first occurrence of each variable in the group    

    # Add entries for each inner circle dynamically
    for inner_circ in inner_circles:
        inner_circ = str(np.round(inner_circ, decimals=decimals)) # ensure consistent rounding
        col_name = f'in_droplet_{inner_circ}'
        agg_dict[col_name] = ['sum', 'mean']

    # Group and aggregate
    df_count_cells = data.groupby(grouping_vars_fixed).agg(agg_dict)

    # Flatten multi-level column names
    df_count_cells.columns = [
        f"{col}_{stat}" if stat != '' else col 
        for col, stat in df_count_cells.columns
    ]
    
    # Rename columns for inner circles
    rename_dict = {}
    for inner_circ in inner_circles:
        inner_circ = str(np.round(inner_circ, decimals=decimals)) # ensure consistent rounding
        col_base = f'in_droplet_{inner_circ}'
        rename_dict[f'{col_base}_sum'] = f'n_cells_in_droplet_{inner_circ}'
        rename_dict[f'{col_base}_mean'] = f'frac_cells_in_droplet_{inner_circ}'

    for id_var in id_vars:
        rename_dict[id_var+'_first'] = id_var

    # Rename with the generated dictionary
    df_count_cells = df_count_cells.rename(columns=rename_dict)

    # Reset index
    df_count_cells = df_count_cells.reset_index()

    ## Sort results in custom order
    # Define the custom order
    custom_order = wells

    # Convert 'Well' to a categorical type with the custom order
    df_count_cells['Well'] = pd.Categorical(df_count_cells['Well'], categories=custom_order, ordered=True)

    # Sort the dataframe by 'Well'
    df_count_cells = df_count_cells.sort_values(['Well'])

    # Rename columns
    df_count_cells.rename(columns={'Time [hr]_first': 'Time [hr]'}, inplace=True)

    return df_count_cells

## Convert to long format
def convert_cell_counts_to_long_format(df_count_cells, inner_circles, condition_variables):
    """
    Convert the wide-format DataFrame with cell counts and fractions into a long-format DataFrame.
    
    Parameters:
    - df_count_cells: DataFrame containing counts and fractions of cells within inner circles.
    - inner_circles: List or array of inner circle fractions used in the original DataFrame.
    - condition_variables: List of condition variable columns to retain in the final DataFrame.
    
    Returns:
    - df_count_cells_long: Long-format DataFrame with separate rows for each inner circle measurement.
    """
    # Step 1: Collect all relevant columns dynamically (based on your formatted circle names)
    inner_strs = [f"{x:.1f}" for x in inner_circles]
    n_cols = [f'n_cells_in_droplet_{s}' for s in inner_strs]
    frac_cols = [f'frac_cells_in_droplet_{s}' for s in inner_strs]
    # id_columns = ['Well', 'Channel', 'Time_frame', 'Time [hr]', 'CD4', 'CD8', 'Condition'] # , 'Droplet_content', 'droplet_area']
    id_columns = condition_variables+['Channel', 'Condition', 'Well', 'WellID', 'Time [hr]', 'Time_frame']

    # Step 2: Melt the dataframe to long format
    df_n_long = df_count_cells.melt(
        id_vars=id_columns,  # keep these
        value_vars=n_cols,
        var_name='metric',
        value_name='n_cells_in_droplet',
    )

    df_frac_long = df_count_cells.melt(
        id_vars=id_columns,
        value_vars=frac_cols,
        var_name='metric',
        value_name='frac_cells_in_droplet'
    )

    # Step 3: Extract the inner_circle value from column names
    df_n_long['inner_circle'] = df_n_long['metric'].str.extract(r'_(\d\.\d+)$').astype(float)
    df_frac_long['inner_circle'] = df_frac_long['metric'].str.extract(r'_(\d\.\d+)$').astype(float)

    # Step 4: Drop the temporary 'metric' column
    df_n_long = df_n_long.drop(columns='metric')
    df_frac_long = df_frac_long.drop(columns='metric')

    # Step 5: Merge both on shared keys
    df_count_cells_long = pd.merge(df_n_long, df_frac_long, on=id_columns+['inner_circle'])
    df_count_cells_long 

    return df_count_cells_long

## Calculate intensity statistics with different normalization methods
from sklearn.preprocessing import StandardScaler, RobustScaler
def compute_intensity_stats(data, intensity_vars, group_vars_fixed = ['Well', 'Channel', 'Time_frame'], metadata_vars = ['Well', 'Channel', 'Time_frame'], stat_choice = 'mean'):
    df_rescaled = data.copy()
    for var in intensity_vars:
        # Standardization (Z-score normalization)
        scaler_standard = StandardScaler()
        df_rescaled[var + '_standardized'] = scaler_standard.fit_transform(df_rescaled[[var]])
        
        # Robust-scaling
        scaler_robust = RobustScaler()
        df_rescaled[var + '_robust_scaled'] = scaler_robust.fit_transform(df_rescaled[[var]])

        # Display the first few rows to check the results
        df_rescaled[[var, var + '_standardized', var + '_robust_scaled']].head()

    ## Calculate summary statistic per data set
    df_intensity_stats = pd.DataFrame() # Make a new dataframe to store the results
    # group_var = list(set(['Well', 'Channel', 'Time_frame', 'Time [hr]'] + metadata_cols_keep))
    cols_copy = [x for x in metadata_vars if x not in group_vars_fixed] # columns to copy without grouping
    for col in cols_copy:
        df_intensity_stats[col] = df_rescaled.groupby(group_vars_fixed)[col].first()

    for var_plot in intensity_vars:   
        if stat_choice == 'median':
            # Median of original data
            df_intensity_stats[f"{var_plot}_{stat_choice}"] = data.groupby(group_vars_fixed)[var_plot].median()
            # Median of rescaled data
            for scaling_choice in ['standardized', 'robust_scaled']:
                var_name = f"{var_plot}_{scaling_choice}"
                df_intensity_stats[var_name+"_"+stat_choice] = df_rescaled.groupby(group_vars_fixed)[f"{var_plot}_{scaling_choice}"].median()
        elif stat_choice == 'mean':
            # Also keep the mean and sem of the original intensity variable for reference
            df_intensity_stats[f"{var_plot}_{stat_choice}"] = data.groupby(group_vars_fixed)[var_plot].mean()
            df_intensity_stats[f"{var_plot}_sem"] = data.groupby(group_vars_fixed)[var_plot].sem()

            # Mean and sem of rescaled data
            for scaling_choice in ['standardized', 'robust_scaled']:
                var_name = f"{var_plot}_{scaling_choice}"
                df_intensity_stats[var_name+"_"+stat_choice] = df_rescaled.groupby(group_vars_fixed)[f"{var_plot}_{scaling_choice}"].mean()
                df_intensity_stats[var_name+"_sem"] = df_rescaled.groupby(group_vars_fixed)[f"{var_plot}_{scaling_choice}"].sem()
            
    df_intensity_stats.reset_index(inplace=True, drop=False)

    return df_intensity_stats

## === Obsolete, replace by calculate_group_timepoint_difference ===
def calculate_intensity_difference(df, grouping_cols, intensity_col, time_col='Time_frame'):
    """
    Calculate the difference in intensity between the final and initial time points for each group.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    grouping_cols (list): List of columns to group by.
    intensity_col (str): The column name for intensity values.
    time_col (str): The column name for time points.

    Returns:
    pd.DataFrame: A dataframe with the intensity differences for each group.
    """
    # Find the initial and final time points
    tmin = df[time_col].min()
    tmax = df[time_col].max()
    
    print(f"Time range in data: {tmin} to {tmax}")
    print(f"Unique time points: {sorted(df[time_col].unique())}")

    if time_col not in grouping_cols:
        grouping_cols = grouping_cols + [time_col]

    # Filter data for initial and final time points
    df_filtered = df[df[time_col].isin([tmin, tmax])]
    
    if df_filtered.empty:
        print("Warning: No data found for initial or final time points")
        return pd.DataFrame()

    # Aggregate to get mean intensity at each time point for each group
    agg_df = df_filtered.groupby(grouping_cols)[intensity_col].mean().reset_index()
    
    # Check if we have data for both time points for each group
    time_point_counts = agg_df.groupby([col for col in grouping_cols if col != time_col])[time_col].nunique()
    complete_groups = time_point_counts[time_point_counts == 2].index
    
    if len(complete_groups) == 0:
        print("Warning: No groups have data for both initial and final time points")
        return pd.DataFrame()

    # Filter to only include groups with both time points
    group_filter = agg_df.set_index([col for col in grouping_cols if col != time_col]).index.isin(complete_groups)
    agg_df_complete = agg_df[group_filter]

    # Pivot the table to have time points as columns
    pivot_df = agg_df_complete.pivot_table(index=[col for col in grouping_cols if col != time_col],
                                          columns=time_col,
                                          values=intensity_col).reset_index()

    # Calculate the difference between final and initial time points
    if tmin in pivot_df.columns and tmax in pivot_df.columns:
        pivot_df[f'{intensity_col}_diff'] = pivot_df[tmax] - pivot_df[tmin]
        # rename time columns for clarity
        pivot_df = pivot_df.rename(columns={tmin: f'{intensity_col}_Time_frame_{tmin}', tmax: f'{intensity_col}_Time_frame_{tmax}'})
        print(f"Successfully calculated intensity differences for {len(pivot_df)} groups")
    else:
        print(f"Available columns in pivot table: {pivot_df.columns.tolist()}")
        print("Warning: Initial or final time point not found in the pivoted data")
        return pd.DataFrame()

    return pivot_df
# ===========================================================

## Statistics
# Significance tests
from types import SimpleNamespace

def safe_kruskal(df, sample_var, test_var):
    groups = [grp[test_var].dropna().to_numpy() for _, grp in df.groupby(sample_var)]
    groups = [g for g in groups if g.size > 0]
    if len(groups) < 2:
        return SimpleNamespace(statistic=np.nan, pvalue=np.nan, note="need >=2 non-empty groups")
    all_vals = np.concatenate(groups)
    if np.unique(all_vals).size == 1:
        return SimpleNamespace(statistic=np.nan, pvalue=np.nan, note="all numbers identical")
    return stats.kruskal(*groups, nan_policy="omit")

def significance_test(data, sample_var, test_var, test='kruskal-wallis', verbose=False):
    """
    Perform a significance test.

    Args:
        data: DataFrame with the input data.
        sample_var: Name of the column of 'data' to group the samples by. Significance tests will compare the samples within each group.
        test_var: Variable to perform significance test on. Must be a column of 'data'.
        test: Type of significance test to perform. Options: 'kruskal-wallis', 'anova', 'kstwo'.
        quiet: If True, do not print the results of the significance test.

    Returns:
        result: Result of the significance test as directly obtained from the scipy.stats function.

    Raises:
        Exception: if the test is not one of the specified types.
    """
    if test=='kruskal-wallis':
        ## Safe version of Kruskal-Wallis test that handles edge cases (e.g. groups with all NaN values, or all values identical)
        # groups = [grp[test_var].dropna().to_numpy() for _, grp in data.groupby(sample_var)]
        # groups = [g for g in groups if g.size > 0]
        # if len(groups) < 2:
        #     result = SimpleNamespace(statistic=np.nan, pvalue=np.nan, note="need >=2 non-empty groups")
        # else:
        #     all_vals = np.concatenate(groups)
        #     if np.unique(all_vals).size == 1:
        #         result = SimpleNamespace(statistic=np.nan, pvalue=np.nan, note="all numbers identical")
        #     else:
        #         result = stats.kruskal(*groups, nan_policy="omit")
        result = safe_kruskal(data, sample_var, test_var)
        ## Original (unsafe) version of Kruskal-Wallis test that does not handle edge cases
        # result = stats.kruskal(*[list(sample) for _, sample in data.groupby(sample_var)[test_var]])
        if verbose:
            print(f"Kruskal-Wallis test on {test_var}, comparing samples labelled by {sample_var}")
            print( result )
        return result
    elif test=='anova':
        result = stats.f_oneway(*[list(sample) for _, sample in data.groupby(sample_var)[test_var]])
        if verbose:
            print(f"ANOVA test on {test_var}, comparing samples labelled by {sample_var}")
            print( result )
        return result
    elif test=='kstwo':
        result = stats.kstwo(*[list(sample) for _, sample in data.groupby(sample_var)[test_var]])
        if verbose:
            print(f"Two-sided Kolmogorov-Smirnov test on {test_var}, comparing samples labelled by {sample_var}")
            print( result )
        return result
    else:
        raise Exception(f"Invalid test type: {test}")

def multiple_significance_tests(data, group_var, sample_var, vars_test_all, test='kruskal-wallis', verbose=False):
    """
    Perform multiple significance test according to pre-defined variables.

    Args:
        data: DataFrame with the input data.
        group_var: Name of the column of 'data' to group the data by. Significance tests will be performed separately for each group.
        sample_var: Name of the column of 'data' to group the samples by. Significance tests will compare the samples within each group.
        vars_test_all: List of variables to perform significance tests on.
        test: Type of significance test to perform. Options: 'kruskal-wallis', 'anova'.

    Returns:
        df_results_cells: DataFrame with the results of the significance tests.
    """
    ## Check that all variables are present in the data
    if isinstance(group_var, list):
        for var in group_var:
            if var not in data.columns:
                raise Exception(f'Group variable {var} is not a column of the input data!')
    else:
        if group_var not in data.columns:
            raise Exception(f'Group variable {group_var} is not a column of the input data!')
        
    ## Perform significance tests
    data_grouped = data.groupby(group_var)

    groups = list(data_grouped.groups.keys())

    # Create an empty DataFrame with the given row and column names
    df_results_cells = pd.DataFrame(columns=['Group Variables', 'Group', 'Sample Variable', 'Samples', 'Test Variable', 'Test', 'Statistic', 'p-value'])
    for group_name, group_data in data_grouped:
        # print(group_name)
        samples = group_data[sample_var].unique()
        for test_var in vars_test_all:
            if verbose:
                print(f"Performing {test} test for {test_var} in group {group_name} with samples {samples}")
            res = significance_test(group_data, sample_var, test_var, test=test, verbose=verbose)
            # Add result to dataframe
            df_results_cells.loc[df_results_cells.shape[0]+1] = [str(group_var), group_name, sample_var, samples, test_var, test, res.statistic, res.pvalue]

    df_results_cells['Significant_0p05'] = df_results_cells['p-value']<0.05
    return df_results_cells

## Perform Hotelling's T-test
def hotelling_t2_test(X, mu_0):
    """
    One-sample Hotelling's T² test (two-sided).
    
    Parameters:
    - X: array of shape (n_samples, n_variables)
    - mu_0: array of shape (n_variables,), hypothesized mean vector
    
    Returns:
    - T2_stat: Hotelling's T² statistic
    - F_stat: Corresponding F statistic
    - p_value: two-sided p-value
    """
    X = np.asarray(X)
    mu_0 = np.asarray(mu_0)
    
    n, p = X.shape
    
    # Sample mean vector
    x_bar = np.mean(X, axis=0)
    
    # Sample covariance matrix
    S = np.cov(X, rowvar=False)
    
    # Inverse covariance matrix
    S_inv = np.linalg.inv(S)
    
    # Hotelling's T² statistic
    diff = x_bar - mu_0
    T2 = n * diff.T @ S_inv @ diff
    
    # Convert to F-statistic
    F_stat = ( (n - p) / (p * (n - 1)) ) * T2
    df1 = p
    df2 = n - p
    
    # Two-sided p-value
    p_value = 1 - f.cdf(F_stat, df1, df2)
    
    return T2, F_stat, p_value

### Compute distance to boundary
## Edit boundary points to match with image size
# Add points on the border and remove points outside the image and
def interpolate_y(x1, y1, x2, y2, x):
    """
    Interpolate y for a given x on the line segment [(x1, y1), (x2, y2)].

    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.
        x: X-coordinate to interpolate y for.

    Returns:
        y: Interpolated y-coordinate.
    """
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def add_boundary_points(path, xL, xR):
    """
    Add points [xL, yL] and [xR, yR] to the path where xL and xR are specified.

    Args:
        path: Coordinates of a line segment
        xL: X-coordinate of the left boundary point.
        xR: X-coordinate of the right boundary point.

    Returns:
        new_path: Path with boundary points added.
    """
    # Sort path if necessary (i.e. if x1<x2<...xN does not hold)
    path = path[path[:, 0].argsort()]

    # Extract the x and y coordinates
    x_coords = path[:, 0]
    y_coords = path[:, 1]

    # Compute yL
    if xL < x_coords[0]:
        # Extend the line segment between points 1 and 2 to intersect with xL
        yL = interpolate_y(x_coords[0], y_coords[0], x_coords[1], y_coords[1], xL)
    #elif np.any(xL==x_coords[0]):
        # If xL Is already in the path
        #print('xL already in path')
        #yL=None
    else:
        # Find the segment where xL belongs
        for i in range(len(x_coords) - 1):
            if x_coords[i] <= xL <= x_coords[i+1]:
                yL = interpolate_y(x_coords[i], y_coords[i], x_coords[i+1], y_coords[i+1], xL)
                break

    # Compute yR
    if xR > x_coords[-1]:
        # Extend the line segment between the last two points to intersect with xR
        yR = interpolate_y(x_coords[-2], y_coords[-2], x_coords[-1], y_coords[-1], xR)
    else:
        # Find the segment where xR belongs
        for i in range(len(x_coords) - 1):
            if x_coords[i] <= xR <= x_coords[i+1]:
                yR = interpolate_y(x_coords[i], y_coords[i], x_coords[i+1], y_coords[i+1], xR)
                break
    
    # Remove points outside range (xL, xR) if necessary 
    filt = (path[:, 0]>xL) & (path[:, 0]<xR)
    path = path[filt, :]

    # Add [xL, yL] at the beginning and [xR, yR] at the end of the path
    new_path = np.vstack(([xL, yL], path, [xR, yR]))
    
    # Ignore possible duplicate path points
    return np.unique(new_path, axis=0)

# Example usage
# path = np.array([[4, 4], [6, 3], [2, 2], [8, 5]])
# xL = 4.5
# xR = 7

# new_path = add_boundary_points(path, xL, xR)
# print("New path with boundary points added:")
# print(new_path)

## Vectorized versions
def distance_to_segments_vectorized(points, segment_start, segment_end):
    """
    Compute the distance from multiple points to multiple line segments.

    Args:
        points: (m, 2) array of points.
        segment_start: (n, 2) array of segment start points.
        segment_end: (n, 2) array of segment end points.

    Returns:
        distances: (m, n) array where distances[i, j] is the distance from points[i] to segment j.
        positions: (m, n) array where positions[i, j] is "above" or "below" for each point.
    """
    # Compute segment vectors
    segment_vec = segment_end - segment_start  # Shape: (n, 2)

    # Vector from segment start to each point
    point_vec = points[:, np.newaxis, :] - segment_start[np.newaxis, :, :]  # Shape: (m, n, 2)

    # Compute segment norms (squared length of each segment)
    segment_norm = np.sum(segment_vec**2, axis=1, keepdims=True)  # Shape: (n, 1)
    
    # Avoid division by zero
    segment_norm = np.where(segment_norm == 0, 1, segment_norm)

    # Projection of point onto the line defined by the segment
    projection = np.sum(point_vec * segment_vec[np.newaxis, :, :], axis=2) / segment_norm.T  # Shape: (m, n)

    # Clip projection values to stay within segment bounds
    projection = np.clip(projection, 0, 1)  # Shape: (m, n)

    # Compute closest points on the segment
    closest_points = segment_start[np.newaxis, :, :] + projection[:, :, np.newaxis] * segment_vec[np.newaxis, :, :]  # Shape: (m, n, 2)

    # Compute distances
    distances = np.linalg.norm(points[:, np.newaxis, :] - closest_points, axis=2)  # Shape: (m, n)

    # Determine "above" or "below" using cross product
    cross_product = np.cross(segment_vec[np.newaxis, :, :], point_vec)  # Shape: (m, n)
    positions = np.where(cross_product > 0, "below", "above")

    return distances, positions

def distance_to_path_vectorized(path, points):
    """
    Calculate the minimum distance from multiple points to a path defined by line segments.
    
    Args:
        path: A numpy array of shape (n, 2), representing points [(x1, y1), (x2, y2), ..., (xn, yn)].
        points: A numpy array of shape (m, 2), representing multiple points [[x1, y1], [x2, y2], ..., [xm, ym]].
    
    Returns:
        distances: A numpy array of shape (m,), containing the minimum distance from each point to the path.
        positions: A numpy array of shape (m,), containing whether each point is "above" or "below" the path.
    """
    path_start = path[:-1]  # Start points of segments
    path_end = path[1:]  # End points of segments

    # Vectorized distance computation for all points
    distances_all, positions_all = distance_to_segments_vectorized(points, path_start, path_end)

    # Get the minimum distance and corresponding position for each point
    min_indices = np.argmin(distances_all, axis=1)
    min_distances = distances_all[np.arange(len(points)), min_indices]
    min_positions = positions_all[np.arange(len(points)), min_indices]

    return min_distances, min_positions

# Calculate the area of concentric circles of rings
def concentric_ring_areas(radius_list):
    areas = []

    # Loop through each pair of consecutive radii
    for i in range(len(radius_list) - 1):
        inner_radius = radius_list[i]
        outer_radius = radius_list[i + 1]

        # Calculate the area of the current ring
        area = np.pi * (outer_radius ** 2 - inner_radius ** 2)
        areas.append(area)
    return areas

## Calculate Clark-Evans R index
from scipy.spatial import KDTree
def calc_clark_evans(xy_data, droplet_radius):
    if len(xy_data) < 2:
        return np.nan, np.nan, np.nan
    tree = KDTree(xy_data)
    # Query all points at once for their two nearest neighbors
    dists, _ = tree.query(xy_data, k=2)
    nearest_neighbor_distances = dists[:, 1]  # skip self (distance 0)
    observed_mean_distance = nearest_neighbor_distances.mean()
    area = np.pi * droplet_radius ** 2
    n = len(xy_data)
    expected_mean_distance = 1 / (2 * np.sqrt(n / area))
    R = observed_mean_distance / expected_mean_distance
    return observed_mean_distance, expected_mean_distance, R

from itertools import product
def calc_clark_evans_batch(data, wells, time_frames, channels, df_boundary_all, scale_factor=None):
    ## Prepare results dataframe
    # Generate all unique pairs using Cartesian product
    pairs = list( product(wells, time_frames, channels))

    # Create a DataFrame from the pairs
    df_nearest_neighbor_results = pd.DataFrame(pairs, columns=['Well', 'Time_frame', 'Channel'])

    # Manually add additional information
    df_nearest_neighbor_results['Observed Mean Distance'] = ''
    df_nearest_neighbor_results['Expected Mean Distance'] = ''
    df_nearest_neighbor_results['Clark-Evans R Index'] = ''
    df_nearest_neighbor_results['Number of Cells'] = 0
    
    # Collect results in a list of dicts
    results = []
    for row_i in range(len(wells)):
        print(f"Processing well {row_i+1}/{len(wells)}: {wells[row_i]}")
        for row_j in range(len(time_frames)):
            filt = (df_boundary_all['Well'] == wells[row_i])
            droplet_radius = df_boundary_all.loc[filt, 'Droplet_radius'].iloc[0]
            for row_k in range(len(channels)):
                mask = (
                    (data['Well'] == wells[row_i]) &
                    (data['Time_frame'] == time_frames[row_j]) &
                    (data['Channel'] == channels[row_k])
                )
                data_filt = data.loc[mask]
                if data_filt.empty:
                    continue

                # All data inside droplet
                data_filt_inside = data_filt.loc[data_filt['Position r_rel'] < droplet_radius]
                xy_data = data_filt_inside[['Position X', 'Position Y']].values
                observed_mean_distance, expected_mean_distance, clark_evans_R = calc_clark_evans(xy_data, droplet_radius)

                id_columns = ['CD4', 'CD8', 'PAM_uM', 'BME_matrix', 'MSC_tumor_ratio', 'Compound_added', 'Compound_conc', 'Channel', 'Condition', 'Well']
                row_dat = data_filt.loc[:, id_columns].drop_duplicates()
                if row_dat.shape[0] == 0:
                    print(f"Warning: No entry found for {wells[row_i]}, {time_frames[row_j]}, {channels[row_k]}")
                    continue
                elif row_dat.shape[0] > 1:
                    print(f"Warning: More than one entry found for {row_dat} for {wells[row_i]}, {time_frames[row_j]}, {channels[row_k]}")
                row_dat = row_dat.iloc[0]

                result = {
                    'Well': wells[row_i],
                    'Time_frame': time_frames[row_j],
                    'Channel': channels[row_k],
                    'Observed Mean Distance': observed_mean_distance,
                    'Expected Mean Distance': expected_mean_distance,
                    'Clark-Evans R Index': clark_evans_R,
                    'Number of Cells': xy_data.shape[0]
                }
                for col_name in id_columns:
                    result[col_name] = row_dat[col_name]

                # Data further inside droplet (scaled)
                if scale_factor is not None:
                    data_filt_inside_scaled = data_filt.loc[data_filt['Position r_rel'] < scale_factor * droplet_radius]
                    xy_data_scaled = data_filt_inside_scaled[['Position X', 'Position Y']].values
                    obs_md_s, exp_md_s, r_s = calc_clark_evans(xy_data_scaled, droplet_radius)
                    result['Observed Mean Distance scaled ' + str(scale_factor)] = obs_md_s
                    result['Expected Mean Distance scaled ' + str(scale_factor)] = exp_md_s
                    result['Clark-Evans R Index scaled ' + str(scale_factor)] = r_s
                    result['Number of Cells scaled ' + str(scale_factor)] = xy_data_scaled.shape[0]

                results.append(result)
    
    # Create DataFrame from results and update df_nearest_neighbor_results
    results_df = pd.DataFrame(results)
    # Merge on Well, Time_frame, Channel (left: df_nearest_neighbor_results, right: results_df)
    df_nearest_neighbor_results = df_nearest_neighbor_results.drop(
        columns=[col for col in results_df.columns if col in df_nearest_neighbor_results.columns and col not in ['Well', 'Time_frame', 'Channel']],
        errors='ignore'
    )
    df_nearest_neighbor_results = pd.merge(
        df_nearest_neighbor_results,
        results_df,
        on=['Well', 'Time_frame', 'Channel'],
        how='left'
    )
    
    return df_nearest_neighbor_results