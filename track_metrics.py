## Compute track metrics from track data (experimental or simulated)
## Note 25-05-14: This module is written mainly for simulated data in the form of Numpy arrays, 
# and is therefore not efficient for experimental data in the form of Pandas DataFrames.
# However, it can be used for experimental data as well, but it is not optimized for that.

## Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram, fcluster, ward
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from utils import find_position_columns

### Compute lists of metrics for df_data_cells or df_data_tracks
def calculate_additional_cell_variables(df_data_cells, variables_to_compute, var_track_id='TrackID2', var_time_index='Time Index', 
                                        position_vars=['Position X [µm]', 'Position Y [µm]', 'Position Z [µm]']):
    supported_variables = ['Time_rel', 'Displacement', 'Displacement_vector_length', 'X_rel', 'Y_rel', 'Z_rel',
                           'Track_length_frames', 'Turning_angle', 'Aspect_ratio', 'Displacement^2', 'Ellipse Axis B Angle']
    for var in variables_to_compute:
        if var not in supported_variables:
            raise Exception(f"Variable {var} not computable!")
    
    if position_vars is None:
        # Try to automatically find position columns based on common naming patterns (only works if position columns are named in a standard way)
        pos_x_var, pos_y_var, pos_z_var = find_position_columns(df_data_cells) # find position columns
        position_vars = [pos_x_var, pos_y_var, pos_z_var]

    # Calculate time from start of track
    if 'Time_rel' in variables_to_compute:
        df_data_cells['Time_rel'] = df_data_cells.groupby(var_track_id)[var_time_index].transform(lambda x: x - x.min())
        df_data_cells = df_data_cells.sort_values([var_track_id, 'Time_rel'])

    # Calculate relative displacements
    # if 'X_rel' in variables_to_compute or 'Y_rel' in variables_to_compute or 'Z_rel' in variables_to_compute:
        # df_data_cells['X_rel'] = df_data_cells.groupby(var_track_id)[pos_x_var].transform(lambda x: x - x.iloc[0])
        # df_data_cells['Y_rel'] = df_data_cells.groupby(var_track_id)[pos_y_var].transform(lambda x: x - x.iloc[0])
        # if pos_z_var is not None:
        #     df_data_cells['Z_rel'] = df_data_cells.groupby(var_track_id)[pos_z_var].transform(lambda x: x - x.iloc[0])

    if ('Displacement' in variables_to_compute) or ('Displacement_vector_length' in variables_to_compute) or ('Turning_angle' in variables_to_compute):
        if 'Displacement X' not in df_data_cells.columns:
            print("Calculating Displacement X...")
            df_data_cells['Displacement X'] = df_data_cells.groupby(var_track_id)[pos_x_var].diff()
        if 'Displacement Y' not in df_data_cells.columns:
            print("Calculating Displacement Y...")
            df_data_cells['Displacement Y'] = df_data_cells.groupby(var_track_id)[pos_y_var].diff()
        if ('Displacement Z' not in df_data_cells.columns) and (pos_z_var is not None):
            print("Calculating Displacement Z...")
            df_data_cells['Displacement Z'] = df_data_cells.groupby(var_track_id)[pos_z_var].diff()
    
    # Calculate displacement vector length
    if 'Displacement_vector_length' in variables_to_compute:
        df_data_cells['Displacement_vector_length'] = np.sqrt(
            df_data_cells['Displacement X']**2 +
            df_data_cells['Displacement Y']**2 +
            df_data_cells['Displacement Z']**2
        )

    # Calculate turning angle
    if 'Track_length_frames' in variables_to_compute:
        ## Add a track length in frames variable to df_data_cells
        df_data_cells['Track_length_frames'] = df_data_cells.groupby(var_track_id)[var_track_id].transform('count')
        print("Added variable 'Track_length_frames' to data")

    ## Turning angle
    if 'Turning_angle' in variables_to_compute:
        df_data_cells['Turning_angle'] = df_data_cells.groupby(var_track_id, group_keys=False).apply(lambda g: calc_turning_angle(
            g['Displacement X'],
            g['Displacement Y'],
            g['Displacement Z'])
        )
        df_data_cells['Turning_angle_degrees'] = df_data_cells['Turning_angle']/ np.pi * 180 # Convert to degrees

        # Alternative definition used by Imaris: angle between velocity and the X axis
        #df_data_cells['Turning_angle'] = np.arctan2(df_data_cells['Velocity Y'], df_data_cells['Velocity X'])*180/np.pi
        #df_data_cells.loc[ (df_data_cells['Velocity X']==0) & (df_data_cells['Velocity Y']==0), 'Turning_angle'  ] = np.nan # non-moving cells
        print("Added variable 'Turning_angle' and 'Turning_angle_degrees' to data")

    # Aspect ratio = major axis / minor axis
    if 'Aspect_ratio' in variables_to_compute:
        df_data_cells['Aspect_ratio'] = df_data_cells['Ellipse Axis Length B']/df_data_cells['Ellipse Axis Length A']
        print("Added variable 'Aspect_ratio' to data")

    # Angle of major ellipse axis with x-axis (in radians, between -pi/2 and pi/2)
    if 'Ellipse Axis B Angle' in variables_to_compute:
        df_data_cells['Ellipse Axis B Angle'] = np.arctan(df_data_cells['Ellipse Axis B Y']/df_data_cells['Ellipse Axis B X'])
        print("Added variable 'Ellipse Axis B Angle' to data")

    # MSD
    # Calculate displacement^2 if needed
    if 'Displacement^2' in variables_to_compute:
        df_data_cells['Displacement^2'] = df_data_cells['Displacement Length']**2
        print("Added variable 'Displacement^2' to data")

    return df_data_cells

def calculate_additional_track_variables(df_data_tracks, df_data_cells, variables_to_compute,
                                         position_vars=['Position X [µm]', 'Position Y [µm]', 'Position Z [µm]']):
    supported_variables = ['Total Turning Angle', 'Spatial Coverage', 'max_displacement']
    for var in variables_to_compute:
        if var not in supported_variables:
            raise Exception(f"Variable {var} not computable!")

    if position_vars is None:
        # Try to automatically find position columns based on common naming patterns (only works if position columns are named in a standard way)
        pos_x_var, pos_y_var, pos_z_var = find_position_columns(df_data_cells) # find position columns
        position_vars = [pos_x_var, pos_y_var, pos_z_var]

    ## Total turning angle
    if 'Total Turning Angle' in variables_to_compute:
        if 'Turning_angle' not in df_data_cells.columns:
            raise Exception("Variable 'Turning_angle' not found in df_data_cells!")
        df_tmp = df_data_cells.groupby('TrackID2').agg({'Turning_angle': lambda x: np.nansum(x)}).rename(columns={'Turning_angle':'Total Turning Angle'})
        df_data_tracks = pd.merge(df_data_tracks.loc[:, [col for col in df_data_tracks.columns if col!='Total Turning Angle']], df_tmp, how='left', on='TrackID2')
        print("Added variable 'Total Turning Angle' to data")

    ## Calculate spatial coverage for all tracks
    if 'Spatial Coverage' in variables_to_compute:
        
        df_tmp = df_data_cells.groupby('TrackID2').apply(calculate_spatial_coverage,
                                                         var_names = ['Time_rel', position_vars[0], position_vars[1], position_vars[2]]).reset_index()
        ## Integrate result into df_data_track
        df_data_tracks = pd.merge(df_data_tracks.loc[:, [col for col in df_data_tracks.columns if col!='Spatial Coverage']], df_tmp[['TrackID2', 'Spatial Coverage']], on='TrackID2')
        print("Added variable 'Spatial Coverage' to data")

    ## Calculate max distance to initial position for all tracks
    if 'max_displacement' in variables_to_compute:
        data_df_copy = df_data_cells.copy()
        if 'Position Z_rel' in data_df_copy:
            data_df_copy['dist_rel'] = np.sqrt(data_df_copy['Position X_rel']**2+data_df_copy['Position Y_rel']**2+data_df_copy['Position Z_rel']**2)
        else:
            data_df_copy['dist_rel'] = np.sqrt(data_df_copy['Position X_rel']**2+data_df_copy['Position Y_rel']**2)

        df_track_tmp = data_df_copy.groupby('TrackID2').agg(
            TrackID2=('TrackID2','first'),
            max_displacement=('dist_rel', 'max')
        )
        df_data_tracks = pd.merge(df_data_tracks.loc[:, [col for col in df_data_tracks.columns if col!='max_displacement']], df_track_tmp[['max_displacement']], on='TrackID2')
        print("Added variable 'max_displacement' to data")

    return df_data_tracks

def calculate_track_intensity_variables(df_data_tracks, df_data_cells, intensity_vars):
    ## Compute track averages for intensity variables
    intensity_vars_cell = [s.replace("_", " ") for s in intensity_vars]
    intensity_vars_track = ['Track Avg '+s.replace("_", " ") for s in intensity_vars]
    print('Intensity vars track:', intensity_vars_track)

    for var in intensity_vars_cell:
        if var not in df_data_cells.columns:
            raise Exception(f"Intensity variable {var} not found in input cell data (df_data_cells)!")
    df_tmp = df_data_cells.loc[:, ['TrackID2']+intensity_vars]
    df_tmp = df_tmp.groupby('TrackID2').mean()
    df_tmp.rename(columns=dict(zip(intensity_vars, intensity_vars_track)), inplace=True)
    # df_tmp
    
    ## Add track averages to df_data_tracks (only once!)
    if not np.all([var in df_data_tracks.columns for var in intensity_vars_track]):
        df_data_tracks_tmp = pd.merge(df_data_tracks.loc[:, [col for col in df_data_tracks.columns if col not in intensity_vars_track]], df_tmp, on='TrackID2')
        print(f'Added track average intensity data {intensity_vars_track}')
        print('Shape before: ', df_data_tracks.shape)
        print('Shape after: ', df_data_tracks_tmp.shape)
        df_data_tracks = df_data_tracks_tmp
        df_data_tracks
    else:
        print('Data already merged!')

    return df_data_tracks

### Compute single quantities
# TO DO: homogenize so that all are for single tracks rather than full data.
# Calculate turning angles (angles between consecutive displacement vectors)
def calc_turning_angle(x, y, z):
    # Dot product of current and next vector
    dot = x * x.shift(1) + y * y.shift(1) + z * z.shift(1)

    # Norm (magnitude) of current and next vectors
    norm_u = np.sqrt(x**2 + y**2 + z**2)
    norm_v = np.sqrt(x.shift(1)**2 + y.shift(1)**2 + z.shift(1)**2)

    # Angle in radians
    cos_angle = dot / (norm_u * norm_v)

    # Avoid numerical issues (e.g. rounding leading to cos_angle > 1 or < -1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angles = np.arccos(cos_angle)

    return angles

## Calculate spatial coverage of a track
def calculate_spatial_coverage(track_data, var_names=['Time_rel', 'Position X', 'Position Y', 'Position Z']):
    """
    Calculate spatial coverage of a single track, formatted as DataFrame. The spatial coverage is defined as the area (2D) or volume (3D) of the convex hull enclosing all track points.
    See also the CellTracksColab tool for an example, https://doi.org/10.1371/journal.pbio.3002740.
    
    Args:
        track_data: DataFrame or pandas with the input track data.
        var_names: List of variable names for time [0] and coordinates [1, 2, (3)] in case track_data is a DataFrame.

    Returns:
        spatial_coverage: Spatial coverage of the track data.

    Raises:
        Exception: If the input data does not contain the necessary columns.
    """
    if isinstance(track_data, pd.DataFrame):
        # Check input data
        for column in var_names:
            if column not in track_data.columns:
                raise Exception(f"Column {column} not found in track_data.")

        track_data = track_data.sort_values(var_names[0])
        
        # Apply spatial calibration to the coordinates
        z_data = len(var_names) > 3 # 'Position Z' in track_data.columns
        calibrated_coords = track_data[ var_names[1:] ].values  # Ensure .values is added here
    elif isinstance(track_data, np.ndarray):
        # Check input data
        z_data = (track_data.shape[1] == 3)
        if track_data.shape[1] < 2:
            raise Exception("Input data must have at least 2 columns for coordinates.")
        calibrated_coords = track_data.copy()
    else:
        raise Exception("Input data must be a DataFrame or a numpy array.")

    # Drop rows with NaN values in the coordinates
    calibrated_coords = calibrated_coords[~np.isnan(calibrated_coords).any(axis=1)]

    if z_data:  # If variance of Z is 0, calculate 2D spatial coverage
        # calculate 3D spatial coverage
        if len(calibrated_coords) < 4:  # Need at least 4 points for a 3D convex hull
            return pd.Series({'Spatial Coverage': 0})
        try:
            hull = ConvexHull(calibrated_coords, qhull_options='QJ')  # 'QJ' joggles the input to avoid precision errors
            spatial_coverage = hull.volume  # Volume of the convex hull in 3D
        except Exception as e:
            print(f"Error calculating 3D spatial coverage: {e}")
            spatial_coverage = 0
    else:  # Calculate 2D spatial coverage
        if len(calibrated_coords) < 3:  # Need at least 3 points for a 2D convex hull
            return pd.Series({'Spatial Coverage': 0})
        try:
            coords_2d = calibrated_coords[:, :2]  # Use only X and Y coordinates
            hull_2d = ConvexHull(coords_2d, qhull_options='QJ')  # 'QJ' joggles the input to avoid precision errors
            spatial_coverage = hull_2d.volume  # Area of the convex hull in 2D
        except Exception as e:
            print(f"Error calculating 2D spatial coverage: {e}")
            spatial_coverage = 0

    return pd.Series({'Spatial Coverage': spatial_coverage})

def compute_msd(track_data, condition_vars, time_frame_col='Time_frame', displacement_col='Displacement Length'):
    """
    Compute the Mean Squared Displacement (MSD) for a given set of tracks.
    Parameters:
    - track_data: DataFrame containing the track data.
    - condition_vars: List of columns to group by (e.g., ['Cell_type', 'TrackID2']).
    - time_frame_col: Column name for time frames (default is 'Time_frame').
    - displacement_col: Column name for displacement (default is 'Displacement Length').
    Returns:
    - msd_all: DataFrame containing the MSD values for each group.
    """
    
    ## Calculate displacement^2 if needed
    if ('Displacement^2' not in track_data.columns):
        track_data['Displacement^2'] = track_data[displacement_col]**2
    
    track_data['Displacement^2'] = track_data['Displacement^2'].astype(float)
    track_data['Displacement^2'] = track_data['Displacement^2'].replace([np.inf, -np.inf], np.nan)
    track_data['Displacement^2'] = track_data['Displacement^2'].fillna(0)
    
    msd_all = track_data.groupby(condition_vars+[time_frame_col]).agg({'Displacement^2': 'mean'}).reset_index() # .groupby('TrackID2')['Displacement^2'].mean()
    msd_all = msd_all.rename(columns={'Displacement^2': 'msd'})
    return msd_all

def compute_displacement_autocorrelation(track_data, group_vars, max_lag=20):
    """
    Computes normalized displacement autocorrelation curves grouped by condition.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns:
        - 'TrackID2', 'Time_rel'
        - 'Position X', 'Position Y', 'Position Z'
        - A categorical condition column (e.g. treatment or cell type)
    
    condition_vars: str
        List of columns to group by (e.g., ['Cell_type', 'TrackID2']).

    max_lag : int
        Maximum lag (in frames) to compute autocorrelation for

    Returns:
    --------
    pd.DataFrame with columns: ['Lag', 'Autocorrelation', 'Condition']
    """

    # # Clean and compute displacements
    # df = df.sort_values(['TrackID2', 'Time_rel']).copy()

    # df['Displacement X'] = df.groupby('TrackID2')['Position X'].diff()
    # df['Displacement Y'] = df.groupby('TrackID2')['Position Y'].diff()
    # df['Displacement Z'] = df.groupby('TrackID2')['Position Z'].diff()

    track_data = track_data.dropna(subset=['Displacement X', 'Displacement Y', 'Displacement Z'])
    
    results = []
    
    for group, group_track_data in track_data.groupby(group_vars):
        autocorr = []

        for lag in range(max_lag + 1):
            dot_products = []

            for track_id, g in group_track_data.groupby('ID'):
                g = g.sort_values('time')

                dx = g['Displacement X'].values
                dy = g['Displacement Y'].values
                dz = g['Displacement Z'].values

                if len(dx) <= lag:
                    continue

                d1 = np.stack([dx[:-lag or None], dy[:-lag or None], dz[:-lag or None]], axis=1)
                d2 = np.stack([dx[lag:], dy[lag:], dz[lag:]], axis=1)

                dot = np.sum(d1 * d2, axis=1)
                dot_products.append(dot)

            if dot_products:
                all_dots = np.concatenate(dot_products)
                autocorr.append(all_dots.mean())
            else:
                autocorr.append(np.nan)

        # Normalize
        autocorr = np.array(autocorr)
        autocorr /= autocorr[0] if autocorr[0] != 0 else 1

        # Store results
        results.extend([
            {'Lag': lag, 'Autocorrelation': ac, 'Group': group}
            for lag, ac in enumerate(autocorr)
        ])

    return pd.DataFrame(results)

## Define turning angle function for numpy arrays
def calc_turning_angle_np(x, y, z=None):
    """
    Calculate the turning angle between consecutive 2D or 3D vectors.
    Inputs:
        x, y: 1D NumPy arrays of equal length
        z: 1D NumPy array (optional, same length as x and y). If None, assumes 2D and sets z=0.
    Returns:
        1D array of angles in radians (length = len(x) - 1)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if z is None:
        z = np.zeros_like(x)
    else:
        z = np.asarray(z)
    # Stack vectors: shape (n, 3)
    u = np.stack([x[:-1], y[:-1], z[:-1]], axis=1)
    v = np.stack([x[1:], y[1:], z[1:]], axis=1)
    # Dot product and norms
    dot = np.sum(u * v, axis=1)
    norm_u = np.linalg.norm(u, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    # Compute angle
    cos_angle = dot / (norm_u * norm_v)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle)
    return angles

## Compute velocities for simulations
def compute_velocities(sims, del_t):
    d = sims.shape[2] 
    if (d > 3) or (d < 2):
        raise ValueError("Input data must be 2D or 3D (x, y, z) coordinates.")

    n_sim, n_steps_sim, _ = sims.shape

    v_all = np.empty((n_sim, n_steps_sim, d))

    # Forward difference for first time step
    v_all[:, 0, :] = (sims[:, 1, :] - sims[:, 0, :]) / del_t

    # Backward difference for last time step
    v_all[:, -1, :] = (sims[:, -1, :] - sims[:, -2, :]) / del_t

    # Central differences for inner time steps
    v_all[:, 1:-1, :] = (sims[:, 2:, :] - sims[:, :-2, :]) / (2 * del_t)

    return v_all

### Compute metric lists
#### For single simulation (with errors, discontinued)
def compute_cell_metrics_from_sim_single_np(sim):
    """
    Compute cell metrics for a single simulation trajectory.
    sim: ndarray of shape (n_steps, 1+d)
    Returns: DataFrame with metrics for each step (n_steps-1 rows)
    """
    n_tp = sim.shape[0]
    n_d = sim.shape[1]-1
    if (n_d < 2) or (n_d > 3):
        raise ValueError("Input data must be time plus 2D (t, x, y) or 3D (t, x, y, z) coordinates.")

    del_t = sim[1, 0] - sim[0, 0]  # Time step (assumed constant)
    displacements = np.diff(sim[:, 1:], axis=0)  # shape: (n_tp-1, n_d)
    displacement_vector_length = np.sqrt(np.sum(displacements**2, axis=1))
    speed = displacement_vector_length / del_t

    # Calculate turning angle (skip first step, so output is n_tp-2)
    turning_angle = np.full((n_tp-1, ), np.nan)
    if n_d == 3:
        if n_tp > 2:
            turning_angle[1:] = calc_turning_angle_np(displacements[:, 0], displacements[:, 1], displacements[:, 2])
    else:  
        if n_tp > 2:
            turning_angle[1:] = calc_turning_angle_np(displacements[:, 0], displacements[:, 1])

    df_cell_metrics = pd.DataFrame({
        'Time': sim[1:, 0],
        'Displacement X': displacements[:, 0],
        'Displacement Y': displacements[:, 1],
        'Speed': speed,
        'Displacement_vector_length': displacement_vector_length,
        'Turning_angle_radians': turning_angle
    })
    if n_d == 3:
        df_cell_metrics['Displacement Z'] = displacements[:, 2]
    return df_cell_metrics

#### For NumPy arrays
def compute_cell_metrics_from_sim(sims, del_t):
    n_sim = sims.shape[0]
    n_steps_sim = sims.shape[1]
    d = sims.shape[2] 
    if (d<2) | (d>3):
        raise ValueError("Input data must be 2D or 3D (x, y, z) coordinates.")
    
    ## Compute cell metrics
    displacements = np.diff(sims, axis=1)  # shape: (n_sim, n_steps, 3)
    displacement_vector_length = np.sqrt(np.sum(displacements**2, axis=2))
    speed = displacement_vector_length / del_t
    turning_angle = np.empty((n_sim, n_steps_sim-1))*np.nan
    if d==3:
        for i in range(n_sim):
            turning_angle[i, 1:] = calc_turning_angle_np(displacements[i, :, 0], displacements[i, :, 1], displacements[i, :, 2])
    elif d==2:
        for i in range(n_sim):
            turning_angle[i, 1:] = calc_turning_angle_np(displacements[i, :, 0], displacements[i, :, 1])
    
    ## Make DataFrame of all metrics
    cell_metrics = {
        'Displacement X': displacements[:,:,0].flatten(),
        'Displacement Y': displacements[:,:,1].flatten(),
        'Speed': speed.flatten(),
        'Displacement_vector_length': displacement_vector_length.flatten(),
        'Turning_angle_radians': turning_angle.flatten()
    }
    if d == 3:
        cell_metrics['Displacement Z'] = displacements[:,:,2].flatten()
    df_cell_metrics = pd.DataFrame(cell_metrics)
    return df_cell_metrics

def compute_track_metrics_from_sim(sims, del_t=1):
    ## Compute metrics from simulated tracks
    n_sim = sims.shape[0]
    d = sims.shape[2]

    # Compute relevant cell metrics
    displacements = np.diff(sims, axis=1)  # shape: (n_sim, n_steps, 3)
    displacement_vector_length = np.sqrt(np.sum(displacements**2, axis=2))
        
    velocity = compute_velocities(sims, del_t)
    velocity_norm = np.sqrt(np.sum(velocity**2, axis=2))
    velocity_norm_2 = displacement_vector_length / del_t
    track_speed_mean_2 = np.mean(velocity_norm_2, axis=1)
    
    # Track duration
    n_steps_sim = sims.shape[1]-1
    track_duration = n_steps_sim*del_t
    # Track displacement length - distance from start to end
    track_displacement_length = np.sqrt(np.sum((sims[:, -1, :]-sims[:, 0, :])**2, axis=1))
    # Track length - sum of all displacements
    track_length = np.sum(displacement_vector_length, axis=1)
    # Track Speed Mean
    track_speed_mean = np.mean(velocity_norm, axis=1)
    # Track Straightness / Persistence - ratio of displacement length to total length
    track_straightness = track_displacement_length / track_length
    # Track Speed Variation
    track_speed_variation = np.std(velocity_norm, axis=1) / np.mean(velocity_norm, axis=1)
    track_speed_variation_2 = np.std(velocity_norm_2, axis=1) / np.mean(velocity_norm_2, axis=1)
    # Spatial Coverage
    track_spatial_coverage = np.array([calculate_spatial_coverage(slice) for slice in sims]).flatten() # <- speed bottleneck!
    # Total Turning Angle
    total_turning_angle = np.empty((n_sim))*np.nan
    if d==3:
        for i in range(n_sim):
            total_turning_angle[i] = np.sum(calc_turning_angle_np(displacements[i, :, 0], displacements[i, :, 1], displacements[i, :, 2]))
    elif d==2:
        for i in range(n_sim):
            total_turning_angle[i] = np.sum(calc_turning_angle_np(displacements[i, :, 0], displacements[i, :, 1]))
        
    ## Combine into a DataFrame
    df_track_metrics = pd.DataFrame({
        'Track Duration': track_duration,
        'Track Displacement Length': track_displacement_length,
        'Track Length': track_length,
        'Track Speed Mean': track_speed_mean,
        'Track Speed Mean 2': track_speed_mean_2,
        'Track Straightness': track_straightness,
        'Track Speed Variation': track_speed_variation,
        'Track Speed Variation 2': track_speed_variation_2,
        'Spatial Coverage': track_spatial_coverage,  # Placeholder for spatial coverage
        'Total Turning Angle': total_turning_angle,
    })
    return df_track_metrics

#### For DataFrames
## Compute metrics for data frames in long format (with column indicating track ID), e.g. for Imaris input
def compute_cell_metrics_from_sim_single(traj, del_t):
    """
    Wrapper for existing function to handle single trajectory.
    traj: ndarray of shape (n_steps, n_dimensions)
    Returns: DataFrame with one row (metrics for this track)
    """
    # Simulate (1, n_steps, n_dimensions) to fit old function
    sims = traj[np.newaxis, :, :]
    return compute_cell_metrics_from_sim(sims, del_t)

def compute_cell_metrics_from_long_df(df_long, del_t, track_id_col = 'TrackID2', time_col = 'Time_frame', coord_cols=['Position X', 'Position Y', 'Position Z']):
    """
    Generalized function to compute metrics from a long-format DataFrame.
    df_long: DataFrame with columns ['TrackID2', 'Time', 'X', 'Y', 'Z', ...]
    Returns: DataFrame with metrics per TrackID2
    """
    results = []

    for track_id, group in df_long.groupby(track_id_col):
        # Ensure correct ordering by time (optional but safe)
        group_sorted = group.sort_values(time_col)

        # Extract trajectory (n_steps_i, n_dimensions)
        traj = group_sorted[coord_cols].to_numpy()

        # Compute metrics using existing function
        metrics_df = compute_cell_metrics_from_sim_single(traj, del_t)

        # Add TrackID2 to result
        metrics_df[track_id_col] = track_id

        results.append(metrics_df)

    # Concatenate all metrics into one DataFrame
    final_metrics_df = pd.concat(results, ignore_index=True)

    return final_metrics_df

def compute_track_metrics_from_sim_single(traj, del_t):
    """
    Wrapper for existing function to handle single trajectory.
    traj: ndarray of shape (n_steps, n_dimensions)
    Returns: DataFrame with one row (metrics for this track)
    """
    # Simulate (1, n_steps, n_dimensions) to fit old function
    sims = traj[np.newaxis, :, :]
    return compute_track_metrics_from_sim(sims, del_t)

def compute_track_metrics_from_long_df(df_long, del_t, track_id_col = 'TrackID2', time_col = 'Time_frame', coord_cols=['Position X', 'Position Y', 'Position Z']):
    """
    Generalized function to compute metrics from a long-format DataFrame.
    df_long: DataFrame with columns ['TrackID2', 'Time', 'X', 'Y', 'Z', ...]
    Returns: DataFrame with metrics per TrackID2
    """
    results = []

    for track_id, group in df_long.groupby(track_id_col):
        # Ensure correct ordering by time (optional but safe)
        group_sorted = group.sort_values(time_col)

        # Extract trajectory (n_steps_i, n_dimensions)
        traj = group_sorted[coord_cols].to_numpy()

        # Compute metrics using existing function
        metrics_df = compute_track_metrics_from_sim_single(traj, del_t)

        # Add TrackID2 to result
        metrics_df[track_id_col] = track_id

        results.append(metrics_df)

    # Concatenate all metrics into one DataFrame
    final_metrics_df = pd.concat(results, ignore_index=True)

    return final_metrics_df

### Clustering and separation score functions from CellPhePy and other sources
### ========= Separation score functions from CellPhePy ==========
def calculate_separation_scores(dfs: list[pd.DataFrame], threshold: float | str = 0) -> pd.DataFrame:
    """
    Calculates the separation score between multiple datasets (n groups) across a number of features.
    A threshold can be supplied to identify discriminatory variables.

    Note that each DataFrame should include only columns of features (i.e., remove
    cell identifier columns prior to function use), and these columns must be
    the same in number and data type across all DataFrames.
    
    :param dfs: List of DataFrames, each representing a different group, containing only feature columns.
    :param threshold: Separation threshold either as a number or the string 'auto'.
        If a number, then features with a separation score below this value are discarded.
        If 'auto', then the threshold is automatically identified.
    :return: A DataFrame comprising 2 columns: Feature and Separation, where
        each row corresponds to a different feature's separation score. Any
        features with separation scores less than threshold are removed.
    """
    # Validate input
    if len(dfs) < 2:
        raise ValueError("At least 2 DataFrames are required.")
    if threshold != "auto" and not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be 'auto' or numeric")

    # Assign group labels to each DataFrame and concatenate them
    for i, df in enumerate(dfs):
        df["group"] = f"g{i + 1}"
    # Combine all DataFrames into one
    df_combined = pd.concat(dfs)
    # Melt into long format
    df_long = pd.melt(df_combined, id_vars="group", var_name="Feature", value_name="value")
    # Aggregate data
    df_agg = df_long.groupby(["group", "Feature"]).agg(["count", "mean", "var"]).reset_index()

    # Remove multi-index columns
    df_agg.columns = [x[0] if i < 2 else x[1] for i, x in enumerate(df_agg.columns.values)]

    # Pivot the DataFrame to have each group as columns for count, mean, and var
    df = df_agg.pivot(columns="group", index="Feature").reset_index()

    # Calculate the grand total (sum of counts across all groups)
    df["sum"] = df["count"].sum(axis=1)
    # Calculate the grand mean (weighted mean of means from all groups)
    overmean = sum(df["count"][f"g{i + 1}"] * df["mean"][f"g{i + 1}"] for i in range(len(dfs))) / df["sum"]
    # Calculate within-group variance (Vw)
    df["Vw"] = sum((df["count"][f"g{i + 1}"] - 1) * df["var"][f"g{i + 1}"] for i in range(len(dfs))) / (
        df["sum"] - len(dfs)
    )

    # Calculate between-group variance (Vb)
    df["Vb"] = sum(df["count"][f"g{i + 1}"] * (df["mean"][f"g{i + 1}"] - overmean) ** 2 for i in range(len(dfs))) / (
        df["sum"] - len(dfs)
    )

    # Calculate the Separation score
    df["Separation"] = df["Vb"] / df["Vw"]

    # Clean up
    sep_df = df[["Feature", "Separation"]]
    sep_df.columns = sep_df.columns.droplevel(1)

    # Threshold check
    if threshold == "auto":
        subset = optimal_separation_features(sep_df)
        sep_df = sep_df.loc[sep_df["Feature"].isin(subset),]
    else:
        sep_df = sep_df.loc[(sep_df["Separation"] >= threshold) & (pd.notna(sep_df["Separation"])),]

    return sep_df

def optimal_separation_features(separation: pd.DataFrame) -> pd.Series:
    """Determines the optimal feature subset by an elbow method on their separation
    scores.

    :param separation: DataFrame containing separation scores for a number of
        features. Has 2 columns: Feature and Separation.

    :return: A Series of the feature names to keep.
    """
    scores = separation["Separation"].values
    ordered = scores[np.argsort(-scores)]
    mins = np.min(ordered)
    maxs = np.max(ordered)
    indices = np.arange(ordered.size)
    dists = ((mins - maxs) * indices) / (ordered.size - 1) + maxs - ordered
    ind = np.argmax(dists)
    thresh = ordered[ind]
    res = separation.loc[scores > thresh]["Feature"]
    return res

### ========== Clustering functions ==========
### Hierarchical clustering from cellphe.clustering
def do_hierarchical_clustering(df: pd.DataFrame, k: int, plot=False) -> np.array:
    """Performs hierarchical clustering on a given data set in order
    to identify k heterogeneous cell clusters.

    :param df: DataFrame containing only feature variables (i.e. no cell IDs or
        other labels).
    :param k: Number of clusters to identify.
    :param plot: Whether to plot the resulting dendrogram. Can help in
        identifying a suitable k.
    :return: A numpy array containing as many entries as there are rows in df.
        Each entry contains the corresponding cell label.
    """
    data_proc = scale(df.values, axis=0)
    linkage = ward(pdist(data_proc))
    labs = fcluster(linkage, t=k, criterion="maxclust")

    if plot:
        thresh = linkage[-(k - 1), 2]
        dendrogram(linkage, color_threshold=thresh)
        plt.show()
    return labs

### ========== Classification functions ==========
def classify_cells(
    train_x: pd.DataFrame, train_y: np.array, test_x: pd.DataFrame, return_probs: bool = False
) -> np.array:
    """Predicts cell class.

    Trains an xgboost classifier to predict cell labels.

    :param train_x: DataFrame containing cell features (but no labels) for the
    :param train_y: Array of labels (can be strings or integers) for the
        training set.
    :param test_x: DataFrame containing cell features (but no labels) for the
        test set.
    :param return_probs: Whether to return the probabilities as well as the
        labels. In this case the function returns a tuple of 2 items.
    :return: If return_probs is False, an N length 1D array where N is the
        number of rows in test_x. Each entry is the associated predicted label.
        If return_probs is True, then a tuple with the first item being the
        labels array, and the second being a matrix of N x M, where M is the
        number of unique classes in train_y.
    """
    # Transform labels
    le = LabelEncoder()
    train_y = le.fit_transform(train_y)

    # Fit xgboost
    mod_xgb = xgb.XGBClassifier(tree_method="hist")
    mod_xgb.fit(train_x, train_y)

    # Make predictions and convert back to the original label range
    preds_xgb_raw = mod_xgb.predict(test_x)
    preds_xgb = le.inverse_transform(preds_xgb_raw)
    
    if return_probs:
        pred_probs = mod_xgb.predict_proba(test_x)
        ret = (preds_xgb, pred_probs)
    else:
        ret = preds_xgb

    return ret