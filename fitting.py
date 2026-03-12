import matplotlib.pyplot as plt
import seaborn as sns

from track_metrics import compute_msd, compute_displacement_autocorrelation, fit_autocorr, fit_msd, exp_decay, msd_prw_D, msd_prw_v


def compute_motility_correction_factor(data_sample, time_step, max_time, objective_speed=1.0, objective_persistence_time=10.0, verbose=False):
    """
    Compute a correction factor for motility metrics based on the average speed of the cells.
    
    Parameters:
    - data: DataFrame containing cell tracking data with position and time information.
    
    Returns:
    - correction_speed: A scalar value to correct cell speed
    - correction_persistence: A scalar value to correct cell persistence time
    """

    from fitting import fit_persistence, fit_speed

    persistence = fit_persistence(data_sample, time_step, max_time, verbose=verbose)
    speed = fit_speed(data_sample, time_step, persistence, verbose=verbose)
    
    return objective_speed / speed, objective_persistence_time / persistence
    

def fit_persistence(data_sample, time_step, max_time, verbose=False):
    
    
    condition_vars = ['cell_type']
    # Compute autocorrelation for the sample data
    autocorr_df = compute_displacement_autocorrelation(data_sample, condition_vars, max_lag=int(max_time / time_step))

    # Add lag in minutes
    autocorr_df['Lag [min]'] = autocorr_df['Lag (time frames)'] * time_step
    autocorr_df
    
    ## Apply fit
    tau_p_estimated = fit_autocorr(autocorr_df['Autocorrelation'].values, verbose=verbose)
    tau_p_estimated_units = tau_p_estimated * time_step
    if verbose:
        print(f"Estimated persistence time τ_p = {tau_p_estimated} time frames, or {tau_p_estimated_units} minutes")

    ## Plot autocorrelation with fitted curve
    autocorr_df['Fitted'] = exp_decay(autocorr_df['Lag [min]'], tau_p_estimated_units)

    if verbose:
        plt.figure()
        sns.lineplot(data=autocorr_df, x='Lag [min]', y='Autocorrelation', hue='Group', errorbar=None)
        sns.lineplot(data=autocorr_df, x='Lag [min]', y='Fitted', color='black', linestyle='--', label='Fitted Exp Decay')
        # plt.yscale('log')
        plt.show()
        
    return tau_p_estimated_units

def fit_speed(data_sample, time_step, persistence, verbose=False):
    # Metadata variables to group by (irrelevant for single sample, but needed for function)
    condition_vars = ['cell_type']

    # Compute MSD
    msd_all = compute_msd(data_sample.copy(), condition_vars, time_frame_col='time', displacement_col='Displacement^2')

    # Add time in minutes
    msd_all['Time [min]'] = (msd_all['time'])# * time_step
    msd_all['MSD'] = msd_all['MSD'].cumsum()
    # plt.plot(msd_all['Time [min]'], msd_all['MSD'], '-', label='Data')
    ## Apply fit
    d = 2
    D_fit_units = fit_msd(msd_all['Time [min]'].values, msd_all['MSD'].values, persistence, d=d, fit_param = 'D', verbose=verbose)
    v_fit_units = fit_msd(msd_all['Time [min]'].values, msd_all['MSD'].values, persistence, d=d, fit_param = 'v', verbose=verbose)

    # Check that the two fits are consistent (D from v should match D from D)
    D_from_v = v_fit_units**2 * persistence / d
    if verbose:
        print(f"Estimated D from D fit: {D_fit_units}")
        print(f"Estimated D from v fit: {D_from_v}")


    ## Plot MSD with fitted curve
    fitted_msd = msd_prw_D(msd_all['Time [min]'].values, D_fit_units, persistence, d=2)
    fitted_msd_v = msd_prw_v(msd_all['Time [min]'].values, v_fit_units, persistence)

    if verbose:
        plt.figure()
        sns.lineplot(data=msd_all, x='Time [min]', y='MSD', hue='cell_type', marker='o', errorbar=None)
        sns.lineplot(x=msd_all['Time [min]'], y=fitted_msd, color='black', linestyle='--', label='Fitted PRW (D)')
        # sns.lineplot(x=msd_all['Time [min]'], y=fitted_msd_v, color='red', linestyle='--', label='Fitted PRW (v)')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlim([1,100])
        # plt.ylim([1,150])
        plt.show()
    
    return D_fit_units