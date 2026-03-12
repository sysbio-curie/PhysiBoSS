## Plot cells and tracks
## Load packages
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Collection
import warnings 
from matplotlib.collections import LineCollection
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from utils import safe_divide, create_plot_title #, concentric_ring_areas
from metrics import safe_kruskal

def plot_cell_positions(data, wells_plot, time_frames_plot, plot_lims, df_droplet_boundary_all=None, hue_var='Channel', hue_order=None, position_var_unit='[µm]', 
                        cmap='tab10', legend=True, legend_outside=True, custom_title=False, time_index_var='Time Index', **kwargs):
    """
    Plot cell positions for given wells and time frames.
    
    Parameters:
    - data: DataFrame containing cell data.
    - wells_plot: List of wells to plot.
    - plot_lims: Dictionary with plot limits.
    - df_droplet_boundary_all: DataFrame containing droplet boundary information for plotting circles.
    - time_frames_plot: List of time frames to plot.
    - hue_var: Variable for coloring points.
    - hue_order: Order of hue categories.
    - position_var_unit: Unit for position variables.
    - cmap: Colormap for points.
    - legend: Whether to show legend.
    - legend_outside: Whether to place legend outside the plot.
    """
    # Set position variables and check existence
    pos_var_x = "Position X" + (" " + position_var_unit if position_var_unit else "")
    pos_var_y = "Position Y" + (" " + position_var_unit if position_var_unit else "")
    position_vars = [pos_var_x, pos_var_y]
    for colname in position_vars + [hue_var]:
        if colname not in data.columns:
            raise ValueError(f"Missing required column: {colname}")
    if time_index_var not in data.columns:
        raise ValueError(f"Missing required column: '{time_index_var}'. Please provide the correct column name for time frames in the data.")

    fig, axs = plt.subplots(len(time_frames_plot), len(wells_plot), figsize=(len(wells_plot) * 5, len(time_frames_plot) * 5))
    for row_i in range(len(wells_plot)):
        for row_j in range(len(time_frames_plot)):
            well = wells_plot[row_i]
            # print(f"Plotting {well} at time frame {time_frames_plot[row_j]}")
            
            ## Select axis
            if len(wells_plot)==1 and len(time_frames_plot)==1:
                ax_plot = axs
            elif len(wells_plot)==1:
                ax_plot = axs[row_j]
            elif len(time_frames_plot)==1:
                ax_plot = axs[row_i]
            else:
                ax_plot = axs[row_j, row_i]
            
            ## Filter data
            filt = (data['Well']==well) & (data[time_index_var]==time_frames_plot[row_j])
            if len(data.loc[filt])==0:
                print(f"No data for {well} at Time Index {time_frames_plot[row_j]}")
                continue
            df_plot = data.loc[filt]
            
            if custom_title:
                # Edit the custom title here
                # tf_to_str = {0: 'Before', 1: 'After'}
                tf_to_time = {0: 0, 1: 20} # time in hours corresponding to time frames
                title = (f"Well {well}, Time = {tf_to_time[time_frames_plot[row_j]]} hr \n"
                f"CD4: {df_plot['CD4'].unique()[0]}, CD8: {df_plot['CD8'].unique()[0]} \n"
                f"Compound added: {df_plot['Compound_added'].unique()[0]} {df_plot['Compound_conc'].unique()[0]}"
                )
                #title=(f"{well} Time frame {time_frames_plot[row_j]}, T-cells: {df_plot['CD4'].unique()[0]}\n"
                #f"BME matrix: {df_plot['BME_matrix'].unique()[0]}, PAM: {df_plot['PAM_uM'].unique()[0]} uM,\n"
                #f"MSC/Tumor load: {safe_divide(df_plot['MSC_load'].unique()[0], df_plot['Tumor_load'].unique()[0])}\n"
                #f"Compound added: {df_plot['Compound_added'].unique()[0]} {df_plot['Compound_conc'].unique()[0]}"
                #)
            else:
                title=f"{well} Time Index {time_frames_plot[row_j]}"

            ## Plot cells
            plot_cells(df_plot, hue_var=hue_var, plot_lims=plot_lims, hue_order=hue_order, title=title, cmap=cmap, 
                    legend=legend, legend_outside=legend_outside, position_vars=position_vars, ax=ax_plot, **kwargs)
            
            ## Plot circle
            # Calculate center and radius
            if df_droplet_boundary_all is not None:
                filt = (df_droplet_boundary_all['Well']==wells_plot[row_i]) 
                row_dat = df_droplet_boundary_all.loc[filt, :]
                radius = (row_dat['Major'].iloc[0]+row_dat['Minor'].iloc[0])/4
                center = (row_dat['X'].iloc[0], row_dat['Y'].iloc[0])
                
                # Draw the circle
                plot_circle(radius, center, ax_plot)
                
                # Draw a smaller circle
                plot_circle(radius*0.9, center, ax_plot, edgecolor='gray', linewidth=0.5)
            
    plt.tight_layout()

    return fig, axs

def plot_histogram_position(data, wells_plot, time_frames_plot, df_boundary_all, time_var='Time_frame', hue_var='Channel', 
                            well_var_plot='Well', figsize=(5,5), alpha=0.7, ylim=None, custom_title=False, **kwargs):
    """
    Plot histograms of cell positions relative to droplet center for specified wells and time frames.
    Parameters:
    - data: DataFrame containing cell data with 'Position r_rel' column.
    - wells_plot: List of wells to plot.
    - time_frames_plot: List of time frames corresponding to the wells.
    - df_boundary_all: DataFrame containing droplet boundary information.
    - time_var: Column name for time frames in the data.
    - well_var: Column name for wells in the data.
    - figsize: Tuple specifying the size of each subplot.
    - alpha: Transparency level for histogram bars.
    - ylim: Y-axis limit for the histograms.
    - custom_title: If True, use a custom title format.
    """
    ## Check if columns exist
    required_columns = ['Position r_rel', well_var_plot, time_var]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    fig, axs = plt.subplots(len(time_frames_plot), len(wells_plot),  figsize=(len(wells_plot) * figsize[0], len(time_frames_plot) * figsize[1])) 
    for row_i in range(len(wells_plot)):
        for row_j in range(len(time_frames_plot)):
            # Select axis
            if len(wells_plot)>1:
                ax_plot = axs[row_j, row_i]
            elif len(time_frames_plot)>1: 
                ax_plot = axs[row_j]
            else:
                ax_plot = axs
            
            # Select data
            df_data = data.loc[(data['Well']==wells_plot[row_i]) & (data[time_var]==time_frames_plot[row_j])]
            
            # Determine droplet radius
            filt = (df_boundary_all['Well']==wells_plot[row_i])
            droplet_radius = df_boundary_all.loc[filt, 'Droplet_radius'].iloc[0]

            # Set bins
            nbins=40
            bins = np.linspace(0, droplet_radius*2, nbins+1)

            ## Set plot title
            #row_filt = (overview_df['Well']==wells_plot[row_i])
            #row_dat = overview_df.loc[row_filt].iloc[0]
            
            if custom_title:
                # Edit the custom title here
                title=(f"Time = {time_frames_plot[row_j]} hr")
            #     title=(f"{wells_plot[row_i]} Time frame {time_frames_plot[row_j]}, T-cells: {row_dat['CD4']}\n"
            #     f"BME matrix: {row_dat['BME_matrix']}, PAM: {row_dat['PAM_uM']} uM,\n"
            #     f"MSC/Tumor load: {safe_divide(row_dat['MSC_load'], row_dat['Tumor_load'])}\n"
            #     f"Compound added: {row_dat['Compound_added']} {row_dat['Compound_conc']}"
            # )
            else:
                title_group_vars = [well_var_plot, time_var]
                title = create_plot_title(df_data, group_var=title_group_vars, title_vars=title_group_vars) 
                # well_str_plot = df_data[well_var_plot].iloc[0]
                # title=f"{well_str_plot}, Time frame {time_frames_plot[row_j]}"

            # Plot data
            sns.histplot(df_data, x='Position r_rel', hue='Channel', bins=bins, multiple='stack', 
                         alpha=alpha, ax=ax_plot, **kwargs)
            ax_plot.plot([droplet_radius, droplet_radius], [0, ylim], 'k-', lw=1.5)
            ax_plot.plot([0.8*droplet_radius, 0.8*droplet_radius], [0, ylim], 'k--', lw=0.8)
            ax_plot.set_xlim([0, droplet_radius*2])
            ax_plot.set_ylim([0, ylim])
            ax_plot.set_title(title, loc='center')
            ax_plot.set_xlabel('Distance to droplet center (µm)')

    if ylim is None:
        # After plotting all subplots, set a common ylim for a stacked bar plot:
        ymaxs = []
        for ax in axs:
            # For stacked bar: sum heights at each x position
            bars = ax.patches
            if bars:
                # Group bars by x position (each group is a stack)
                x_to_heights = {}
                for bar in bars:
                    x = bar.get_x()
                    width = bar.get_width()
                    # Use (x, width) as key to distinguish bars at different positions
                    key = (x, width)
                    x_to_heights.setdefault(key, []).append(bar.get_height())
                # Sum the heights for each stack
                stack_heights = [sum(hs) for hs in x_to_heights.values()]
                if stack_heights:
                    ymaxs.append(max(stack_heights))
        if ymaxs:
            common_ylim = max(ymaxs)*1.05
            for ax in axs:
                ax.set_ylim((0, common_ylim))
                ax.plot([droplet_radius, droplet_radius], [0, common_ylim], 'k-', lw=1.5)
                ax.plot([0.8*droplet_radius, 0.8*droplet_radius], [0, common_ylim], 'k--', lw=0.8)
    plt.tight_layout()
    # plt.show()

    return fig, axs

def plot_histogram_bins_inside_droplet(data, wells, time_frames_plot, overview_df, df_boundary_all, bins, 
                                       time_var='Time_frame', well_var_plot='Well', figsize=(5, 5), ylim=None, custom_title=False):
    """
    Plot histograms of cell densities in concentric rings inside the droplet for specified wells and overview IDs.

    Parameters:
    - data: DataFrame containing cell data with 'Position r_rel' column.
    - wells: List of wells to plot.
    - time_frames_plot: List of overview IDs corresponding to the wells.
    - overview_df: DataFrame containing overview information for titles.
    - df_boundary_all: DataFrame containing droplet boundary information.
    - bins: List or array of bin edges for the concentric rings.
    - figsize: Tuple specifying the size of each subplot.
    - ylim: Y-axis limit for the histograms.
    """
    _, axs = plt.subplots(len(time_frames_plot), len(wells), figsize=(len(wells) * figsize[0], len(time_frames_plot) * figsize[1])) 
    for row_i in range(len(time_frames_plot)):
        for row_j in range(len(wells)):
            # Select axis
            if len(wells)>1:
                ax_plot = axs[row_i, row_j]
            else: 
                ax_plot = axs[row_j]            

            ## Calculate ring areas
            # Determine droplet radius
            filt = (df_boundary_all['Well']==wells[row_j])
            droplet_radius = df_boundary_all.loc[filt, 'Droplet_radius'].iloc[0]
            
            ## Fix bins
            if isinstance(bins, int):
                nbins = bins
                bin_edges = np.linspace(0, droplet_radius, nbins+1)
            elif isinstance(bins, (list, np.ndarray)):
                nbins = len(bins)-1
                bin_edges = np.array(bins)
            else:
                raise Exception("bins must be either an integer or a list/array of bin edges.")
            
            # Determine ring areas        
            bin_size = (bin_edges[1] - bin_edges[0])
            # ring_areas = concentric_ring_areas(bins)

            ## Calculate densities
            df_plot = data.loc[(data['Well']==wells[row_j]) & (data[time_var]==time_frames_plot[row_i])]
            df_plot = df_plot.loc[ df_plot['Position r_rel'] < droplet_radius ]

            # Define bins
            df_plot['r_rel_bins'] = pd.cut(df_plot['Position r_rel'], bins=bin_edges)
            #df_plot['r_rel_bins']
            
            # Aggregate data by histogram bins
            df_plot_counts = df_plot.groupby(['Channel', 'r_rel_bins']).size().reset_index(name='count')
            df_plot_counts['ring_areas'] = df_plot_counts['r_rel_bins'].apply(lambda x: np.pi*(x.right**2 - x.left**2) ).astype('float')
            df_plot_counts['cell_densities'] = df_plot_counts['count']/df_plot_counts['ring_areas']
            #df_plot_counts['r_rel_bincenter'] = df_plot_counts['r_rel_bins'].apply(lambda x: x.mid)/bin_size
            df_plot_counts['r_rel_bincenter_rel'] = df_plot_counts['r_rel_bins'].apply(lambda x: x.mid).astype('float')/bin_size/nbins

            # df_plot_counts
            sns.barplot(df_plot_counts, x='r_rel_bincenter_rel', y='cell_densities', hue='Channel', ax=ax_plot)
            if ylim is not None:
                ax_plot.set_ylim([0, ylim])
            #bin_centers_rel = np.linspace(1/nbins/2, 1-1/nbins/2, nbins) # bin centers in relative units
            #ax_plot.set_xticks(np.arange(nbins), (np.arange(nbins)+1))
            edges = (np.arange(0, nbins+1))/(nbins)
            xticklabels = ['('+str(edges[i])+', '+str(edges[i+1])+')' for i in range(nbins)]
            ax_plot.set_xticks(np.arange(nbins), xticklabels, rotation=90)
            ax_plot.set_xlabel('Relative distance to droplet center')
            ax_plot.set_ylabel('Cell density ( $\\mu m^{-2}$ )')

            ## Set plot title
            row_filt = (overview_df['Well']==wells[row_j])
            row_dat = overview_df.loc[row_filt].iloc[0]
            if custom_title:
                # Edit the custom title here
                title = (
                    f"{wells[row_j]} Time frame {time_frames_plot[row_i]}\n"
                    f"T-cells: {row_dat['CD4']}\n"
                    f"BME matrix: {row_dat['BME_matrix']}, PAM: {row_dat['PAM_uM']} uM\n"
                    f"Compound added: {row_dat['Compound_added']} {row_dat['Compound_conc']}"
                )
            else:
                title_group_vars = [well_var_plot, time_var]
                title = create_plot_title(df_plot, group_var=title_group_vars, title_vars=title_group_vars) 
                # title=f"{wells[row_j]}, Time frame {time_frames_plot[row_i]}"

            ax_plot.set_title(title, loc='center')

    plt.tight_layout()
    plt.show()

def plot_circle(radius, center, ax, edgecolor='black', facecolor='none', linewidth=2, **kwargs):
    # Create a circle with specified center and radius
    circle = patches.Circle(center, radius, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, **kwargs)

    # Add the circle to the plot
    ax.add_patch(circle)

# Plot cells that are not tracks
def plot_cells(data, hue_var, plot_lims, hue_order=None, position_vars=['Position X', 'Position Y'], 
               title=None, cmap=None, legend='auto', legend_outside=False, figsize=(5,5), ax=None,  **kwargs):
    """
    Plot cell positions with coloring based on a specified variable.
    Parameters:
    - data: DataFrame containing cell data with position and hue variables.
    - hue_var: Column name for coloring the points.
    - plot_lims: Dictionary with plot limits (xmin, xmax, ymin, ymax).
    - hue_order: List specifying the order of hue categories.
    - title: Title of the plot.
    - cmap: Colormap for the points. Can be a string, list, or dictionary.
    - legend: Whether to show legend. Can be 'auto', True, or False.
    - legend_outside: Whether to place the legend outside the plot.
    - figsize: Tuple specifying the size of the plot if a new figure is created.
    - ax: Matplotlib axis to plot on. If None, a new figure and axis will be created.
    - position_vars: List of column names for position variables (default: ['Position X', 'Position Y']). The function will check for the existence of these columns in the data.
    """

    # Determine plot variables
    x_var = position_vars[0]
    y_var = position_vars[1]

    if x_var not in data.columns or y_var not in data.columns:
        raise Exception("The input data does not contain the necessary columns. Missing column: ", x_var, " or ", y_var)
    
    for colname in [x_var, y_var, hue_var]:
        if colname not in data.columns:
            raise Exception("The input data does not contain the necessary columns. Missing column: ", colname)

    # Define color map
    # cmap can be either a list or a dictionary
    if hue_order is not None:
        hue_var_list = hue_order
    else:
        hue_var_list = data[hue_var].unique()
        hue_var_list.sort() # sort hue variables, optional
        hue_order = hue_var_list

    if cmap is not None:
        if isinstance(cmap, dict):
            colormap = cmap
        elif isinstance(cmap, list):
            if (len(cmap.colors) >= len(hue_var_list)):
                colors = cmap.colors        
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color map
        colormap = {hue_var_list[i]: colors[i] for i in range(len(hue_var_list))}
    
    # Create a plot
    if not ax:
        _, ax = plt.subplots(figsize=figsize)
    if cmap=='continuous':
        sns.scatterplot(data, x=x_var, y=y_var, hue=hue_var, legend=legend, ax=ax, **kwargs)
    else:
        sns.scatterplot(data, x=x_var, y=y_var, hue=hue_var, palette=colormap, legend=legend, ax=ax, hue_order=hue_order, **kwargs)

    # Add a legend without plotting anything
    if legend and cmap!='continuous':
        for hue in hue_var_list:
            ax.plot([np.nan, np.nan], [np.nan, np.nan], color=colormap[hue], label=str(hue))
    
    if legend_outside:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Set plot limits
    if plot_lims is not None:
        ax.set_xlim(plot_lims['xmin'], plot_lims['xmax'])
        ax.set_ylim(plot_lims['ymin'], plot_lims['ymax'])
    
    # Set labels
    ax.set_aspect('equal', 'box')
    ax.set_title(title, loc='center')
    ax.set_xlabel(position_vars[0])
    ax.set_ylabel(position_vars[1])
    
    # Invert the y-axis
    ax.invert_yaxis()

    return ax

## Plot graphs vs time
def plot_graphs_vs_time(data, group_var, group_order, y_var, x_var='Time [hr]', hue_var='Channel', hue_order=None, sem_var=None, 
                        xlabel='Time [hr]', ylabel='', ylims=None, xlims=None, n_cols=6, figsize=(5, 5), custom_title=False, title_vars=None,
                        palette=None, legend_outside=False, max_line_length=30, **kwargs):
    """
    Plot graphs of a specified variable against time for different groups.
    
    Parameters:
    - data: DataFrame containing the data to plot.
    - group_var: Column name to group the data by (e.g., 'Well').
    - group_order: List specifying the order of groups to plot.
    - x_var: Column name for the x-axis variable.
    - y_var: Column name for the y-axis variable.
    - sem_var: Column name for the y-axis SEM variable.
    - hue_var: Column name for the hue variable (e.g., cell type).
    - hue_order: List specifying the order of hue_var categories (e.g., cell types).
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - ylims: Tuple specifying y-axis limits (min, max). If None, auto-scale.
    - xlims: Tuple specifying x-axis limits (min, max). If None, auto-scale.
    - n_cols: Number of columns in the subplot grid.
    - figsize: Tuple specifying the size of each subplot.
    - custom_title: If True, use a custom title format.
    - palette: Color palette for the hue variable.
    - legend_outside: Whether to place the legend outside the plot.
    - max_line_length: Maximum number of characters per line in the plot title.
    - kwargs: Additional keyword arguments for seaborn lineplot.
    """
    # Group data by group_var
    if group_var is None:
        data['all'] = 'data'  # Create a dummy group if group_var is None
        group_var = 'all'
    
    ## if isinstance(data[group_var].values.dtype, pd.CategoricalDtype):
    #     if 'NA' not in data[group_var].cat.categories:
    #         data[group_var] = data[group_var].cat.add_categories(['NA'])
    # data[group_var] = data[group_var].fillna('NA') # Replace nan values by 'NA' for grouping
    
    ## Ensure group_var is a list for uniform processing
    if isinstance(group_var, str):
        group_var_list = [group_var]
    else:
        group_var_list = list(group_var)

    for col in group_var_list:
        if pd.api.types.is_categorical_dtype(data[col]):
            if 'NA' not in data[col].cat.categories:
                data[col] = data[col].cat.add_categories(['NA'])
        data[col] = data[col].fillna('NA')

    ## Now you can safely group by group_var (string or list)
    grouped_data = data.groupby(group_var_list if len(group_var_list) > 1 else group_var_list[0], dropna=False)

    # grouped_data = data.groupby(group_var, dropna=False)
    # ============================================= copied from plot_grouped_data() function ===
    ## determine group order
    # if not isinstance(group_order, list): # group_order needs to be a list
    #     try:
    #         group_order = list(group_order)
    #         # check whether group matches with group list
    #         valid_groups = list(grouped_data.groups.keys())
    #         print(f"group_order: {group_order}")
    #         for group in group_order:
    #             if group not in valid_groups:
    #                 raise Exception(f'Group {group} not found! Please enter a valid list of groups for group_order.')
    #     except:
    #         if group_var is not None:
    #             group_order = list(data.groupby(group_var).groups.keys()) # this might include categories not in data
    #             # filter out groups that are not in data
    #             group_order = [group for group in group_order if group in grouped_data.groups]
    #             # group_order = np.unique(data[group_var]).tolist() # this does not work if group_var is a list
    #             print(f"Given group_order is empty or invalid! Resorting to automatic group order based on group_var: {group_order}!", UserWarning)
    #         else:
    #             # all data labelled under the same group
    #             group_order = ['all']
    # print(group_order)

    # Determine group order
    if not isinstance(group_order, list) or not group_order:
        # Use all categories if categorical, else use observed groups
        if isinstance(data[group_var].values.dtype, pd.CategoricalDtype):
            group_order = [cat for cat in data[group_var].cat.categories if cat in grouped_data.groups]
        else:
            group_order = list(grouped_data.groups.keys())
    else:
        # Filter out groups not present in the data
        group_order = [g for g in group_order if g in grouped_data.groups]
        if not group_order:
            raise Exception("Given group_order has no valid groups present in the data!")

    ## check hue order
    if hue_order is not None:  # Explicitly check for None
        if not isinstance(hue_order, list):
            try:
                hue_order = list(hue_order)
            except (TypeError, ValueError):
                warnings.warn("Given hue_order is invalid! Resorting to automatic hue order!", UserWarning)
                hue_order = None  # Resort to default behavior
            except:
                raise Exception("Something went wrong with hue_order!")
        # Keep only hues that are present in the data
        hue_order = [hue for hue in hue_order if hue in data[hue_var].unique()]
        # Check that all hues in the data are included in hue_order
        data_hues = set(data[hue_var].unique())  # Use a set for faster lookup
        for hue in data_hues:
            if hue not in hue_order:
                warnings.warn(f"Group '{hue}' from data not found in hue_order! It will be excluded from the plot.", UserWarning)
        # Check hue_order list validity
        #valid_hues = set(data[hue_var].unique())  # Use a set for faster lookup
        #for hue in hue_order:
        #    if hue not in valid_hues:
        #        raise Exception(f"Group '{hue}' from hue_order not found! Please enter a valid list.", UserWarning)
    else:
        hue_order = data[hue_var].unique().tolist()
        hue_order.sort()
    # =============================================
    if palette is None:
        palette_colors = sns.color_palette('tab10', n_colors=len(hue_order))
        palette = dict(zip(hue_order, palette_colors))
    if isinstance(palette, str):
        palette_colors = sns.color_palette(palette, n_colors=len(hue_order))
        palette = dict(zip(hue_order, palette_colors))
    
    n_plots = grouped_data.ngroups
    n_rows = int(np.ceil(n_plots / n_cols)) # Compute the number of rows needed
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize[0], n_rows * figsize[1]))

    if n_plots > 1:
        axs = axs.flatten()
    for idx, group_name in enumerate(group_order):
    # for idx, (group_name, df_group) in enumerate(grouped_data):
        # Check if group name exists
        if group_name not in grouped_data.groups:
            raise Exception(f"Group {group_name} not found in grouped data. Available groups: {list(grouped_data.groups.keys())}")
        else:
            # print(f"Plotting group: {group_name}")
            df_group = grouped_data.get_group(group_name).copy()

        if n_plots > 1:
            ax_plot = axs[idx]
        else:
            ax_plot = axs
        
        ## Format title string
        for col in df_group.select_dtypes(include='category').columns:
            df_group[col] = df_group[col].astype(object)
        df_group = df_group.replace('', np.nan).fillna('NA')

        # Format according to title_vars
        title = create_plot_title(df_group, group_var=group_var, title_vars=title_vars, max_line_length=max_line_length)        
        if custom_title:
            # Edit the custom title here
            title = custom_title
            # title = (f"T-cells: {df_group.iloc[0]['CD4']},\n"
            #     f"BME matrix: {df_group.iloc[0]['BME_matrix']},\n"
            #     f"MSC/Tumor load: {df_group.iloc[0]['MSC_tumor_ratio']},\n"
            #     f"{df_group.iloc[0]['Compound_added']} {df_group.iloc[0]['Compound_conc']}"
            #     )
        
        # Make plot
        sns.lineplot(df_group, x=x_var, y=y_var, hue=hue_var, hue_order=hue_order, ax=ax_plot, palette=palette, **kwargs)
        if sem_var:
            # Overlay SEM shading manually
            for _, group in enumerate(hue_order):
                sub = df_group[df_group[hue_var] == group]
                ax_plot.fill_between(sub[x_var],
                                sub[y_var] - sub[sem_var],
                                sub[y_var] + sub[sem_var],
                                color=palette[group],
                                alpha=0.15)

        ax_plot.set_title(title, loc='center')
        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel(ylabel)
        
        # Set y-axis limits
        ax_plot.set_ylim(ylims)
        # Set x-axis limits
        ax_plot.set_xlim(xlims)

        # Set legend
        if legend_outside:
            ax_plot.legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            ax_plot.legend(loc='best')

    if ylims is None:
        # After plotting all subplots, set a common ylim:
        ymins, ymaxs = [], []
        for ax in axs:
            # Get current y-limits based on the data in the axis
            lines = ax.get_lines()
            if lines:
                ydata = [line.get_ydata() for line in lines]
                ymins.append(min([min(y) for y in ydata if len(y) > 0]))
                ymaxs.append(max([max(y) for y in ydata if len(y) > 0]))
            # For bar/box plots, you may need to check ax.patches or ax.collections

        if ymins and ymaxs:
            common_ylim = (min(ymins)*0.95, max(ymaxs)*1.05)
            for ax in axs:
                ax.set_ylim(common_ylim)
   
    # Hide any empty subplots if n_plots is not a perfect multiple of n_cols
    for j in range(idx + 1, n_rows * n_cols):
      # print("Removing empty subplot at index:", j)
      fig.delaxes(axs[j])

    plt.tight_layout()
    # plt.show()

    return fig, axs

## Timelapse data (tracks)
def plot_tracks(data, hue_var, plot_lims=None, plot_relative=False, position_vars=['Position X', 'Position Y'], trackid_var='TrackID2', time_var='Time Index',
                color_by='initial_time', plot_dot=False, dot_size=36, alpha=1, title=None, cmap=None, legend=True, legend_outside=False, 
                boundary_pts=None, ax=None, **kwargs):
    """
    Plot cell tracks for an individual image.

    Args:
        data: Cell track data in the form of a DataFrame containing the columns 'TrackID2', 'Time', position variables (e.g., 'Position X', 'Position Y'), and the hue variable for coloring.
        hue_var: Column name for coloring the tracks (e.g., 'Channel').
        plot_lims: Dictionary with plot limits (xmin, xmax, ymin, ymax).
        plot_relative: If True, plot relative positions.
        alpha: Transparency of the tracks.
        color_by: 'initial_time' or 'final_time' to color the tracks.
        plot_dot: If True, plot a dot at the first time point and a cross at the last time point.
        dot_size: Size of the dot and cross.
        cmap: Colormap to color the tracks.
        legend: If True, show the legend.
        legend_outside: If True, show the legend outside the plot.
        ax: Axes of the plot.

    Returns:
        None

    Raises:
        Exception: If the input data does not contain the necessary columns.
    """
    # Check if necessary columns exist
    for colname in [trackid_var, time_var]:
        if colname not in data.columns:
            raise Exception("The input data does not contain the necessary columns. Missing column: ", colname) 
    x_var = position_vars[0]
    y_var = position_vars[1]
    if x_var not in data.columns or y_var not in data.columns:
        raise Exception("The input data does not contain the necessary columns. Missing column: ", x_var, " or ", y_var)
    
    # Sort data points
    if 'T_start' in data.columns:
        df_sorted = data.sort_values(by=['T_start', trackid_var, time_var])
    else:
        df_sorted = data.sort_values(by=[trackid_var, time_var])
    
    # Define color map
    # cmap can be either a list or a dictionary
    hue_var_list = df_sorted[hue_var].unique()
    # hue_var_list.sort() # sort hue variables, optional
    if cmap:
        if isinstance(cmap, dict):
            colormap = cmap
        elif isinstance(cmap, list):
            if (len(cmap.colors) >= len(hue_var_list)):
                colors = cmap.colors        
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color map
        colormap = {hue_var_list[i]: colors[i] for i in range(len(hue_var_list))}

    # Define styles
    ls = 'solid'
    
    # Create a line plot for each 'TrackID2' with different colors
    if ax is None:
        _, ax = plt.subplots()

    # Prepare line segments
    segments = []
    colors = []
    start_points = []
    end_points = []
    start_colors = []
    end_colors = []

    for _, group in df_sorted.groupby(trackid_var):
        x = group[x_var].values
        y = group[y_var].values
        color = colormap[group[hue_var].iloc[0]] if color_by == 'initial_time' else colormap[group[hue_var].iloc[-1]]

        # Store line segment
        segments.append(np.column_stack([x, y]))
        colors.append(color)

        # Store start & end points **without color**
        if plot_dot:
            start_points.append((x[0], y[0]))
            end_points.append((x[-1], y[-1]))
            start_colors.append(color)
            end_colors.append(color)

    
    # Convert to NumPy arrays for efficient plotting
    segments = np.array(segments, dtype=object)  # Ensures flexible shape handling
    start_points = np.array(start_points)
    end_points = np.array(end_points)

    # Use LineCollection for efficient batch plotting
    lc = LineCollection(segments, colors=colors, alpha=alpha, **kwargs)
    ax.add_collection(lc)

    # Efficiently scatter start & end points with correct colors
    if plot_dot and len(start_points) > 0:
        ax.scatter(start_points[:, 0], start_points[:, 1], s=dot_size, c=start_colors, label=None)
        ax.scatter(end_points[:, 0], end_points[:, 1], s=dot_size, c=end_colors, marker='x', label=None)

    # Plot boundary
    if boundary_pts is not None:
        ax.scatter(boundary_pts[:,0], boundary_pts[:,1], color='grey', ls='--')
        ax.plot(boundary_pts[:,0], boundary_pts[:,1], color='grey', ls='--')

    # Add a legend without plotting anything
    if legend:
        for hue in hue_var_list:
            ax.plot([np.nan, np.nan], [np.nan, np.nan], color=colormap[hue], ls=ls, label=str(hue))
    
    if legend_outside:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax.legend(loc='best')
    
    # Set plot limits
    if plot_lims is not None:
        ax.set_xlim(plot_lims['xmin'], plot_lims['xmax'])
        ax.set_ylim(plot_lims['ymin'], plot_lims['ymax'])
    
    # Set labels
    ax.set_title(title, loc='center')
    ax.set_xlabel(position_vars[0])
    ax.set_ylabel(position_vars[1])
    
    # Invert the y-axis
    ax.invert_yaxis()
 
## Plot cell tracks
def plot_tracks_grouped(data, group_var, hue_var, n_rows, n_cols, plot_lims, *, groups=None, trackid_var='TrackID2', time_var='Time Index',
                        plot_relative=False, alpha=0.8, color_by='initial_time', plot_dot=False, dot_size=36, cmap=None, 
                        legend=True, legend_outside=False, boundaries_all=None, 
                        figsize=(5,5), position_var_unit='[µm]', save_fname=None, **kwargs):
    """
    Plot cell tracks for each well in a grouped data set.

    Args:
        data: Input data formatted as pandas dataframe with columns 'TrackID2', 'Time', position variables, as well as the columns specified in 'hue_var'.
        hue_var: Name of the column of 'data' to color the tracks by.
        plot_lims: Dictionary with keys 'xmin', 'xmax', 'ymin', 'ymax' for the plot limits.
        n_rows: Number of rows of subplots.
        n_cols: Number of columns of subplots.
        # wells: List of well names to plot.
        plot_relative: If True, plot relative positions.
        alpha: Transparency of the tracks.
        color_by: 'initial_time' or 'final_time' to color the tracks.
        plot_dot: If True, plot a dot at the first time point and a cross at the last time point.
        dot_size: Size of the dot and cross.
        cmap: Colormap to color the tracks.
        legend: If True, show the legend.
        legend_outside: If True, show the legend outside the plot.
        boundaries_all: (Optional) If given, will plot the droplet boundary as a dashed line. Dictionary with well names as keys and boundary points as values.
        save_fname: File name to save the plot.

    Returns:
        None

    Raises:
        Exception: cmap keys do not correspond to group keys.
    """
    ## Handle exceptions
    if cmap:
        for group in data[hue_var].unique():
            if group not in cmap.keys():
                raise Exception(f"Key '{group}' not found in cmap!")
            
    if boundaries_all is None:
        boundary_pts = None # required for plot_tracks argument below

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
    else: 
        if data.shape[0]==0:
            raise ValueError("Input data is empty!")
        # Specify position variable names
        if not plot_relative:
            pos_var_x = "Position X" + (" " + position_var_unit if position_var_unit else "")
            pos_var_y = "Position Y" + (" " + position_var_unit if position_var_unit else "")
        else:
            pos_var_x = "Position X_rel"
            pos_var_y = "Position Y_rel"
        position_vars = [pos_var_x, pos_var_y]

        # Check required columns
        required_columns = [trackid_var, time_var, pos_var_x, pos_var_y, hue_var, group_var]
        for col in required_columns:
            if col not in data.columns:
                raise Exception(f"Input data is missing required column: {col}")
        
    ## Make plot
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize[0], n_rows * figsize[1])) 
    if n_rows * n_cols == 1:
        axs = np.array([axs])  # Convert to 1D array for consistency
    else:
        axs = axs.flatten()
    
    data[group_var] = data[group_var].fillna('NA') # Replace nan values by 'NA' for grouping
    grouped_data = data.groupby(group_var)
    if groups is None:
        groups = list(data.groupby(group_var).groups.keys())
    for i in range(len(groups)):
        plot_group = groups[i]
        if plot_group in grouped_data.groups.keys():
            if boundaries_all is not None:
                boundary_pts = boundaries_all[plot_group]
            data_well = grouped_data.get_group(plot_group).copy()
            # Plot title - customize as needed
            #plot_title=f"{plot_group}"
            plot_title=f"{plot_group} \n Number of cells: {len(data_well['TrackID2'].unique())}"  
            # plot_title=f"{plot_group} - T-cells: {data_well.iloc[0]['Condition']}"  
            # plot_title=f"{plot_group} - T-cells: {data_well.iloc[0]['Condition']} \n Droplet: {data_well.iloc[0]['Droplet_content']}"  
            plot_tracks(data_well, hue_var=hue_var, plot_lims=plot_lims, plot_relative=plot_relative, position_vars=position_vars, 
                        trackid_var=trackid_var, time_var=time_var, color_by=color_by, plot_dot=plot_dot, dot_size=dot_size, 
                        alpha=alpha, title=plot_title, cmap=cmap, legend=legend, legend_outside=legend_outside, 
                        boundary_pts=boundary_pts, ax=axs[i], **kwargs)

        # Plot boundary
        #if boundaries_all is not None:
        #    boundary_pts = boundaries_all[well]
            #axs[i].scatter(boundary_pts[:,0], boundary_pts[:,1], color='grey', ls='--')
            #axs[i].plot(boundary_pts[:,0], boundary_pts[:,1], color='grey', ls='--')

    # Hide any empty subplots if n_plots is not a perfect multiple of n_cols
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axs[j])

    plt.tight_layout()
    if save_fname:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight')
    plt.show()

## Plot grouped data
def plot_grouped_data(data, plot_var, group_var, hue_var, *, plot_type='box', group_order=None, hue_order=None, 
                      n_cols=4, palette=None, rotation=None, xlims=None, ylims=None, xlabel=None, ylabel=None, xscale=None, yscale=None, 
                      title=None, figsize=(4, 4), legend_outside=False, title_vars=None, save_fig_name=None, s_test=None, 
                      x_var=None, t_norm=1, image_roi='in_droplet', # optional parameters for specific plots
                      **kwargs):
    """
    Plot grouped data with different plot types.

    Args:
        data: Input data formatted as pandas dataframe.
        plot_var: Name of the column of 'data' to plot.
        group_var: Name of the column of 'data' to group the plots by. Can also be a list of columns.
        hue_var: Name of the column of 'data' to color the plots by. Can also be a list of columns.
        plot_type: Type of plot to generate. Options: 'box', 'hist_count', 'hist', 'scatter', 'lineplot', 'time_avg_mean_sem', 'time_avg_units', 'infiltration_count', 'infiltration_fraction', 'infiltration_rate', 'msd'.
        group_order: Order of the groups in the plot.
        hue_order: Order of the hue in the plot.
        x_var: Name of the column of 'data' to use as the x-axis variable, used 
        n_cols: Number of columns of subplots.
        palette: Color palette to use for the plot.
        rotation: Rotation of the x-axis labels.
        xlims: Tuple of the x-axis limits.
        ylims: Tuple of the y-axis limits.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        xscale: Scale for the x-axis.
        yscale: Scale for the y-axis.
        title: Title of the plot.
        figsize: Size of the figure.
        legend_outside: If True, show the legend outside the plot.
        title_vars: List of column names to include in the plot title.
        t_norm: Normalization factor for the time variable. Only used if plot_type='infiltration_rate'.
        save_fig_name: File name to save the plot.
        s_test: Statistical test to perform. Options: 'anova', 'mannwhitneyu', 'kruskal-wallis', None. If set to None, no test is performed.
        image_roi: Region of interest for the image. Options: 'in_droplet', 'out_droplet'.
        kwargs: Additional keyword arguments for seaborn plotting functions.
    
    Returns:
        axs: Axes of the plot.
    
    Raises:
        Exception: If the input data is empty.
        Exception: If plot_type is not one of the implemented plot types.
        Exception: If plot_var is not a column in the data or is otherwise not well-defined.
        Exception: If group_var is given but is not a column or list of columns in the data.
        Exception: If hue_var is not a column or list of columns in the data.
    """
    ## Check input data and variables and redefine if necessary
    # Check if data is empty
    if data.empty:
        raise Exception("The data DataFrame is empty. Ensure it is properly loaded and populated.")
    
    # Check if plot_type is an accepted type
    plot_types = ['box', 'bar', 'hist', 'hist_count', 'scatter', 'scatter_line', 'lineplot', 'time_avg_mean_sem', 
                  'infiltration_count', 'infiltration_rate', 'infiltration_fraction', 'msd'] # list of accepted plot types
    if plot_type not in plot_types:
        raise Exception(f"The plot type {plot_type} is not one of the accepted plot types.")
    
    # Check if plot_var is well-defined
    if plot_type not in ['infiltration_count', 'infiltration_fraction', 'infiltration_rate']:  # Skip these cases
        if not plot_var:
            raise Exception("plot_var cannot be None!")
        column_to_check = f"{plot_var}_Mean" if plot_type in ['time_avg_mean_sem'] else plot_var # include exception for time_avg_mean_sem
        if column_to_check not in data.columns:
            raise Exception(f"Given plot_var {column_to_check} is not a column in data. Available columns: {data.columns.tolist()}")
    
    # Check if x_var is well-defined for scatter and scatter_line
    if plot_type in ['scatter', 'scatter_line', 'lineplot', 'time_avg_mean_sem', 'infiltration_count', 'infiltration_fraction', 'infiltration_rate']:
        if not x_var:
            raise Exception(f"x_var cannot be None for plot_type {plot_type}!")
        if x_var not in data.columns:
            raise Exception(f"Given x_var {x_var} is not a column in data. Available columns: {data.columns.tolist()}")

    # Check if group_var is well defined
    if group_var is None:
        data.loc[:, 'group_var'] = 'all'
        grouped_data = data.groupby('group_var')
    else:
        # Check if group_var all present
        if isinstance(group_var, list):
            missing_group_vars = [var for var in group_var if var not in data.columns]
            if missing_group_vars:
                raise Exception(f"The following group_var columns are missing in data: {missing_group_vars}")
        elif isinstance(group_var, str):
            if group_var not in data.columns:
                raise Exception(f"The following group_var column is missing in data: {group_var}")
        else:
            raise Exception(f"The group_var variable {group_var} must be a string or list of strings.")
        
        # group data by group_var
        data[group_var] = data[group_var].fillna('NA') # Replace nan values by 'NA' for grouping
        grouped_data = data.groupby(group_var)

    # Check if hue_var is in df_group columns
    if isinstance(hue_var, str): 
        if hue_var not in data.columns:
            raise Exception(f"hue_var '{hue_var}' is not a column in data. Available columns: {data.columns.tolist()}")
    elif isinstance(hue_var, list): 
        # if group_var is a list, make a new column in data that combines the columns in the list
        if not np.all([var in data.columns for var in hue_var]):
            raise Exception(f"All elements in hue_var ({hue_var}) must be data columns. Available columns: {data.columns.tolist()}")
        else:
            hue_var_merged_string = '_'.join(hue_var)
            data[hue_var_merged_string] = data[hue_var].astype(str).agg("_".join, axis=1)
            hue_var = hue_var_merged_string
    elif hue_var is None:
        hue_var = 'hue_var'
        data = data.copy()
        data['hue_var'] = ''
    else:
        raise Exception(f"The hue_var variable {hue_var} must be a string or list of strings.")
    
    ## determine group order
    if not isinstance(group_order, list): # group_order needs to be a list
        try:
            group_order = list(group_order)
            # check whether group matches with group list
            valid_groups = list(grouped_data.groups.keys())
            print(f"group_order: {group_order}")
            for group in group_order:
                if group not in valid_groups:
                    raise Exception(f'Group {group} not found! Please enter a valid list of groups for group_order.')
        except:
            if group_var:
                group_order = list(data.groupby(group_var).groups.keys())
                # print(group_order)
            else:
                # all data labelled under the same group
                group_order = ['all']
    
    ## check hue order
    if hue_order is not None:  # Explicitly check for None
        if not isinstance(hue_order, list):
            try:
                hue_order = list(hue_order)
            except (TypeError, ValueError):
                warnings.warn("Given hue_order is invalid! Resorting to automatic hue order!", UserWarning)
                hue_order = None  # Resort to default behavior
            except:
                raise Exception("Something went wrong with hue_order!")
        # Check hue_order list validity
        valid_hues = set(data[hue_var].unique())  # Use a set for faster lookup
        for hue in hue_order:
            if hue not in valid_hues:
                raise Exception(f"Group '{hue}' from hue_order not found! Please enter a valid list.", UserWarning)

    ## prepare plot    
    if not group_var:
        n_rows = 1
        n_cols = 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize = figsize)
    else:
        n_plots = grouped_data.ngroups
        n_rows = int(np.ceil(n_plots / n_cols)) # Compute the number of rows needed
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize[0], n_rows * figsize[1]))
        axs = axs.flatten()
    
    # Plot individual graphs
    for idx, group_name in enumerate(group_order):
        # print(group_name)
        # Check if group name exists
        if group_name not in grouped_data.groups:
            Warning(f"Group {group_name} not found in grouped data.")
            continue
        else:
            df_group = grouped_data.get_group(group_name).copy()

        # subset hue_order if necessary
        if hue_var is not None:
            df_group_hue_order = df_group[hue_var].unique()
            if hue_order:
                df_group_hue_order = [hue for hue in hue_order if hue in df_group_hue_order]
        else:
            df_group_hue_order = None

        if (n_rows==1) & (n_cols==1):
            ax_plot = axs
        else:
            ax_plot = axs[idx]          

        # Plot graph
        if plot_type=='box':
            sns.boxplot(data=df_group, x=hue_var, y=plot_var, palette=palette, order=df_group_hue_order, ax=ax_plot, **kwargs)
        if plot_type=='bar':
            sns.barplot(data=df_group, x=hue_var, y=plot_var, palette=palette, order=df_group_hue_order, ax=ax_plot, **kwargs)
        elif plot_type=='hist':
            sns.histplot(data=df_group, x=plot_var, hue=hue_var, stat='probability', common_norm=False, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)    
        elif plot_type=='hist_count':
            sns.histplot(data=df_group, x=plot_var, hue=hue_var, stat='count', palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs) 
        elif plot_type=='scatter':
            sns.scatterplot(data=df_group, x=x_var, y=plot_var, hue=hue_var, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)    
        elif plot_type=='scatter_line':
            sns.scatterplot(data=df_group, x=x_var, y=plot_var, hue=hue_var, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)    
            sns.lineplot(data=df_group, x=x_var, y=plot_var, hue=None, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)
        elif plot_type=='lineplot':
            sns.lineplot(data=df_group, x=x_var, y=plot_var, hue=hue_var, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)
        elif plot_type=='time_avg_mean_sem':
            mean_var = plot_var + "_Mean"
            sem_var = plot_var + "_SEM"

            # Skip if column is missing or all NaN
            if mean_var not in df_group.columns or df_group[mean_var].isna().all():
                print(f"Warning: Data column {mean_var} for {group_name} is empty!")
            else:
                # Make sure data is numeric
                df_group[x_var] = pd.to_numeric(df_group[x_var], errors='coerce').astype(float)
                df_group[mean_var] = pd.to_numeric(df_group[mean_var], errors='coerce').astype(float)

                # Plot the mean line
                sns.lineplot(
                    data=df_group,
                    x=x_var,
                    y=mean_var,
                    hue=hue_var,
                    palette=palette,
                    hue_order=df_group[hue_var].unique(),
                    ax=ax_plot,
                    **kwargs
                )

                # Get assigned colors from seaborn
                handles, labels = ax_plot.get_legend_handles_labels()
                color_map = dict(zip(labels, [h.get_color() for h in handles]))

                # Optional: remove legend if you're re-adding later
                # ax_plot.legend_.remove()

                # Plot the SEM ribbon
                if sem_var in df_group.columns:
                    for key, sub_df in df_group.groupby(hue_var):
                        ax_plot.fill_between(
                            sub_df[x_var],
                            sub_df[mean_var] - sub_df[sem_var],
                            sub_df[mean_var] + sub_df[sem_var],
                            alpha=0.2,
                            color=color_map.get(str(key), 'gray') 
                        )
                            # color=palette[key] if isinstance(palette, dict) else None)
        #elif plot_type=='time_avg_units': # same as lineplot now, add {'units': 'TrackID2'} to kwargs
        #   sns.lineplot(data=df_group, x=x_var, y=plot_var, hue=hue_var, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)
        
        elif plot_type=='infiltration_count': # 'infiltration_count'
            # df_group_in_droplet = df_group.loc[ df_group['Image Region Name']==image_roi ]
            df_group_in_droplet = df_group.copy()
            df_tmp = df_group_in_droplet.groupby([hue_var, x_var]).size().reset_index(name='count')
            sns.lineplot(df_tmp, x=x_var, y='count', hue=hue_var, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)
            ax_plot.set_title(f"{group_name} {image_roi}", loc='center')
            s_test = None
        elif plot_type=='infiltration_rate':
            df_group_in_droplet = df_group.loc[ df_group['Image Region Name']==image_roi]
            df2 = df_group_in_droplet.groupby([hue_var, x_var]).size().to_frame(name='Values')
            cd4_ref_mean = df2.loc['CD4'].loc[0:t_norm, 'Values'].mean()
            cd8_ref_mean = df2.loc['CD8'].loc[0:t_norm, 'Values'].mean()
            df2['Infiltration_rate'] = df2.apply(lambda row: row['Values'] / cd4_ref_mean if row.name[0] == 'CD4' 
                                        else row['Values'] / cd8_ref_mean if row.name[0] == 'CD8' 
                                        else row['Values'], axis=1)
            sns.lineplot(df2, x=x_var, y='Infiltration_rate', hue=hue_var, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)
            s_test = None
        elif plot_type=='infiltration_fraction':
            df1 = df_group.groupby([hue_var, x_var]).size()
            df_group_in_droplet = df_group.loc[ df_group['Image Region Name']==image_roi ]
            df2 = df_group_in_droplet.groupby([hue_var, x_var]).size()
            df_fraction = (df2/df1).reset_index(name='Infiltration_fraction')
            sns.lineplot(df_fraction, x=x_var, y='Infiltration_fraction', hue=hue_var, palette=palette, hue_order=df_group_hue_order, ax=ax_plot, **kwargs)

        # Plot customization
        if title is None:
            title_plot = create_plot_title(df_group, group_var=group_var, title_vars=title_vars)
        else:
            title_plot = title
        ax_plot.set_title(title_plot, loc='center')

        # if not ax_plot.get_title(): # if plot has no title yet
        #     if group_var:
        #         ax_plot.set_title(group_name, loc='center')
        #     else: 
        #         ax_plot.set_title(title, loc='center')
        if ylims:
            ax_plot.set_ylim(ylims[0], ylims[1])
        if xlims:
            ax_plot.set_xlim(xlims[0], xlims[1])
        if xscale=='log':
            ax_plot.set_xscale('log', nonpositive='clip')
        if yscale=='log':
            ax_plot.set_yscale('log', nonpositive='clip')
        if rotation:
            #ax_plot.set_xticklabels(ax_plot.get_xtickslabels(), rotation=rotation) # ax_plot.get_xtickslabels()
            plt.setp(ax_plot.get_xticklabels(), rotation=rotation)

        if xlabel:
            ax_plot.set_xlabel(xlabel)
        if ylabel:
            ax_plot.set_ylabel(ylabel)

        # legend outside
        if legend_outside:
            #ax_plot.legend(loc='upper left', bbox_to_anchor=(1, 1))
            sns.move_legend(ax_plot, "upper left", bbox_to_anchor=(1, 1))

        # Perform significance test on group data, comparing subgroups labelled by hue_var
        if s_test=='kruskal-wallis':
            print(f"Kruskal-Wallis test on {plot_var}, for the data set {group_name}, comparing groups labelled by {hue_var}.")
            result = safe_kruskal(df_group, hue_var, plot_var)
            print(result)
        elif s_test=='anova':
            print(f"ANOVA test on {plot_var}, for the data set {group_name}, comparing groups labelled by {hue_var}.")
            print( stats.f_oneway(*[list(group) for _, group in df_group.groupby(hue_var)[plot_var]]) )
        elif s_test=='mannwhitneyu':
            print(f"Mann-Whitney U test on {plot_var}, for the data set {group_name}, comparing groups labelled by {hue_var}.")
            print( stats.mannwhitneyu(*[list(group) for _, group in df_group.groupby(hue_var)[plot_var]]) )
    
    # Hide any empty subplots if n_plots is not a perfect multiple of n_cols
    for j in range(idx + 1, n_rows * n_cols):
       fig.delaxes(axs[j])

    plt.tight_layout()
    if save_fig_name:
        print('Saving as: ', save_fig_name)
        plt.savefig(save_fig_name, bbox_inches='tight')

    return axs
    #plt.show()

## Plot heatmap of scaled data
def scale_data_plot_heatmap(
        data: pd.DataFrame, 
        vars_plot_all: str | Collection[str], 
        group_var: str | Collection[str], 
        scaling_choice: str = 'standardized', 
        stat_choice: str = 'mean', 
        row_scaling: str | None = None,
        plot_title: str = 'Heatmap of Track Metrics',
        save_fig_name: str | None = None,
        sort_labels: bool = False,
        col_cluster: bool = True,
        row_cluster: bool = True,
        cbar_pos: tuple = (-0.05, .5, .02, .3),
        transpose: bool = False,
        plot: bool = True,
        verbose: bool = False,
        **kwargs):
    
    """
    Scale the data and plot a heatmap of the summary statistics.

    Args:
        data: Input data formatted as pandas dataframe.
        vars_plot_all: List of variables to plot as rows of the heatmap.
        group_var: Name of the column of 'data' to group the data by. Columns of the heatmap will correspond to different groups.
        scaling_choice: Type of scaling to perform. Options: 'standardized', 'robust_scaled'.
        stat_choice: Type of statistic to calculate. Options: 'mean', 'median'.
        row_scaling: Type of scaling to perform row-wise. Options: 'minmax', 'robust', None.
        plot_title: Title of the plot.
        save_fig_name: File name to save the plot.     
        sort_labels: If True, sort the labels of the heatmap.
        col_cluster: If True, cluster the columns of the heatmap.
        row_cluster: If True, cluster the rows of the heatmap.  
        cbar_pos: Position of the colorbar.
        transpose: If True, transpose the heatmap.
        kwargs: Additional keyword arguments for seaborn clustermap function.
    
    Returns:
        None
    """
    ## Rescale data
    # df_data_tracks_scaled = data.copy()
    scaled_cols = []
    for var in vars_plot_all:
        # Check if variable is in data
        if var not in data.columns:
            raise Exception(f"Variable {var} not found in data. Available columns: {data.columns.tolist()}")
        # Check if variable is numeric
        if not pd.api.types.is_numeric_dtype(data[var]):
            raise Exception(f"Variable {var} is not numeric. Available columns: {data.columns.tolist()}")
        
        # Standardization (Z-score normalization)
        if scaling_choice == 'standardized':
            if verbose:
                print(f"Standardizing variable {var}...")
            scaler = StandardScaler()
            # data[var + '_standardized'] = 
            scaled = scaler.fit_transform(data[[var]])
            scaled_cols.append(pd.DataFrame(scaled, columns=[var + '_standardized']))
        elif scaling_choice == 'robust_scaled':
            if verbose:
                print(f"Robust scaling variable {var}...")
            scaler = RobustScaler()
            # data[var + '_robust_scaled'] =
            scaled = scaler.fit_transform(data[[var]])
            scaled_cols.append(pd.DataFrame(scaled, columns=[var + '_robust_scaled']))
        elif scaling_choice == 'no_scaling':
            if verbose:
                print(f"No scaling applied to variable {var}...")
            #data[var + '_no_scaling'] = data[var]
            scaled = data[[var]].values
            scaled_cols.append(pd.DataFrame(scaled, columns=[var + '_no_scaling']))
            # print(f"No scaling applied to {var}.")
        else:
            raise Exception(f"Invalid scaling choice: {scaling_choice}. Must be 'standardized' or 'robust_scaled'.")
    
    # Combine the scaled columns with the original data
    scaled_cols_df = pd.concat(scaled_cols, axis=1)
    data = pd.concat([data.reset_index(drop=True), scaled_cols_df.reset_index(drop=True)], axis=1)

    # Check that the data was merged correctly
    if data.shape[0] != scaled_cols_df.shape[0]:
        raise Exception("Merging failed: row counts do not match.")

    # Replace nan values by 'NA' for grouping 
    data[group_var] = data[group_var].fillna('NA') 

    # Display the first few rows to check the results
    #print(data.shape)
    #print(data.head())

    ## Calculate statistic
    df_stats = pd.DataFrame() # Make a new dataframe to store the results
    for var_plot in vars_plot_all:
        var_name = f"{var_plot}_{scaling_choice}_{stat_choice}"
        if stat_choice == 'median':
            df_stats[var_name] = data.groupby(group_var, sort=False)[f"{var_plot}_{scaling_choice}"].median()
        elif stat_choice == 'mean':
            df_stats[var_name] = data.groupby(group_var, sort=False)[f"{var_plot}_{scaling_choice}"].mean()
    
    ## Row-wise scaling
    if row_scaling=='minmax':
        df_stats[:] = MinMaxScaler().fit_transform(df_stats)
    elif row_scaling=='robust':
        df_stats[:] = RobustScaler().fit_transform(df_stats)
    elif verbose:
        print("No row scaling applied.")
    
    ## Transpose the DataFrame to have groups as columns and variables as rows and add a small noise to avoid singular matrix
    df_stats = (df_stats + np.random.normal(0, 1e-8, df_stats.shape)).T

    ## Sort values
    if sort_labels:
        if isinstance(group_var, str):
            df_stats.sort_values(by=group_var, ascending=True, inplace=True)
        elif isinstance(group_var, list):
            raise Exception(f"Sorting for lists not yet implemented! Group_var is {group_var}.")
        else:
            raise Exception(f"Invalid group_var: {group_var}. Must be a string or list of strings.")
    
    # print(df_stats.columns)
    if (df_stats.shape[0] < 2) | (df_stats.shape[1] < 2):
        print(f"Results DataFrame too small for clustering. Shape: {df_stats.shape}")
        col_cluster = False
        row_cluster = False

    if transpose:
        df_stats = df_stats.T

    ## Plot a heatmap of the summary statistics (new - seaborn)
    if plot:
        g=sns.clustermap(
            df_stats,
            method='average',       # or 'average', 'single', 'complete'
            metric='correlation',  # or 'correlation', 'cosine', etc.
            cmap='viridis',      # or any matplotlib colormap
            col_cluster=col_cluster,
            row_cluster=row_cluster,  
            cbar_pos=cbar_pos,  
            **kwargs,
        )
        g.figure.suptitle(plot_title, fontsize=14, x=0.5, y=1.02)
        if save_fig_name:
            plt.savefig(save_fig_name, bbox_inches='tight')
        plt.show()

        ## Plot a heatmap of the summary statistics (old - matplotlib)
        # _, axs = plt.subplots(figsize=(12, 8))
        # sns.heatmap(df_stats.set_index(group_var).T, cmap='viridis', annot=False, fmt=".2f", cbar=True, ax=axs, **kwargs)
        # plt.title(plot_title)
        # if save_fig_name:
        #     plt.savefig(save_fig_name, bbox_inches='tight')
        # plt.show()

    return df_stats