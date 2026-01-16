#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (C) 2023 Ioannis Fountoulakis, Christos Evangelidis

#    This file is part of SSA2py.

#    SSA2py is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License,
#    or any later version.

#    SSA2py is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with SSA2py.  If not, see <https://www.gnu.org/licenses/>.

# Imports
#########

import numpy as np
import os, glob, re, math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Functions
###########

def find_max(pairs, values, max_values):
    """Find maximum brightness values for each spatial location"""
    for p, v in zip(pairs, values):
        key = tuple(p)
        if key in max_values:
            max_values[key] = max(max_values[key], v)
        else:
            max_values[key] = v
    return max_values

def oneD2IMSHOW(x, y, z):
    """Interpolate scattered data onto a regular grid"""
    # Generate Grid Points
    X = np.linspace(np.min(x), np.max(x), 1000)
    Y = np.linspace(np.min(y), np.max(y), 1000)
    X, Y = np.meshgrid(X, Y)
    
    # Interpolate the scattered data onto the grid
    Z = griddata((x, y), z, (X, Y), method='cubic')
    return Z, X, Y

# Main Function
###############

def plot_depth_cross_sections(Data, evla, evlo, evdp, min_lon=None, min_lat=None, max_lon=None,
                               max_lat=None, min_depth=None, max_depth=None, rolling_window=0.5,
                               colormap='viridis', mincolor=None, maxcolor=None, cross_section='longitude',
                               filename='DepthCrossSections', outpath='.', fileformat='png', dpi=400):
    """
    Plot depth cross-sections at different time ranges in a grid layout.
    
    Similar to MaxBrightTimeStep_2 but showing depth cross-sections instead of map views.
    
    Input:
    ------
    Data: string
        Path with SSA2py output data (npy files).
    evla: float
        Event Latitude.
    evlo: float
        Event Longitude.
    evdp: float
        Event Depth (km).
    min_lon: float
        Minimum Longitude for plotting range.
    min_lat: float
        Minimum Latitude for plotting range.
    max_lon: float
        Maximum Longitude for plotting range.
    max_lat: float
        Maximum Latitude for plotting range.
    min_depth: float
        Minimum Depth (km) for plotting range.
    max_depth: float
        Maximum Depth (km) for plotting range.
    rolling_window: float
        Time window (seconds) to aggregate maximum brightness.
    colormap: string
        Matplotlib colormap name for brightness visualization.
    mincolor: float
        Minimum colormap value (brightness).
    maxcolor: float
        Maximum colormap value (brightness).
    cross_section: string
        Type of cross-section: 'longitude' (lon vs depth) or 'latitude' (lat vs depth).
    filename: string
        Base filename for output plots.
    outpath: string
        Directory to save output figures.
    fileformat: string
        Output format ('png', 'pdf', etc.).
    dpi: float
        Resolution in dots per inch.

    Output
    ------
    Saved figures showing depth cross-sections at different time ranges.
    """

    # Read Maximum Brightness to get scale
    maxBr = np.load(os.path.join(Data, "out_Max.npy"))

    if (mincolor is None) and (maxcolor is None):
        vmin = np.min(maxBr[:,0]) - (0.2*np.min(maxBr[:,0]))
        vmax = np.max(maxBr[:,0])
    else:
        vmin = mincolor
        vmax = maxcolor

    # Get all npy files (excluding out_Max.npy)
    npy_files = [file for file in glob.glob(os.path.join(Data, "out*.npy")) if "out_Max.npy" not in file]

    # Sort files by timestep
    def extract_number_from_path(path):
        match = re.search(r'out_(-?\\d+\\.\\d+)\\.npy', path)
        if match:
            return float(match.group(1))
        return -1

    npy_files = sorted(npy_files, key=extract_number_from_path)
    timesteps = list(map(extract_number_from_path, npy_files))

    # Calculate time ranges for aggregation
    timeranges = np.round(np.arange(timesteps[0], timesteps[-1], rolling_window), 2)

    new_timeranges = []
    for i in range(len(timeranges)):
        start = timeranges[i]
        if i == len(timeranges) - 1:
            end = timesteps[-1] if timeranges[i] + rolling_window > timesteps[-1] else timeranges[i] + rolling_window
        else:
            end = timeranges[i+1]
        new_timeranges.append((start, end))

    # Build subplots (3x3 grid)
    max_subs = 9
    num_figs = math.ceil(len(new_timeranges)/max_subs)
    rest = len(new_timeranges)

    for f in range(num_figs):
        fig, axs = plt.subplots(3, 3, figsize=(18, 15))
        axs = axs.flatten()
    
        loops = rest if f == num_figs - 1 else max_subs
        rest -= max_subs if f < num_figs - 1 else 0
     
        for l in range(loops):
            # Keep files in this time range
            s = f*9 + l
            start, end = new_timeranges[s]
            new_npy = []

            for npy in npy_files:
                fnum = float(npy.split('/')[-1].split('.npy')[0].split('_')[1])
                if start <= fnum <= end:
                     new_npy.append(npy)
                     
            # Find max brightness in this timestep
            max_values = {}
        
            for _npy in new_npy:
                data = np.load(_npy)
                if cross_section == 'longitude':
                    # Lon vs Depth: use columns [1 (lon), 3 (depth)]
                    spatial_coords = data[:, [1, 3]]
                else:  # latitude
                    # Lat vs Depth: use columns [2 (lat), 3 (depth)]
                    spatial_coords = data[:, [2, 3]]
                    
                values = data[:,0]  # brightness
                max_values = find_max(spatial_coords, values, max_values)
    
            # Export values from dictionary
            keysList = np.array(list(max_values.keys()))
            values = np.array(list(max_values.values()))
    
            # Interpolate to grid
            Z, x, y = oneD2IMSHOW(keysList[:,0], keysList[:,1], values)
        
            # Plot with aspect='equal' for 1:1 ratio
            im = axs[l].imshow(Z, origin='lower', 
                              extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
                              aspect='equal', cmap=colormap, interpolation='bicubic', 
                              vmin=vmin, vmax=vmax)
       
            # Add time range label
            props = dict(boxstyle='square', facecolor='white', alpha=0.9)
            axs[l].text(0.03, 0.97, str(start) + " / " + str(end) + ' s',
                       transform=axs[l].transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)

            # Set labels
            if cross_section == 'longitude':
                axs[l].set_xlabel('Longitude (°)', fontweight='bold', fontsize=12)
                axs[l].set_ylabel('Depth (km)', fontweight='bold', fontsize=12)
            else:
                axs[l].set_xlabel('Latitude (°)', fontweight='bold', fontsize=12)
                axs[l].set_ylabel('Depth (km)', fontweight='bold', fontsize=12)

            # Set plot limits
            if cross_section == 'longitude':
                if min_lon is not None and max_lon is not None:
                    axs[l].set_xlim([min_lon, max_lon])
            else:
                if min_lat is not None and max_lat is not None:
                    axs[l].set_xlim([min_lat, max_lat])
                    
            if min_depth is not None and max_depth is not None:
                axs[l].set_ylim([min_depth, max_depth])

            # Invert y-axis (depth increases downward)
            axs[l].invert_yaxis()
            axs[l].grid(True, alpha=0.3)
            
            # Plot hypocenter
            if cross_section == 'longitude':
                axs[l].plot(evlo, evdp, '*', color='red', linewidth=5, markersize=20,
                           markeredgecolor='k', markeredgewidth=1.5, alpha=0.8)
            else:
                axs[l].plot(evla, evdp, '*', color='red', linewidth=5, markersize=20,
                           markeredgecolor='k', markeredgewidth=1.5, alpha=0.8)

        
        # Add colorbar at top
        cbar_ax = fig.add_axes([0.685, 0.93, 0.20, 0.02])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Maximum Brightness', fontweight='bold', fontsize=12)
    
        # Add title
        cross_type = "Longitude" if cross_section == "longitude" else "Latitude"
        fig.suptitle(f"Depth Cross-Section ({cross_type} vs Depth) - Time Range Evolution (Fig {f+1}/{num_figs})", 
                    fontsize=16, fontweight='bold')
    
        # Remove empty subplots
        if l < 8:
            for j in range(l+1, max_subs):
                axs[j].remove()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.savefig(os.path.join(outpath, filename+'_' + str(f) +'.'+fileformat), dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return
