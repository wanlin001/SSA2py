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

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata

def plot_depth_time_evolution(ax, BR, indicesBrig, evdepth, colormap='pink', 
                               cross_section='longitude', evlo=None, evla=None):
    """
    Plot Depth vs Time Evolution showing how brightness migrates through depth over time
    
    Arguments:
    ----------
    ax: matplotlib axis
        The axis to plot on
    BR: numpy array
        Brightness results array with columns [brightness, lon, lat, depth, time]
    indicesBrig: numpy array
        Indices of brightness data to plot
    evdepth: float
        Event depth in km
    colormap: str
        Colormap name (default: 'pink')
    cross_section: str
        Type of cross-section: 'longitude' or 'latitude'
    evlo: float
        Event longitude (required for longitude cross-section)
    evla: float
        Event latitude (required for latitude cross-section)
    
    Returns:
    --------
    scatter: matplotlib scatter object
        The scatter plot object for colorbar
    """
    
    # Extract data from brightness results
    timeSl = BR[indicesBrig, -1]  # Time slice (column 4)
    depthSl = BR[indicesBrig, 3]  # Depth slice (column 3)
    brSl = BR[indicesBrig, 0]     # Brightness slice (column 0)
    lonSl = BR[indicesBrig, 1]    # Longitude slice (column 1)
    latSl = BR[indicesBrig, 2]    # Latitude slice (column 2)
    
    # Normalize brightness for sizing
    brSl_norm = (brSl - np.nanmin(brSl)) / (np.nanmax(brSl) - np.nanmin(brSl))
    
    # Size points based on brightness
    s = [(n+1)**8 for n in brSl_norm]
    
    # Get colormap
    cm = plt.cm.get_cmap(colormap)
    
    # Create scatter plot: Time (x-axis) vs Depth (y-axis), colored by brightness
    sc = ax.scatter(timeSl, depthSl, s=s, c=brSl, cmap=cm, 
                    linewidth=1, edgecolor='black', alpha=0.8)
    
    # Plot hypocenter
    ax.plot(0, evdepth, '*', color='red', linewidth=5, markersize=18, 
            markeredgecolor='k', markeredgewidth=1.5, clip_on=False, zorder=10)
    
    # Set labels and styling
    ax.set_xlabel('Relative to Origin Time (s)', fontsize=12, labelpad=8)
    ax.set_ylabel('Depth (km)', fontsize=12, labelpad=8)
    
    # Set title based on cross-section type
    if cross_section == 'longitude' and evlo is not None:
        ax.set_title(f'Depth-Time Evolution\n(Lon={evlo:.2f}°)', 
                     fontsize=11, fontweight='bold')
    elif cross_section == 'latitude' and evla is not None:
        ax.set_title(f'Depth-Time Evolution\n(Lat={evla:.2f}°)', 
                     fontsize=11, fontweight='bold')
    else:
        ax.set_title('Depth-Time Evolution', fontsize=11, fontweight='bold')
    
    # Invert y-axis (depth increases downward)
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set time limits
    ax.set_xlim([BR[indicesBrig[0], -1], BR[indicesBrig[-1], -1]])
    
    return sc


def plot_depth_time_heatmap(ax, BR, indicesBrig, evdepth, colormap='pink',
                             cross_section='longitude', evlo=None, evla=None,
                             interpolation='cubic'):
    """
    Plot Depth vs Time Evolution as a heatmap showing brightness migration through depth
    
    Arguments:
    ----------
    ax: matplotlib axis
        The axis to plot on
    BR: numpy array
        Brightness results array with columns [brightness, lon, lat, depth, time]
    indicesBrig: numpy array
        Indices of brightness data to plot
    evdepth: float
        Event depth in km
    colormap: str
        Colormap name (default: 'pink')
    cross_section: str
        Type of cross-section: 'longitude' or 'latitude'
    evlo: float
        Event longitude (required for longitude cross-section)
    evla: float
        Event latitude (required for latitude cross-section)
    interpolation: str
        Interpolation method for griddata (default: 'cubic')
    
    Returns:
    --------
    im: matplotlib image object
        The image plot object for colorbar
    """
    
    # Extract data from brightness results
    timeSl = BR[indicesBrig, -1]  # Time slice
    depthSl = BR[indicesBrig, 3]  # Depth slice
    brSl = BR[indicesBrig, 0]     # Brightness slice
    
    # Create grid for interpolation
    time_grid = np.linspace(timeSl.min(), timeSl.max(), 200)
    depth_grid = np.linspace(depthSl.min(), depthSl.max(), 100)
    time_mesh, depth_mesh = np.meshgrid(time_grid, depth_grid)
    
    # Interpolate brightness values onto grid
    try:
        br_grid = griddata((timeSl, depthSl), brSl, 
                          (time_mesh, depth_mesh), 
                          method=interpolation, 
                          fill_value=np.nan)
    except:
        # Fall back to linear interpolation if cubic fails
        br_grid = griddata((timeSl, depthSl), brSl, 
                          (time_mesh, depth_mesh), 
                          method='linear', 
                          fill_value=np.nan)
    
    # Get colormap
    cm = plt.cm.get_cmap(colormap)
    
    # Plot heatmap
    im = ax.imshow(br_grid, origin='lower', aspect='auto', 
                   extent=[timeSl.min(), timeSl.max(), depthSl.min(), depthSl.max()],
                   cmap=cm, interpolation='bilinear', alpha=0.9)
    
    # Plot hypocenter
    ax.plot(0, evdepth, '*', color='red', linewidth=5, markersize=18, 
            markeredgecolor='k', markeredgewidth=1.5, clip_on=False, zorder=10)
    
    # Set labels and styling
    ax.set_xlabel('Relative to Origin Time (s)', fontsize=12, labelpad=8)
    ax.set_ylabel('Depth (km)', fontsize=12, labelpad=8)
    
    # Set title based on cross-section type
    if cross_section == 'longitude' and evlo is not None:
        ax.set_title(f'Depth-Time Evolution (Heatmap)\n(Lon={evlo:.2f}°)', 
                     fontsize=11, fontweight='bold')
    elif cross_section == 'latitude' and evla is not None:
        ax.set_title(f'Depth-Time Evolution (Heatmap)\n(Lat={evla:.2f}°)', 
                     fontsize=11, fontweight='bold')
    else:
        ax.set_title('Depth-Time Evolution (Heatmap)', fontsize=11, fontweight='bold')
    
    # Invert y-axis (depth increases downward)
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    return im
