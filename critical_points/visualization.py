"""
Visualization Tools for Critical Points
========================================

Plotting functions for:
1. Energy landscapes with critical points marked
2. Gradient norm fields (valleys = critical points)
3. Stability basins
4. Bifurcation diagrams
5. Eigenvalue spectra along branches

Author: Claude
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from critical_points.detector import (
    CriticalPoint, CriticalPointScan, CriticalPointType,
    compute_energy_landscape, scan_for_critical_points
)
from critical_points.stability import HessianAnalysis
from critical_points.bifurcation import BifurcationDiagram, BifurcationEvent


# Color scheme for critical point types
CP_COLORS = {
    CriticalPointType.STABLE: 'green',
    CriticalPointType.UNSTABLE: 'red',
    CriticalPointType.SADDLE: 'orange',
    CriticalPointType.DEGENERATE: 'purple',
    CriticalPointType.CENTER: 'blue',
    CriticalPointType.UNKNOWN: 'gray'
}

CP_MARKERS = {
    CriticalPointType.STABLE: 'o',      # Circle for minima
    CriticalPointType.UNSTABLE: 's',    # Square for maxima
    CriticalPointType.SADDLE: '^',      # Triangle for saddles
    CriticalPointType.DEGENERATE: 'D',  # Diamond for degenerate
    CriticalPointType.CENTER: '*',      # Star for centers
    CriticalPointType.UNKNOWN: 'x'      # X for unknown
}


def plot_energy_landscape_2d(
    X: np.ndarray,
    Y: np.ndarray,
    energy: np.ndarray,
    critical_points: Optional[List[CriticalPoint]] = None,
    agent_idx: int = 0,
    param: str = 'mu_q',
    dims: Tuple[int, int] = (0, 1),
    title: str = "Energy Landscape",
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot energy landscape as a 2D contour plot with critical points marked.

    Args:
        X, Y: Meshgrid coordinates
        energy: Energy values on grid
        critical_points: List of critical points to mark
        agent_idx: Which agent's parameters
        param: Parameter name
        dims: Which dimensions
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Contour plot
    levels = 30
    contour = ax.contourf(X, Y, energy, levels=levels, cmap=cmap)
    ax.contour(X, Y, energy, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    plt.colorbar(contour, ax=ax, label='Free Energy F')

    # Mark critical points
    if critical_points:
        for cp in critical_points:
            if agent_idx in cp.location:
                loc = cp.location[agent_idx]
                if param == 'mu_q':
                    x = loc['mu_q'][dims[0]] if len(loc['mu_q'].shape) == 1 else loc['mu_q'].flat[dims[0]]
                    y = loc['mu_q'][dims[1]] if len(loc['mu_q'].shape) == 1 else loc['mu_q'].flat[dims[1]]
                elif param == 'mu_p':
                    x = loc['mu_p'][dims[0]] if len(loc['mu_p'].shape) == 1 else loc['mu_p'].flat[dims[0]]
                    y = loc['mu_p'][dims[1]] if len(loc['mu_p'].shape) == 1 else loc['mu_p'].flat[dims[1]]
                else:
                    continue

                color = CP_COLORS.get(cp.type, 'gray')
                marker = CP_MARKERS.get(cp.type, 'x')

                ax.scatter(x, y, c=color, marker=marker, s=200, edgecolors='white',
                          linewidth=2, zorder=10, label=f'{cp.type.value} (F={cp.energy:.2f})')

    # Labels and legend
    ax.set_xlabel(f'{param}[{dims[0]}]', fontsize=12)
    ax.set_ylabel(f'{param}[{dims[1]}]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_energy_landscape_3d(
    X: np.ndarray,
    Y: np.ndarray,
    energy: np.ndarray,
    critical_points: Optional[List[CriticalPoint]] = None,
    agent_idx: int = 0,
    param: str = 'mu_q',
    dims: Tuple[int, int] = (0, 1),
    title: str = "Energy Landscape (3D)",
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 9),
    view_angle: Tuple[int, int] = (30, 45),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot energy landscape as a 3D surface with critical points marked.

    Args:
        X, Y: Meshgrid coordinates
        energy: Energy values on grid
        critical_points: List of critical points to mark
        agent_idx: Which agent
        param: Parameter name
        dims: Which dimensions
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        view_angle: (elevation, azimuth) viewing angles
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(X, Y, energy, cmap=cmap, alpha=0.8,
                           linewidth=0, antialiased=True)

    # Mark critical points
    if critical_points:
        for cp in critical_points:
            if agent_idx in cp.location:
                loc = cp.location[agent_idx]
                if param == 'mu_q':
                    x = loc['mu_q'][dims[0]] if len(loc['mu_q'].shape) == 1 else loc['mu_q'].flat[dims[0]]
                    y = loc['mu_q'][dims[1]] if len(loc['mu_q'].shape) == 1 else loc['mu_q'].flat[dims[1]]
                elif param == 'mu_p':
                    x = loc['mu_p'][dims[0]] if len(loc['mu_p'].shape) == 1 else loc['mu_p'].flat[dims[0]]
                    y = loc['mu_p'][dims[1]] if len(loc['mu_p'].shape) == 1 else loc['mu_p'].flat[dims[1]]
                else:
                    continue

                color = CP_COLORS.get(cp.type, 'gray')
                marker = CP_MARKERS.get(cp.type, 'x')

                ax.scatter([x], [y], [cp.energy], c=color, marker=marker,
                          s=200, edgecolors='white', linewidth=2, zorder=10)

    # Labels
    ax.set_xlabel(f'{param}[{dims[0]}]', fontsize=10)
    ax.set_ylabel(f'{param}[{dims[1]}]', fontsize=10)
    ax.set_zlabel('Free Energy F', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='F')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_gradient_norm_field(
    X: np.ndarray,
    Y: np.ndarray,
    gradient_norm: np.ndarray,
    critical_points: Optional[List[CriticalPoint]] = None,
    agent_idx: int = 0,
    param: str = 'mu_q',
    dims: Tuple[int, int] = (0, 1),
    title: str = "Gradient Norm Field (Valleys = Critical Points)",
    cmap: str = 'hot_r',
    figsize: Tuple[int, int] = (10, 8),
    log_scale: bool = True,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot gradient norm field. Valleys (dark regions) indicate critical points.

    Args:
        X, Y: Meshgrid coordinates
        gradient_norm: Gradient norm values on grid
        critical_points: List of critical points to mark
        agent_idx: Which agent
        param: Parameter name
        dims: Which dimensions
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        log_scale: Use log scale for gradient norm
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use log scale if requested
    if log_scale:
        data = np.log10(gradient_norm + 1e-10)
        label = 'log10(||nabla F||)'
    else:
        data = gradient_norm
        label = '||nabla F||'

    # Contour plot
    levels = 30
    contour = ax.contourf(X, Y, data, levels=levels, cmap=cmap)
    ax.contour(X, Y, data, levels=levels, colors='white', alpha=0.2, linewidths=0.3)

    plt.colorbar(contour, ax=ax, label=label)

    # Mark critical points
    if critical_points:
        for cp in critical_points:
            if agent_idx in cp.location:
                loc = cp.location[agent_idx]
                if param == 'mu_q':
                    x = loc['mu_q'][dims[0]] if len(loc['mu_q'].shape) == 1 else loc['mu_q'].flat[dims[0]]
                    y = loc['mu_q'][dims[1]] if len(loc['mu_q'].shape) == 1 else loc['mu_q'].flat[dims[1]]
                else:
                    x = loc['mu_p'][dims[0]] if len(loc['mu_p'].shape) == 1 else loc['mu_p'].flat[dims[0]]
                    y = loc['mu_p'][dims[1]] if len(loc['mu_p'].shape) == 1 else loc['mu_p'].flat[dims[1]]

                color = CP_COLORS.get(cp.type, 'lime')
                marker = CP_MARKERS.get(cp.type, 'x')

                ax.scatter(x, y, c=color, marker=marker, s=200, edgecolors='white',
                          linewidth=2, zorder=10)

    ax.set_xlabel(f'{param}[{dims[0]}]', fontsize=12)
    ax.set_ylabel(f'{param}[{dims[1]}]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_stability_basin(
    X: np.ndarray,
    Y: np.ndarray,
    basin_mask: np.ndarray,
    critical_point: CriticalPoint,
    agent_idx: int = 0,
    param: str = 'mu_q',
    dims: Tuple[int, int] = (0, 1),
    title: str = "Basin of Attraction",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot the basin of attraction of a stable critical point.

    Args:
        X, Y: Meshgrid coordinates
        basin_mask: Boolean mask of points in basin
        critical_point: The attractor
        agent_idx: Which agent
        param: Parameter name
        dims: Which dimensions
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot basin
    ax.contourf(X, Y, basin_mask.astype(float), levels=[0, 0.5, 1],
                colors=['lightgray', 'lightgreen'], alpha=0.7)
    ax.contour(X, Y, basin_mask.astype(float), levels=[0.5],
               colors='darkgreen', linewidths=2)

    # Mark critical point
    if agent_idx in critical_point.location:
        loc = critical_point.location[agent_idx]
        if param == 'mu_q':
            x = loc['mu_q'][dims[0]]
            y = loc['mu_q'][dims[1]]
        else:
            x = loc['mu_p'][dims[0]]
            y = loc['mu_p'][dims[1]]

        ax.scatter(x, y, c='green', marker='*', s=400, edgecolors='white',
                  linewidth=2, zorder=10, label='Attractor')

    ax.set_xlabel(f'{param}[{dims[0]}]', fontsize=12)
    ax.set_ylabel(f'{param}[{dims[1]}]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_bifurcation_diagram(
    diagram: BifurcationDiagram,
    observable: str = 'energy',
    agent_idx: int = 0,
    param_key: str = 'mu_q',
    dim: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot a bifurcation diagram showing how critical points change with parameter.

    Args:
        diagram: BifurcationDiagram from bifurcation scan
        observable: What to plot ('energy', 'mu_q', 'mu_p', 'stability_index')
        agent_idx: Which agent
        param_key: For 'mu_q' or 'mu_p' observable
        dim: Which dimension of mu to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Collect data from branches
    for branch in diagram.branches:
        param_vals = []
        obs_vals = []
        colors = []

        for param_val, cp in branch:
            param_vals.append(param_val)

            if observable == 'energy':
                obs_vals.append(cp.energy)
            elif observable == 'gradient_norm':
                obs_vals.append(cp.gradient_norm)
            elif observable == 'stability_index':
                obs_vals.append(cp.stability_index)
            elif observable == 'mu_q' and agent_idx in cp.location:
                mu = cp.location[agent_idx]['mu_q']
                obs_vals.append(mu.flat[dim] if mu.size > 1 else float(mu))
            elif observable == 'mu_p' and agent_idx in cp.location:
                mu = cp.location[agent_idx]['mu_p']
                obs_vals.append(mu.flat[dim] if mu.size > 1 else float(mu))
            else:
                obs_vals.append(np.nan)

            colors.append(CP_COLORS.get(cp.type, 'gray'))

        if len(param_vals) > 0:
            # Plot branch
            ax.scatter(param_vals, obs_vals, c=colors, s=30, alpha=0.7)

    # Mark bifurcations
    for bif in diagram.bifurcations:
        ax.axvline(bif.param_value, color='red', linestyle='--', alpha=0.5,
                  label=f'{bif.type.value} at {bif.param_value:.3f}')

    # Labels
    ax.set_xlabel(diagram.param_name, fontsize=12)

    if observable == 'energy':
        ylabel = 'Free Energy F'
    elif observable == 'gradient_norm':
        ylabel = '||nabla F||'
    elif observable == 'stability_index':
        ylabel = 'Stability Index (# negative eigenvalues)'
    else:
        ylabel = f'{observable}[{dim}]'

    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f'Bifurcation Diagram: {diagram.param_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend for critical point types
    legend_elements = [
        plt.scatter([], [], c=CP_COLORS[CriticalPointType.STABLE], s=50, label='Stable'),
        plt.scatter([], [], c=CP_COLORS[CriticalPointType.SADDLE], s=50, label='Saddle'),
        plt.scatter([], [], c=CP_COLORS[CriticalPointType.UNSTABLE], s=50, label='Unstable'),
    ]
    ax.legend(loc='upper right')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_eigenvalue_spectrum(
    param_values: np.ndarray,
    eigenvalue_matrix: np.ndarray,
    bifurcation_points: Optional[List[float]] = None,
    title: str = "Hessian Eigenvalue Spectrum",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot eigenvalue spectrum along a parameter branch.

    Eigenvalues crossing zero indicate bifurcations.

    Args:
        param_values: Array of parameter values
        eigenvalue_matrix: (n_params, n_eigenvalues) matrix
        bifurcation_points: Parameter values where bifurcations occur
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_eigs = eigenvalue_matrix.shape[1]
    cmap = cm.get_cmap('tab10')

    # Plot each eigenvalue branch
    for i in range(n_eigs):
        eigenvalues = eigenvalue_matrix[:, i]
        valid_mask = ~np.isnan(eigenvalues)

        if np.any(valid_mask):
            color = cmap(i % 10)
            ax.plot(param_values[valid_mask], eigenvalues[valid_mask],
                   'o-', color=color, markersize=3, linewidth=1,
                   label=f'lambda_{i}')

    # Zero line (stability boundary)
    ax.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)

    # Mark bifurcations
    if bifurcation_points:
        for bif_param in bifurcation_points:
            ax.axvline(bif_param, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Parameter', fontsize=12)
    ax.set_ylabel('Hessian Eigenvalues', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_critical_point_summary(
    scan: CriticalPointScan,
    title: str = "Critical Points Summary",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a summary figure with multiple views of critical points.

    Includes:
    1. Energy landscape with critical points
    2. Gradient norm field
    3. Energy bar chart by type
    4. Eigenvalue spectrum

    Args:
        scan: CriticalPointScan result
        title: Overall title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=figsize)

    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    params = scan.scan_parameters
    grid_range = params.get('grid_range', (-2, 2))
    resolution = params.get('grid_resolution', 20)

    x = np.linspace(grid_range[0], grid_range[1], resolution)
    y = np.linspace(grid_range[0], grid_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # 1. Energy landscape
    if scan.energy_landscape is not None:
        levels = 30
        contour1 = ax1.contourf(X, Y, scan.energy_landscape, levels=levels, cmap='viridis')
        plt.colorbar(contour1, ax=ax1, label='F')

        for cp in scan.critical_points:
            # Get position (simplified - assume first agent)
            if 0 in cp.location:
                loc = cp.location[0]
                mu = loc.get('mu_q', np.zeros(2))
                dims = params.get('dims', (0, 1))
                if len(mu.shape) == 0:
                    x_pos, y_pos = 0, 0
                else:
                    x_pos = mu.flat[dims[0]] if mu.size > dims[0] else 0
                    y_pos = mu.flat[dims[1]] if mu.size > dims[1] else 0

                color = CP_COLORS.get(cp.type, 'gray')
                ax1.scatter(x_pos, y_pos, c=color, s=150, edgecolors='white', linewidth=2)

    ax1.set_title('Energy Landscape')
    ax1.set_xlabel(f'{params.get("param", "mu")}[{params.get("dims", (0,1))[0]}]')
    ax1.set_ylabel(f'{params.get("param", "mu")}[{params.get("dims", (0,1))[1]}]')

    # 2. Gradient norm field
    if scan.gradient_field is not None:
        log_grad = np.log10(scan.gradient_field + 1e-10)
        contour2 = ax2.contourf(X, Y, log_grad, levels=30, cmap='hot_r')
        plt.colorbar(contour2, ax=ax2, label='log10(||nabla F||)')

    ax2.set_title('Gradient Norm (valleys = CPs)')
    ax2.set_xlabel(f'{params.get("param", "mu")}[{params.get("dims", (0,1))[0]}]')
    ax2.set_ylabel(f'{params.get("param", "mu")}[{params.get("dims", (0,1))[1]}]')

    # 3. Energy bar chart by type
    type_counts = {}
    type_energies = {}

    for cp in scan.critical_points:
        cp_type = cp.type.value
        if cp_type not in type_counts:
            type_counts[cp_type] = 0
            type_energies[cp_type] = []
        type_counts[cp_type] += 1
        type_energies[cp_type].append(cp.energy)

    if type_counts:
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        colors = [CP_COLORS.get(CriticalPointType(t), 'gray') for t in types]

        bars = ax3.bar(types, counts, color=colors, edgecolor='black')
        ax3.set_ylabel('Count')
        ax3.set_title('Critical Points by Type')

        # Add energy annotations
        for bar, t in zip(bars, types):
            energies = type_energies[t]
            if energies:
                mean_e = np.mean(energies)
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'F={mean_e:.2f}', ha='center', va='bottom', fontsize=8)

    # 4. Eigenvalue plot (if available)
    has_eigenvalues = any(cp.hessian_eigenvalues is not None for cp in scan.critical_points)

    if has_eigenvalues:
        for i, cp in enumerate(scan.critical_points):
            if cp.hessian_eigenvalues is not None:
                x_vals = np.arange(len(cp.hessian_eigenvalues))
                color = CP_COLORS.get(cp.type, 'gray')
                ax4.scatter(x_vals, cp.hessian_eigenvalues, c=color, s=50, alpha=0.7,
                           label=f'CP {i}' if i < 5 else None)

        ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Eigenvalue Index')
        ax4.set_ylabel('Eigenvalue')
        ax4.set_title('Hessian Eigenvalues')
        if len(scan.critical_points) <= 5:
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Hessian computed\n(call classify_all_critical_points)',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Hessian Eigenvalues')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_animation_frames(
    system,
    param_name: str,
    param_values: np.ndarray,
    output_dir: Path,
    agent_idx: int = 0,
    param: str = 'mu_q',
    dims: Tuple[int, int] = (0, 1),
    grid_range: Tuple[float, float] = (-2.0, 2.0),
    grid_resolution: int = 20
) -> List[Path]:
    """
    Create animation frames showing how energy landscape changes with parameter.

    Args:
        system: MultiAgentSystem
        param_name: Control parameter to vary
        param_values: Array of parameter values for frames
        output_dir: Directory to save frames
        agent_idx: Which agent
        param: Parameter for landscape
        dims: Which dimensions
        grid_range: Landscape range
        grid_resolution: Landscape resolution

    Returns:
        List of paths to frame images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    original_value = getattr(system.config, param_name)

    for i, param_val in enumerate(param_values):
        print(f"Frame {i+1}/{len(param_values)}: {param_name}={param_val:.4f}")

        # Set parameter
        setattr(system.config, param_name, param_val)

        # Compute landscape and critical points
        X, Y, energy = compute_energy_landscape(
            system, agent_idx, param, dims[0], dims[1],
            grid_range, grid_resolution
        )

        scan = scan_for_critical_points(
            system, agent_idx, param, dims,
            grid_range, grid_resolution,
            gradient_threshold=0.5,
            refine=True,
            verbose=False
        )

        # Plot
        fig = plot_energy_landscape_2d(
            X, Y, energy, scan.critical_points,
            agent_idx, param, dims,
            title=f'{param_name} = {param_val:.4f}'
        )

        frame_path = output_dir / f'frame_{i:04d}.png'
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        frame_paths.append(frame_path)

    # Restore parameter
    setattr(system.config, param_name, original_value)

    return frame_paths
