import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from .utils import metagene_loadings
import pandas as pd
import seaborn as sns


def ordered_matrixplot(d, n_genes=5, groups=None, **kwargs):
    """
    Create a matrix plot of ranked groups, with columns ordered by score instead of absolute score.
    This separates up- from down-regulated genes better, resulting in visually-cleaner plots.

    Parameters
    ----------
    d : AnnData
        Annotated data matrix.
    n_genes : int, optional
        Number of top genes to display. Default is 5.
    groups : list, optional
        List of groups to include in the plot. If None, all groups are included. Default is None.
    **kwargs : dict, optional
        Additional arguments passed to scanpy.pl.matrixplot.

    Returns
    -------
    None
    """
    # Extract top genes and their scores
    top_genes = np.stack([np.array(list(x)) for x in d.uns['rank_genes_groups']['names']][:n_genes])
    top_scores = np.stack([np.array(list(x)) for x in d.uns['rank_genes_groups']['scores']][:n_genes])

    # Order top genes by actual score, not absolute value
    ordered_top_genes = np.take_along_axis(top_genes, np.argsort(-1 * top_scores, axis=0), axis=0)

    # Extract grouping key and group names
    grouping_key = d.uns['rank_genes_groups']['params']['groupby']
    group_names = list(d.uns['rank_genes_groups']['names'].dtype.fields.keys())

    # Create a mapping of group names to ordered top genes
    ordered_top_mapping = {group_names[i]: ordered_top_genes[:, i] for i in range(len(group_names))}

    # Filter groups if specified
    if groups is not None:
        ordered_top_mapping = {k: v for k, v in ordered_top_mapping.items() if k in groups}

    # Create the matrix plot
    sc.pl.matrixplot(d, var_names=ordered_top_mapping, groupby=grouping_key, **kwargs)

def plot_metagenes(data, comps=None, key='sca', **kwargs):
    """
    Plot the loadings of metagenes for the specified components.

    Parameters
    ----------
    data : dict or AnnData
        The data containing the metagene loadings. If a dictionary, it should have a key 'loadings' with the loadings matrix.
        If an AnnData object, the loadings should be stored in `data.varm[key + '_loadings']`.
    comps : list of int, optional
        List of component indices to plot. If None, all components are plotted. Default is None.
    key : str, optional
        Key to access the loadings in the AnnData object. Default is 'sca'.
    **kwargs : dict, optional
        Additional arguments passed to the `metagene_loadings` function.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plots.
    """
    if comps is None:
        if isinstance(data, dict):
            comps = list(range(data['loadings'].shape[1]))
        else:
            comps = list(range(data.varm[key + '_loadings'].shape[1]))

    # Create subplots for each component
    fig, axs = plt.subplots(len(comps), 1)

    # Get the metagene loadings
    loadings = metagene_loadings(data, key=key, **kwargs)

    for i, comp in enumerate(comps):
        # Create a DataFrame for the loadings of the current component
        df = pd.DataFrame(loadings[comp])
        # Plot the loadings as a bar plot
        sns.barplot(data=df, x='genes', y='scores', ax=axs.flatten()[i])
        axs.flatten()[i].set_title('Component {}'.format(i + 1))

        # Rotate the x-axis labels for better readability
        for tick in axs.flatten()[i].get_xticklabels():
            tick.set_rotation(45)

    # Adjust the figure size and layout
    fig.set_size_inches(0.5 * df.shape[0], 5 * len(comps))
    plt.subplots_adjust(hspace=0.5)

    return fig
