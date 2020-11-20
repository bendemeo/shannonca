import scanpy as sc

class Benchmarker:

    def __init__(self, datasets, embedders=[], metrics={}):
        self.datasets = datasets
        self.embedders = embedders
        self.metrics = metrics

        for d in datasets:
            for e in embedders:
                e.embed(d)

        for name,metric in metrics.items():
            for d in datasets:
                sc.pp.neighbors(d, metric=metric, key_added=name)




    def plot_umaps(self, n_neighbors=10, metrics=None, embedders=None, datasets=None, **kwargs):

        if metrics is None:
            metrics = self.metrics
        if embedders is None:
            embedders = self.embedders
        if datasets is None:
            datasets = self.datasets

        # for name,metric in metrics.values:
        #     for d in datasets:
        #         if name not in d.uns or d.uns[name]['params']['n_neighbors'] != n_neighbors:
        #             sc.pp.neighbors(d, metric=metric, use_rep='X', key_added=name, n_neighbors=n_neighbors)
        #             sc.tl.umap(d, neighbors_key=name)
        #             d.obsm['X_{}_umap'.format(name)] = d.obsm['X_umap']
        #
        # for e in embedders:
        #     for d in datasets:
        #         if e.name not in d.uns:
        #             e.embed(d)
        #             sc.tl.umap(d, neighbors_key=e.name)
        #             d.obsm['X_{}_umap'.format(e.name)] = d.obsm['X_umap']













    def umap_plots(self, ax):