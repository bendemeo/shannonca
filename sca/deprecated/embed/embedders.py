from scalpel.deprecated.utils import geometric_median
from fbpca import pca
from sklearn.preprocessing import normalize as sklearn_normalize
import numpy as np
import scanpy as sc
from scipy.sparse import issparse, diags

class Embedder:

    def __init__(self, name='Embedder'):
        self.name = name
        pass

    def embed(self, adata, name=None):
        return adata

    def __mul__(self, other):
        """compose two embedders, self first"""
        return CompoundEmbedder(embedders=[self,other], name=None)

class CompoundEmbedder(Embedder):

    def __init__(self, embedders, name=None):
        if name is None:
            name = '+'.join([e.name for e in embedders])
        self.name = name

        self.embedders = embedders

    def embed(self, adata, name=None):
        if name is None:
            name = self.name

        cur_data = adata
        for e in self.embedders:
            e.embed(cur_data, name=self.name)
            print(cur_data.obsm.keys())
            cur_data = sc.AnnData(cur_data.obsm['X_'+self.name])
            print(type(cur_data))

        adata.obsm['X_'+self.name] = cur_data.X

class PCAEmbedder(Embedder):

    def __init__(self, n_pcs=50, transform='log1p', name=None):
        if name is None:
            name='PCA_{}'.format(n_pcs)

        self.n_pcs = n_pcs
        self.transform = transform
        self.name = name

    def embed(self, adata, name=None):
        if name is None:
            name = self.name
        expr = adata.X


        if self.transform == 'log1p':
            if issparse(expr):
                expr.data = np.log(1+expr.data)
            else:
                expr = np.log(1+expr)
        elif self.transform == 'binarize':
            expr = (expr > 0).astype('float')

        U, s, Vt = pca(expr, k=self.n_pcs)

        adata.obsm['X_{}'.format(name)] = U*s



class GenePCAEmbedder(Embedder):

    def __init__(self, gene_pcs=50, normalize=True, norm='l2', idf=True, binarize=True, name='GenePCAEmbedder',
                 aggregator='avg'):
        self.gene_pcs = gene_pcs
        self.normalize = normalize
        self.norm = norm
        self.name = name
        self.binarize = binarize
        self.idf = idf
        self.aggregator = aggregator # how a bunch of gene vecs are converted to a cell vec


    def embed(self, adata, name=None):

        if name is None:
            name = self.name

        if self.binarize:
            expr = (adata.X>0).astype('float')
        else:
            expr = adata.X

        if self.idf:
            idf_vals = np.log(float(expr.shape[0])/np.sum((expr > 0), axis=0))

            idf_vals = np.nan_to_num(idf_vals, nan=0, posinf=0, neginf=0)

            print(np.tile(idf_vals, (expr.shape[0], 1)).shape)
            print(expr.shape)

            expr = expr.multiply(np.tile(idf_vals, (expr.shape[0],1)))

        U, s, Vt = pca(expr.transpose(), k=self.gene_pcs)
        gene_pcs = U*s

        if self.normalize:  # project to unit sphere
            sklearn_normalize(gene_pcs, copy=False)

        result = (expr) * gene_pcs

        if self.aggregator == 'avg': # divide by genes expressed
            total_exprs = (np.array(expr.sum(axis=1))).flatten()
            inv_exprs = np.divide(1,total_exprs)

            print(diags(inv_exprs).shape)

            result = diags(inv_exprs) * result

        elif self.aggregator == 'median': # compute geometric median
            result = np.zeros((expr.shape[0], self.gene_pcs))
            row_idx = np.split(expr.indices, expr.indptr[1:-1])

            for i, idx in enumerate(row_idx):
                result[i,:] = geometric_median(gene_pcs[idx,:])




        adata.obsm['X_{}'.format(name)] = result

class GeneEmbedder(Embedder):

    def __init__(self, name='GeneEmbedder', post_pcs=50):
        self.name = name

    def embed(self, adata, name=None):
        if name is None:
            name = self.name

        xxt = (adata.X>0)*(adata.X>0).transpose()
        print(xxt.shape)

        U, s, Vt = pca(xxt, k=self.gene_pcs)
        pcs = U*s

        adata.obsm['X_{}'.format(name)] = pcs

class GeneCentroidEmbedder(Embedder):
    def __init__(self, name='centroid', gene_pcs=50, idf=False, binarize=True):
        self.name = name
        self.gene_pcs = gene_pcs
        self.idf = idf
        self.binarize = binarize

    def embed(self, adata, name=None):
        if name is None:
            name = self.name

        if self.binarize:
            expr = (adata.X>0).astype('float')
        else:
            expr = adata.X

        U, s, Vt = pca(expr.transpose(), k=self.gene_pcs)
        gene_pcs = U * s

        (expr > 0)*gene_pcs


        # 1. embed genes with PCA








