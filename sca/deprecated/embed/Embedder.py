import scanpy as sc




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