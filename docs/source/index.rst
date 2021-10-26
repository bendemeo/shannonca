.. shannonca documentation master file, created by
   sphinx-quickstart on Thu Oct  7 14:23:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to shannonca's documentation!
=====================================
shannonca (Shannon Component analysis, or SCA) is a linear dimensionality reduction technique which captures and preserves informative features of the input data. Given input data with many features and observations, it models the enrichment or depletion of each feature in a neighborhood of each observation to produce an information score matrix, which is then distilled into a smaller set of features for downstream analysis. For a detailed description, please see our `pre-print <https://www.biorxiv.org/content/10.1101/2021.01.19.427303v2>`_. SCA integrates with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_ for use in single-cell pipelines.

In this implementation, we offer two ways to interact with SCA. The top-level API provides quick, easy access to SCA's core functionality via functions that accept high-dimensional data and output lower-dimensional embeddings. Even this API is highly customizable, but we have made a few basic decisions, like the use of SVD to generate a base embedding, that simplify the user experience. See `Quick start <https://shannonca.readthedocs.io/en/latest/quickstart.html>`_ for a simple tutorial. 

For users looking to customize and extend SCA beyond standard use cases, we offer an object-oriented API which allows more nuanced construction of dimensionality reduction tools using SCA's information-theoretic foundation. For example, the object-oriented API may be used to tweak the multiple testing correction method, base embedding strategy, or generation of k-nearest neighborhood graphs from embeddings. 

Contents
--------

.. toctree::

   quickstart
   api
   objapi
