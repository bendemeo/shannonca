Object-oriented API
===============================
.. module:: shannonca
.. currentmodule:: shannonca

This is a more in-depth interface to SCA, allowing the user to define and construct arbitrary embedding strategies. There are three components to SCA's workflow, each represented by a Python class:
-**Connectors** accept data in some representation, and calculate the nearest neighbors of each observation. For example, the ``MetricConnector`` with ``metric=euclidean`` computes neighbors using Euclidean distance.
-**Scorers** accept a representation of the data and a list of neighbors for each observation, and compute scores indicating, for each observation and each feature in the input representation, the degree of over- or under-expression locally. For example, the ``WilcoxonScorer`` performs a Wilcoxon rank-sum test for each feature's over- or under-expression in the neighborhood of each observation, and takes the negative log *p*-value as the score. 
-**Correctors** transform probabilities to account for multiple testing. For example, the ``FWERCorrector`` performs family-wise error rate correction. 
-**Embedders** compute low-dimensional embeddings of the input data, possibly using the above tools. For example, the ``SVDEmbedder`` projects the data onto the top right singular vectors. The ``SCAEmbedder`` first uses an ``SVDEmbedder`` to make the initial embedding, computes neighborhoods using a ``Connector`` on this representation, computes scores using a ``Scorer`` on the neighborhoods, and performs SVD on the resulting score matrix to obtain axes for projection. 

This object-oriented framework allows the user to easily swap out different strategies for base embedding generation, neighborhood construction, and score computation. The top-level API uses this framework under the hood.

Embedders
----------------------------------

An ``Embedder`` produces low-dimensional representations of data via the method ``embed``. They include ``SVDEmbedders`` and ``SCAEmbedders``. 

.. autosummary::
   :toctree: generated/
   
   embedders.Embedder
   embedders.SCAEmbedder
   embedders.SVDEmbedder

Scorers
----------------------------------

A ``Scorer`` takes a representation of data and a list of neighborhoods for each observation, and computes significance scores for each feature in the neighborhood of each observation. These can be used to compute embeddings, as the ``SCAEmbedder`` does, or simply as an alternate featurization. 

.. autosummary::
   :toctree: generated/
   
   scorers.Scorer
   scorers.WilcoxonScorer
   scorers.BinomialScorer
   scorers.TScorer
   scorers.ChunkedScorer
   
Connectors
----------------------------------

A ``Connector`` takes a representation of data and computes neighborhoods of each observation. 

.. autosummary::
   :toctree: generated/
   
   connectors.Connector
   connectors.MetricConnector

Correctors
----------------------------------

``Correctors`` accept matrices or arrays of probabilities, and correct them according to some multiple testing correction scheme. 

.. autosummary::
   :toctree: generated/
   
   correctors.FWERCorrector
