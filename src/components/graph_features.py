import json
from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, check_is_fitted

from config import paths
from config.constants import GROUP_COLUMN


class GraphExtractor(TransformerMixin, BaseEstimator):
    _parameter_constraints = {"city": [str]}

    def __init__(self, city: str) -> None:
        self.city = city

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse=False, cast_to_ndarray=False)
        with open(paths.EXTERNAL_DATA_DIR / f"{self.city.lower()}_neighbors.json", "r") as fp:
            adjacency_dict = json.load(fp)

        self.graph = nx.from_dict_of_lists(adjacency_dict)
        self.node_features_ = pd.DataFrame(
            {
                "node_degree": dict(self.graph.degree),
                "degree_centrality": nx.degree_centrality(self.graph),
                "closeness_centrality": nx.closeness_centrality(self.graph),
                "betweenness_centrality": nx.betweenness_centrality(self.graph),
                "eigenvector_centrality": nx.eigenvector_centrality(
                    self.graph.subgraph(max(nx.connected_components(self.graph), key=len)),
                    max_iter=500,
                    tol=1e-6,
                ),
                "pagerank": nx.pagerank(self.graph),
                "clustering_coef": nx.clustering(self.graph),
            },
            index=self.graph.nodes,
        ).fillna(0)
        self.node_features_.index.name = GROUP_COLUMN
        self.feature_names_out_ = self.node_features_.columns.to_numpy()
        return self

    def transform(self, X):
        check_is_fitted(self, "graph")
        X = self._validate_data(X, reset=False, cast_to_ndarray=False)
        return X.join(self.node_features_, on=GROUP_COLUMN, how="left")

    def get_feature_names_out(self, input_features=None):
        return np.concatenate([input_features, self.feature_names_out_])
