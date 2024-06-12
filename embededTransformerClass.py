from sklearn.base import BaseEstimator
import pandas as pd

class EmbeddingsTransformer(BaseEstimator):
    def __init__(self, embeddings_file):
        """
        Initialize the EmbeddingsTransformer.

        Parameters:
        - embeddings_file (str): Path to the file containing pre-computed embeddings.
        """
        self.embeddings_file = embeddings_file

    def fit(self, *args, **kwargs):
        """
        Load pre-computed embeddings from the specified file.

        Returns:
        - self
        """
        self.embeddings = pd.read_pickle(self.embeddings_file)
        return self

    def transform(self, X):
        """
        Transform input data using pre-computed embeddings.

        Parameters:
        - X (pd.DataFrame): Input data with indices corresponding to embeddings.

        Returns:
        - pd.DataFrame: Transformed data using pre-computed embeddings.
        """
        return self.embeddings.loc[X.index]

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the input data.

        Parameters:
        - X (pd.DataFrame): Input data with indices corresponding to embeddings.
        - y: Ignored. This parameter exists for compatibility.

        Returns:
        - pd.DataFrame: Transformed data using pre-computed embeddings.
        """
        return self.fit().transform(X)
