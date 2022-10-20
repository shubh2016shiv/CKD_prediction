import pandas as pd
from sklearn.model_selection import train_test_split


class SplitData:
    def __init__(self, X: pd.core.frame.DataFrame,
                 y: pd.core.frame.Series,
                 train_size: float = 0.8,
                 stratify: bool = True,
                 random_state: float = 100,
                 ):
        self.X = X
        self.y = y
        self.train_size = train_size
        self.stratify = stratify
        self.random_state = random_state

    def perform_split(self):
        """
        Perform the data split into 80% training and 20% test dataset in stratified way
        :return: train data, train labels, test data, test labels
        """
        if self.stratify:
            X_train, X_test, y_train, y_test \
                = train_test_split(self.X, self.y, train_size=self.train_size, random_state=self.random_state,stratify=self.y,shuffle=True)
        else:
            X_train, X_test, y_train, y_test \
                = train_test_split(self.X, self.y, train_size=self.train_size, random_state=self.random_state,shuffle=True)

        return X_train, X_test, y_train, y_test
