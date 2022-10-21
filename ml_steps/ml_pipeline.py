import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
import pickle
import warnings

warnings.filterwarnings(action='ignore')


# For loading the pickle file containing selected features
def load_pickle_file(pickle_name):
    with open(pickle_name, 'rb') as f:
        return pickle.load(f)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom Feature selector class to be used in Pipeline later
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


class NullValueImputater(BaseEstimator, TransformerMixin):
    """
    Custom Null Value Imputater Class to be used in Pipeline later
    """
    def __init__(self, numerical_imputing_strategy, categorical_imputing_strategy):
        self.numerical_imputing_strategy = numerical_imputing_strategy
        self.categorical_imputing_strategy = categorical_imputing_strategy
        self.numerical_variables = None
        self.categorical_variables = None
        self._numerical_imputer = SimpleImputer(strategy=self.numerical_imputing_strategy)
        self._categorical_imputer = SimpleImputer(strategy=self.categorical_imputing_strategy)
        self._X_numeric_imputed = None
        self._X_categorical_imputed = None

    def fit(self, X, y=None):
        self.numerical_variables = X.select_dtypes(exclude='O').columns.tolist()
        self.categorical_variables = X.select_dtypes(include='O').columns.tolist()
        self._X_numeric_imputed = self._numerical_imputer.fit(X[self.numerical_variables])
        self._X_categorical_imputed = self._categorical_imputer.fit(X[self.categorical_variables])
        return self

    def transform(self, X, y=None):
        self._X_numeric_imputed = self._numerical_imputer.transform(X[self.numerical_variables])
        X_numeric_imputed_df = pd.DataFrame.from_records(self._X_numeric_imputed, columns=self.numerical_variables)

        self._X_categorical_imputed = self._categorical_imputer.fit_transform(X[self.categorical_variables])
        X_categorical_imputed_df = pd.DataFrame.from_records(self._X_categorical_imputed,
                                                             columns=self.categorical_variables)

        return pd.concat([X_numeric_imputed_df, X_categorical_imputed_df], axis=1)


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    Custom Scaler Class to be used in Pipeline later
    """
    def __init__(self):
        self.numerical_variables = None
        self.categorical_variables = None
        self._X_numeric_normalized = None
        self.normalizer = RobustScaler()

    def fit(self, X, y=None):
        self.numerical_variables = X.select_dtypes(exclude='O').columns.tolist()
        self.categorical_variables = X.select_dtypes(include='O').columns.tolist()
        self._X_numeric_normalized = self.normalizer.fit(X[self.numerical_variables])
        return self

    def transform(self, X, y=None):
        self._X_numeric_normalized = self.normalizer.transform(X[self.numerical_variables])
        X_numeric_normalized_df = pd.DataFrame.from_records(self._X_numeric_normalized,
                                                            columns=self.numerical_variables)
        X_categorical_df = X[self.categorical_variables]
        return pd.concat([X_numeric_normalized_df, X_categorical_df], axis=1)


# For encoding the categories
def handle_categorical_variables(X):
    X = X.copy(deep=True)
    X['rbc'] = X['rbc'].map({'normal': 0, 'abnormal': 1})
    X['pc'] = X['pc'].map({'normal': 0, 'abnormal': 1})
    X['pcc'] = X['pcc'].map({'notpresent': 0, 'present': 1})
    X['ba'] = X['ba'].map({'notpresent': 0, 'present': 1})
    X['htn'] = X['htn'].map({'no': 0, 'yes': 1})
    X['dm'] = X['dm'].map({'no': 0, 'yes': 1})
    X['cad'] = X['cad'].map({'no': 0, 'yes': 1})
    X['appet'] = X['appet'].map({'poor': 0, 'good': 1})
    X['pe'] = X['pe'].map({'no': 0, 'yes': 1})
    X['ane'] = X['ane'].map({'no': 0, 'yes': 1})
    return X


class CustomCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Custom category encoder class to be used in Pipeline later
    """
    def __init__(self):
        self.numerical_variables = None
        self.categorical_variables = None

    def fit(self, X, y=None):
        self.numerical_variables = X.select_dtypes(exclude='O').columns.tolist()
        self.categorical_variables = X.select_dtypes(include='O').columns.tolist()
        return self

    def transform(self, X, y=None):
        X_categorical_df = handle_categorical_variables(X[self.categorical_variables])
        X_numeric_df = X[self.numerical_variables]
        return pd.concat([X_numeric_df, X_categorical_df], axis=1)


class MachineLearningPipeline:
    def __init__(self, machine_learning_model):
        self.machine_learning_model = machine_learning_model

    def create_custom_machine_learning_pipeline(self):
        """
        Function to create Custom Machine Learning Pipeline
        :return: pipeline
        """
        # Feature Engineering Pipeline
        feature_engineering_pipeline = Pipeline([
            ('imputer',
             NullValueImputater(numerical_imputing_strategy='mean', categorical_imputing_strategy='most_frequent')),
            ('scaler',
             CustomStandardScaler()),
            ('category_encoder',
             CustomCategoryEncoder())
        ])

        # Feature selection pipeline
        selected_features = load_pickle_file('resources/processed_data/selected_features.pkl')

        feature_selection_pipeline = Pipeline([
            ('feature_selection',
             FeatureSelector(columns=selected_features))
        ])

        # Combining all the pipeline element along with machine learning model
        pipeline = Pipeline([
            ('feature_engineering', feature_engineering_pipeline),
            ('feature_selection', feature_selection_pipeline),
            ("estimator", self.machine_learning_model)
        ])

        # save machine learning pipeline as html
        with open('resources/pipeline_diagram/pipeline.html', 'w', encoding='utf-8') as f:
            f.write(estimator_html_repr(pipeline))

        return pipeline

