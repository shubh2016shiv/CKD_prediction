import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class FeatureEngineering:
    def __init__(self, X:pd.core.frame.DataFrame):
        self.X = X
        self.numerical_column_imputer_transformer = None
        self.categorical_column_imputer_transformer = None
        self.scaler_column_transformer = None
        self.numerical_variables = None
        self.categorical_variables = None

    def get_numerical_and_categorical_columns(self):
        """
        Get Numerical and Categorical Column
        :return: Tuple of list of Numerical and Categorical Columns
        """
        numerical_variables = self.X.select_dtypes(include='number').columns
        categorical_variables = [column for column in self.X.columns if column not in numerical_variables]

        return list(numerical_variables), list(categorical_variables)

    def impute_missing_values(self, X):
        """
        Impute the missing values
        :return: Dataframe without missing values
        """
        numerical_imputer = SimpleImputer(strategy="mean")
        self.numerical_variables, self.categorical_variables = self.get_numerical_and_categorical_columns()
        # Indicate which numerical variables to impute:
        self.numerical_column_imputer_transformer = ColumnTransformer(
            [("numerical_imputer", numerical_imputer, self.numerical_variables)],
            remainder="passthrough")
        # Find the mean value per variable using the fit function:
        self.numerical_column_imputer_transformer.fit(X)

        # Replace missing data:
        X = self.numerical_column_imputer_transformer.transform(X)

        X = pd.DataFrame(
            X,
            columns=self.numerical_variables + self.categorical_variables,
        )

        categorical_imputer = SimpleImputer(strategy="most_frequent")

        # Indicate which categorical variables to impute:
        self.categorical_column_imputer_transformer = ColumnTransformer(
            [("categorical_imputer", categorical_imputer, self.categorical_variables)],
            remainder="passthrough")
        # Find the mean value per variable using the fit function:
        self.categorical_column_imputer_transformer.fit(X)

        # Replace missing data:
        X = self.categorical_column_imputer_transformer.transform(X)

        X = pd.DataFrame(
            X,
            columns= self.categorical_variables + self.numerical_variables
        )

        return X

    def normalize_data(self, X):
        """
        Normalize the data
        :return: pandas.core.frame.DataFrame
        """
        scaler = RobustScaler()
        # Indicate which variables to normalize:
        self.scaler_column_transformer = ColumnTransformer([("scaler", scaler, self.numerical_variables)],
                                                           remainder="passthrough")
        # Normalize the numerical columns
        self.scaler_column_transformer.fit(X)
        X = self.scaler_column_transformer.transform(X)
        X = pd.DataFrame(
            X,
            columns=self.numerical_variables + self.categorical_variables,
        )
        return X

    @staticmethod
    def handle_categorical_variables(X):
        """
        Static function for handling the cateegorical columns
        :return: pandas.core.frame.DataFrame
        """
        X['rbc'] = X['rbc'].map({'normal':0,'abnormal':1})
        X['pc'] = X['pc'].map({'normal':0,'abnormal':1})
        X['pcc'] = X['pcc'].map({'notpresent':0,'present':1})
        X['ba'] = X['ba'].map({'notpresent':0,'present':1})
        X['htn'] = X['htn'].map({'no':0,'yes':1})
        X['dm'] = X['dm'].map({'no':0,'yes':1})
        X['cad'] = X['cad'].map({'no':0,'yes':1})
        X['appet'] = X['appet'].map({'poor':0,'good':1})
        X['pe'] = X['pe'].map({'no':0,'yes':1})
        X['ane'] = X['ane'].map({'no':0,'yes':1})

        return X

    def perform_feature_engineering(self, X):
        """
        Perform the Feature Engineering
        :return: pandas.core.frame.DataFrame
        """
        X = self.impute_missing_values(X)
        X = self.normalize_data(X)
        X = self.handle_categorical_variables(X)
        return X
