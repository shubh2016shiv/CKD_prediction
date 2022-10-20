import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.graph_objects as go

from sklearn.model_selection import cross_validate


def load_pickle_file(pickle_name):
    """
    Loading the serialized pickle file
    :return: Object
    """
    with open(pickle_name, 'rb') as f:
        return pickle.load(f)


def save_as_pickle_file(obj, pickle_file_name):
    """
    Serialize an Object in pickle format
    :return: None
    """
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(obj, f)


class FeatureSelection:
    def __init__(self, X: pd.core.frame.DataFrame,
                 y: pd.core.frame.Series):
        self.X = X
        self.y = y
        self.numerical_columns = load_pickle_file('resources/processed_data/numerical_columns.pkl')
        self.categorical_columns = load_pickle_file('resources/processed_data/categorical_columns.pkl')
        #self.categorical_columns = [column for column in self.categorical_columns if column!='classification']

    def anova_test_feature_scores(self):
        """
        Perform ANOVA testing to visually identify the relationship strength between numerical feature variables and categorical target variable
        :return: ploty figure
        """
        X = self.X[self.numerical_columns]
        fs = SelectKBest(score_func=f_classif, k=len(X.columns))

        # learn relationship from training data
        fs.fit(X, self.y)
        # transform train input data
        fs.transform(X)

        sorted_indices = np.argsort(fs.scores_)[::-1]
        feature_scores_df = pd.DataFrame.from_dict({'column_names': [X.columns[index] for index in sorted_indices],
                                                    'f_test_scores': [fs.scores_[index] for index in sorted_indices],
                                                    'p_values': [fs.pvalues_[index] for index in sorted_indices]})

        fig = px.bar(feature_scores_df, y='f_test_scores', x='column_names',
                     title="Feature Scores using ANOVA F-Test for Numerical Variables",hover_data=['p_values'],color='p_values')
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        return fig

    def chi_square_test_feature_scores(self):
        """
        Perform Chi-Square testing to visually identify the relationship strength between categorical feature variables and categorical target variable
        :return: ploty figure
        """
        X = self.X[[column for column in self.categorical_columns if column != 'classification']]
        fs = SelectKBest(score_func=chi2, k=len(X.columns))

        # learn relationship from training data
        fs.fit(X, self.y)
        # transform train input data
        fs.transform(X)

        sorted_indices = np.argsort(fs.scores_)[::-1]
        feature_scores_df = pd.DataFrame.from_dict({'column_names': [X.columns[index] for index in sorted_indices],
                                                    'chi-square_test_scores': [fs.scores_[index] for index in sorted_indices],
                                                    'p_values': [fs.pvalues_[index] for index in sorted_indices]})

        fig = px.bar(feature_scores_df, y='chi-square_test_scores', x='column_names',
                     title="Feature Scores using Chi-Square Test for Categorical Variables",hover_data=['p_values'],color='p_values')
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        return fig

    def feature_correlation_with_target(self):
        self.X.to_csv('Data_After_Feature_Engineering.csv', index=False)
        df = pd.read_csv("Data_After_Feature_Engineering.csv").iloc[:, 1:]
        corr_df = df.corr()

        lower_triangle_df = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool))
        fig = px.imshow(lower_triangle_df, color_continuous_scale='RdBu_r', origin='lower', text_auto=True,
                        aspect="auto")
        return fig

    def plot_feature_importance(self,importance, names, model_type):
        """
        Identity the important feature based on Random Forest model
        :return: plotly figure
        """
        # Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        # Create a DataFrame using a Dictionary
        data = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)

        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=True, inplace=True)

        fig = px.bar(fi_df, y='feature_names', x='feature_importance',
                     title=model_type + ' FEATURE IMPORTANCE', hover_data=['feature_importance'],
                     color='feature_importance')
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

        return fig

    def analyse_feature_importance(self):
        """
        Analysing the importance of each feature variable on target variable
        :return: plotly figure
        """
        model = RandomForestClassifier(random_state=100)
        model.fit(self.X, self.y)
        fig = self.plot_feature_importance(model.feature_importances_,self.X.columns,'RANDOM FOREST')
        return fig

    def select_features(self):
        """
        Selecting 9 best numerical features (based on ANOVA Testing) and 5 best categorical features (based on Chi-Square Testing)
        :return: list of selected columns
        """
        # For selecting 9 Best Numerical Features
        anova_k_best = SelectKBest(score_func=f_classif, k=9)
        # For selecting 5 Best Categorical Features
        chi2_k_best = SelectKBest(score_func=chi2, k=5)

        X_numerical = self.X[self.numerical_columns].copy(deep=True)
        X_categorical = self.X[[column for column in self.categorical_columns if column != 'classification']].copy(deep=True)
        y = self.y.copy(deep=True)

        anova_k_best.fit(X_numerical, y)
        chi2_k_best.fit(X_categorical, y)

        selected_num_cols = list(X_numerical.columns[list(anova_k_best.get_support())])
        selected_cat_cols = list(X_categorical.columns[list(chi2_k_best.get_support())])

        return selected_num_cols + selected_cat_cols

    def compare_performance_before_after_feat_sel(self):
        """
        Comparing the performance before and after feature selection to check if the decision to select certain features was right
        :return: pandas dataframe with performace comparison results
        """
        # Before feature selection
        X = self.X.copy(deep=True)
        y = self.y.copy(deep=True)
        model_without_selection = RandomForestClassifier(random_state=100)
        model_without_selection.fit(X, y)
        cv_results_without_selection = cross_validate(model_without_selection, X,
                                                      y)
        cv_results_without_selection = pd.DataFrame(cv_results_without_selection)

        # After feature selection
        selected_features = self.select_features()
        X_selected_features = X[selected_features].copy(deep=True)
        model_with_selection = RandomForestClassifier(random_state=100)
        cv_results_with_selection = cross_validate(
            model_with_selection, X_selected_features, y.values.ravel())
        cv_results_with_selection = pd.DataFrame(cv_results_with_selection)

        # Combining the results
        cv_results = pd.concat(
            [cv_results_without_selection, cv_results_with_selection],
            axis=1,
            keys=["Without feature selection", "With feature selection"],
        )
        # swap the level of the multi-index of the columns
        cv_results = cv_results.swaplevel(axis="columns")

        return cv_results

    def perform_feature_selection(self,X: pd.core.frame.DataFrame):
        """
        Perform the feature selection
        :return: Pandas Dataframe with selected feature variables
        """
        selected_features = self.select_features()
        save_as_pickle_file(obj=selected_features,pickle_file_name='resources/processed_data/selected_features.pkl')
        feature_selected_df = X[self.select_features()]
        return feature_selected_df
