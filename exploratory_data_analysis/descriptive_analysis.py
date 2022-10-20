import pandas as pd
import numpy as np
import scipy.stats as st
import pickle

class DescriptiveStatistics:
    def __init__(self, data: pd.core.frame.DataFrame):
        self.data = data.copy(deep=True)
        self.data['classification'] = self.data['classification'].apply(lambda x: x.replace("\t", ""))
        
        # Remove the Inconsistencies in data
        self.data['pcv'] = self.data['pcv'].replace(['\t?'], np.nan).replace(['\t43'], '43')
        self.data['pcv'] = self.data['pcv'].apply(lambda x: float(x))

        self.data['wc'] = self.data['wc'].replace(['\t?'], np.nan).replace(['\t6200'], '6200').replace(['\t8400'], '8400')
        self.data['wc'] = self.data['wc'].apply(lambda x: float(x))

        self.data['rc'] = self.data['rc'].replace(['\t?'], np.nan)
        self.data['rc'] = self.data['rc'].apply(lambda x: float(x))

        self.data['dm'] = self.data['dm'].replace(['\tno'], 'no').replace(['\tyes'], 'yes').replace([' yes'], 'yes')

        self.data['cad'] = self.data['cad'].replace(['\tno'], 'no')

        self.descriptive_stats_df = None

    def get_numerical_and_categorical_vars(self):
        """
        Get the Numerical and Categorical Vraibles from the dataframe
        :return: Tuple of Numerical and Categorical Variables as list
        """
        numerical_variables = self.data.select_dtypes(include='number').columns
        categorical_variables = [column for column in self.data.columns if column not in numerical_variables]
        with open('resources/processed_data/numerical_columns.pkl', 'wb') as f:
            pickle.dump(numerical_variables, f)

        with open('resources/processed_data/categorical_columns.pkl', 'wb') as f:
            pickle.dump(categorical_variables, f)

        return numerical_variables, categorical_variables

    def perform_descriptive_analysis(self, operations: list, columns: list):
        """
        Perform Descriptive Analysis for a given operation (like calculating mean, median, mode or 95% confidence interval) on given list of columns
        :return:
        """
        if self.descriptive_stats_df is None:
            self.descriptive_stats_df = pd.DataFrame(columns=columns)

        if "Mean" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].mean().values, 2))]],
                                                       columns=columns, index=["Mean"])])

        if "Median" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].median().values, 2))]],
                                                       columns=columns, index=["Median"])])

        if "Mode" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].mode().values[0], 2))]],
                                                       columns=columns, index=["Mode"])])

        if "Sum" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].sum().values, 2))]],
                                                       columns=columns, index=["Sum"])])

        if "Std. Deviation" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].std().values, 2))]],
                                                       columns=columns, index=["Std. Deviation"])])

        if "Variance" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].var().values, 2))]],
                                                       columns=columns, index=["Variance"])])

        if "Minimum" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].min().values, 2))]],
                                                       columns=columns, index=["Minimum"])])

        if "Maximum" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].max().values, 2))]],
                                                       columns=columns, index=["Maximum"])])

        if "Range" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in list(
                                                           np.round(self.data[columns].max().values, 2) - np.round(
                                                               self.data[columns].min().values, 2))]],
                                                       columns=columns, index=["Range"])])

        if "Quartile 1" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in list(
                                                           np.round(self.data[columns].quantile(0.25).values, 2))]],
                                                       columns=columns, index=["Quartile 1"])])

        if "Quartile 3" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in list(
                                                           np.round(self.data[columns].quantile(0.75).values, 2))]],
                                                       columns=columns, index=["Quartile 3"])])

        if "Skew" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].skew().values, 2))]],
                                                       columns=columns, index=["Skew"])])

        if "Kurtosis" in operations:
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=[[str(val) for val in
                                                              list(np.round(self.data[columns].kurt().values, 2))]],
                                                       columns=columns, index=["Kurtosis"])])

        if "95% Confidence Interval of Mean" in operations:
            confidence_intervals = []
            data_copy = self.data.copy(deep=True)
            data_copy = data_copy.fillna(self.data.mean())
            for col in columns:
                confidence_interval = tuple(np.round(
                    st.t.interval(alpha=0.95, df=len(data_copy[col]) - 1, loc=np.mean(data_copy[col]),
                                  scale=st.sem(data_copy[col])), 2))
                confidence_intervals.append(str(confidence_interval))
            self.descriptive_stats_df = pd.concat([self.descriptive_stats_df,
                                                   pd.DataFrame(
                                                       data=np.array([confidence_intervals]),
                                                       columns=columns, index=["95% Confidence Interval of Mean"])])
