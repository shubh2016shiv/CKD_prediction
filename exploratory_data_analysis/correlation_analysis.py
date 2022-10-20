from exploratory_data_analysis.descriptive_analysis import DescriptiveStatistics
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pointbiserialr
from utility import HelperFunctions


class CorrelationAnalysis:
    def __init__(self, data: pd.core.frame.DataFrame, target_var):
        self.data = data.copy(deep=True)

        self.data['classification'] = self.data['classification'].apply(lambda x: x.replace("\t", ""))

        self.data['pcv'] = self.data['pcv'].replace(['\t?'], np.nan).replace(['\t43'], '43')
        self.data['pcv'] = self.data['pcv'].apply(lambda x: float(x))

        self.data['wc'] = self.data['wc'].replace(['\t?'], np.nan).replace(['\t6200'], '6200').replace(['\t8400'],
                                                                                                       '8400')
        self.data['wc'] = self.data['wc'].apply(lambda x: float(x))

        self.data['rc'] = self.data['rc'].replace(['\t?'], np.nan)
        self.data['rc'] = self.data['rc'].apply(lambda x: float(x))

        self.data['dm'] = self.data['dm'].replace(['\tno'], 'no').replace(['\tyes'], 'yes').replace([' yes'], 'yes')

        self.data['cad'] = self.data['cad'].replace(['\tno'], 'no')

        descriptive_stat_class = DescriptiveStatistics(self.data)
        self.numerical_cols, self.categorical_cols = descriptive_stat_class.get_numerical_and_categorical_vars()
        self.target_var = target_var
        self.helper_function = HelperFunctions()

    def corr_between_numerical_vars(self):
        """
        Compute the correlation among the numerical variables and show correlation as heatmap diagram using three correlation techniques - Pearson, Kendall and Spearman
        :return: None
        """
        st.write("**Correlation among all Numerical Variables**")
        corr_option = st.selectbox("Choose the Correlation Type", options=["Pearson", "Spearman", "Kendall"])
        if corr_option == "Pearson":
            corr = self.data.drop(self.categorical_cols, axis=1).corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        elif corr_option == "Spearman":
            corr = self.data.drop(self.categorical_cols, axis=1).corr(method='spearman')
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        elif corr_option == "Kendall":
            corr = self.data.drop(self.categorical_cols, axis=1).corr(method='kendall')
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

    def corr_between_numerical_and_categorical(self):
        """
        Compute the correlation between selected numerical variable and categorical variable to find the potential relationship between them using statistical testing using point-biserial correlation testing technique
        :return: None
        """
        st.write("**Correlation Between Numerical and Categorical Variables**")
        numerical_var_options = st.multiselect("Select Numerical Variable", options=self.numerical_cols)
        categorical_var_options = st.multiselect("Select Categorical Variable", options=self.categorical_cols)

        #st.write(f'*{self.helper_function.write_text_to_blog('correlation_hypothesis.txt')}*')
        st.write(self.helper_function.write_text_to_blog('correlation_hypothesis.txt'))

        numerical_column_list = []
        categorical_column_list = []
        p_value_list = []
        message_list = []

        for num_col in numerical_var_options:
            for cat_col in categorical_var_options:
                if cat_col == 'rbc':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['rbc'].fillna(value=self.data['rbc'].mode().values[0]).map(
                            {'normal': 0, 'abnormal': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")
                if cat_col == 'pc':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['pc'].fillna(value=self.data['pc'].mode().values[0]).map(
                            {'normal': 0, 'abnormal': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")
                if cat_col == 'pcc':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['pcc'].fillna(value=self.data['pcc'].mode().values[0]).map(
                            {'notpresent': 0, 'present': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")
                if cat_col == 'ba':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['ba'].fillna(value=self.data['ba'].mode().values[0]).map(
                            {'notpresent': 0, 'present': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")
                if cat_col == 'htn':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['htn'].fillna(value=self.data['htn'].mode().values[0]).map({'no': 0, 'yes': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")
                if cat_col == 'dm':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['dm'].fillna(value=self.data['dm'].mode().values[0]).map({'no': 0, 'yes': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")

                if cat_col == 'cad':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['cad'].fillna(value=self.data['cad'].mode().values[0]).map({'no': 0, 'yes': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")

                if cat_col == 'appet':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['appet'].fillna(value=self.data['appet'].mode().values[0]).map(
                            {'poor': 0, 'good': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")
                if cat_col == 'pe':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['pe'].fillna(value=self.data['pe'].mode().values[0]).map({'no': 0, 'yes': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")
                if cat_col == 'ane':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['ane'].fillna(value=self.data['ane'].mode().values[0]).map({'no': 0, 'yes': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")
                if cat_col == 'classification':
                    point_biserial_corr, p_value = pointbiserialr(
                        self.data[num_col].fillna(value=self.data[num_col].mean()),
                        self.data['classification'].map({'ckd': 0, 'notckd': 1}))
                    numerical_column_list.append(num_col)
                    categorical_column_list.append(cat_col)
                    p_value_list.append(p_value)
                    message_list.append(
                        "Correlation Exists" if p_value < 0.05 else "No statistical evidence of correlation")

        # point_biserial_corr, p_value = pointbiserialr(
        #     self.data['age'].fillna(value=self.data['age'].mode().values[0]),
        #     self.data['pc'].fillna(value=self.data['pc'].mode().values[0]).map({'normal': 0, 'abnormal': 1}))

        st.markdown("""---""")

        biserial_result_df = pd.DataFrame.from_dict({'Numerical Column': numerical_column_list,
                                                     'Categorical Column': categorical_column_list,
                                                     'p value': [str(p_value) for p_value in p_value_list],
                                                     'Inference': message_list})
        if len(biserial_result_df) != 0:
            st.write("Results:")
            st.dataframe(biserial_result_df, use_container_width=True)
