import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt
from exploratory_data_analysis.descriptive_analysis import DescriptiveStatistics

st.set_option('deprecation.showPyplotGlobalUse', False)


class Visualization:
    def __init__(self, data: pd.core.frame.DataFrame):
        self.data = data.copy(deep=True)
        self.data['classification'] = self.data['classification'].apply(lambda x: x.replace("\t",""))
        self.data['pcv'] = self.data['pcv'].replace(['\t?'], np.nan).replace(['\t43'], '43')
        self.data['pcv'] = self.data['pcv'].apply(lambda x: float(x))

        self.data['wc'] = self.data['wc'].replace(['\t?'], np.nan).replace(['\t6200'], '6200').replace(['\t8400'],
                                                                                                       '6200')
        self.data['wc'] = self.data['wc'].apply(lambda x: float(x))

        self.data['rc'] = self.data['rc'].replace(['\t?'], np.nan)
        self.data['rc'] = self.data['rc'].apply(lambda x: float(x))

        self.data['dm'] = self.data['dm'].replace(['\tno'], 'no').replace(['\tyes'], 'yes').replace([' yes'], 'yes')

        self.data['cad'] = self.data['cad'].replace(['\tno'], 'no')

        descriptive_stat_class = DescriptiveStatistics(self.data)
        self.numerical_cols, self.categorical_cols = descriptive_stat_class.get_numerical_and_categorical_vars()

    def show_histogram(self, columns):
        def plot_histogram_as_subplots(__columns, color=None):
            i = 0
            while i < len(__columns):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if i >= len(__columns):
                        break

                    if color is None:
                        fig = px.histogram(self.data, x=columns[i])
                        j.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.histogram(self.data, x=columns[i],color=color)
                        j.plotly_chart(fig, use_container_width=True)

                    i += 1

        if 'classification' in columns:
            plot_histogram_as_subplots(columns,color='classification')
        else:
            plot_histogram_as_subplots(columns)

    def calculateOutliersInliers(self, numerical_col):
        q1 = self.data[numerical_col].quantile(0.25)
        q3 = self.data[numerical_col].quantile(0.75)
        IQR = q3 - q1
        outliers = self.data[numerical_col][
            (self.data[numerical_col] < (q1 - 1.5 * IQR)) | (self.data[numerical_col] > (q3 + 1.5 * IQR))]
        inliers = self.data[numerical_col][
            ~((self.data[numerical_col] < (q1 - 1.5 * IQR)) | (self.data[numerical_col] > (q3 + 1.5 * IQR)))]
        return (round(len(outliers) / len(self.data) * 100, 3), round(len(inliers) / len(self.data) * 100, 3))

    def show_box_plot(self, columns):
        if set(columns).issubset(set(self.numerical_cols)):
            i = 0
            while i < len(columns):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if i >= len(columns):
                        break

                    fig = px.box(self.data, y=columns[i], points="all", notched=False)
                    j.plotly_chart(fig, use_container_width=True)
                    with j:
                        perc_inliers, perc_outliers = self.calculateOutliersInliers(columns[i])
                        st.info("Column: '{}' has {}% outliers and {}% inliers".format(columns[i], perc_inliers,
                                                                                       perc_outliers))
                    i += 1
        else:
            st.warning("Categorical columns are not allowed for Box Plots. Remove the categorical columns : %s" % str(
                self.categorical_cols))

    def show_violin_plot(self, columns):
        if set(columns).issubset(set(self.numerical_cols)):
            i = 0
            while i < len(columns):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if i >= len(columns):
                        break

                    fig = px.violin(self.data, y=columns[i], points="all", box=True)
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1
        else:
            st.warning(
                "Categorical columns are not allowed for Violin Plots. Remove the categorical columns : %s" % str(
                    self.categorical_cols))

    def show_pie_plot(self, columns):

        aggregation_option = st.selectbox("Choose the type of Aggregation", options=['Average', 'Max', 'Min', 'Count'])
        if aggregation_option != 'Count':
            numerical_var_agg_option = st.multiselect("Numerical Variables", options=self.numerical_cols)
            group_by_option = st.selectbox("Group By Categorical Column", options=self.categorical_cols)

            def perform_aggregation(numerical_var_agg_option, group_by_option):
                if numerical_var_agg_option == "Average":
                    return self.data.groupby([group_by_option]).mean().reset_index
                elif numerical_var_agg_option == "Max":
                    return self.data.groupby([group_by_option]).max().reset_index
                elif numerical_var_agg_option == "Min":
                    return self.data.groupby([group_by_option]).min().reset_index

            i = 0
            while i < len(columns):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if i >= len(columns):
                        break

                    fig = px.pie(perform_aggregation(), values=columns[i], names='Gender')
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1

    def calculate_null_values_perc(self,columns):
        null_values_df = pd.DataFrame(self.data[columns].isna().sum()).reset_index()
        null_values_df.columns = ['Column', 'Number of Null Values']
        null_values_df['Null Values Percent'] = (null_values_df['Number of Null Values'] / self.data.isna().sum().sum()) * 100
        null_values_df.sort_values(by='Null Values Percent', ascending=True, inplace=True)
        return null_values_df

    def perform_visualization(self, operation, columns):
        if operation == 'Distributions':
            self.show_histogram(columns)
        if operation == "Missing Values":
            fig = plt.figure()
            msno.matrix(self.data, figsize=(25, 5), fontsize=12)
            st.pyplot()

            null_values_df = self.calculate_null_values_perc(columns=columns)
            fig = px.bar(data_frame=null_values_df,x='Null Values Percent',y='Column',orientation='h',color='Null Values Percent')
            st.plotly_chart(fig, use_container_width=True)

        if operation == "Bubble Chart":
            if len(columns) == 2 and set(columns).issubset(set(self.numerical_cols)):
                third_dim = st.selectbox(label="Select the Third Dimension (Numerical)",
                                         options=['None',
                                                  'Aspartate_Aminotransferase',
                                                  'Alamine_Aminotransferase',
                                                  'Alkaline_Phosphotase'])
                fig = px.scatter(self.data, x=columns[0], y=columns[1],
                                 size=None if third_dim == "None" else third_dim,
                                 color="Gender",
                                 hover_name="Dataset", size_max=60, height=1000)
                st.plotly_chart(fig, use_container_width=True)
            elif len(columns) > 2:
                st.warning("Only two numerical columns are allowed for Bubble chart")
            elif not set(columns).issubset(set(self.numerical_cols)):
                st.warning("Selected column should only be Numerical. Categorical columns are not allowed")
        if operation == "Box Plots":
            self.show_box_plot(columns)

        if operation == "Violin Plots":
            self.show_violin_plot(columns)
