import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt
from exploratory_data_analysis.descriptive_analysis import DescriptiveStatistics

st.set_option('deprecation.showPyplotGlobalUse', False)


def get_proper_column_name(col_name):
    """
        Returns the proper name of the abbreviated column name in data
    :return: String
    """
    if col_name == 'bp':
        return 'Blood Pressure'
    elif col_name == 'sg':
        return 'Specific Gravity'
    elif col_name == 'al':
        return 'Albumin'
    elif col_name == 'su':
        return 'Sugar'
    elif col_name == 'rbc':
        return 'Red Blood Cells'
    elif col_name == 'pc':
        return 'Pus Cell'
    elif col_name == 'pcc':
        return 'Pus Cell Clumps'
    elif col_name == 'ba':
        return 'Bacteria'
    elif col_name == 'bgr':
        return 'Blood Glucose Random'
    elif col_name == 'bu':
        return 'Blood Urea'
    elif col_name == 'sc':
        return 'Serum Creatinine'
    elif col_name == 'sod':
        return 'Sodium'
    elif col_name == 'pot':
        return 'Potassium'
    elif col_name == 'hemo':
        return 'Hemoglobin'
    elif col_name == 'pcv':
        return 'Packed Cell Volume'
    elif col_name == 'wc':
        return 'White Blood Cell Count'
    elif col_name == 'rc':
        return 'Red Blood Cell Count'
    elif col_name == 'htn':
        return 'Hypertension'
    elif col_name == 'dm':
        return 'Diabetes Mellitus'
    elif col_name == 'cad':
        return 'Coronary Artery Disease'
    elif col_name == 'appet':
        return 'Appetite'
    elif col_name == 'pe':
        return 'Pedal Edema'
    elif col_name == 'ane':
        return 'Anemia'
    else:
        return 'classification (CKD or No-CKD)'

class Visualization:
    def __init__(self, data: pd.core.frame.DataFrame):
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

        # get the numerical and categorical column from Descriptive Analysis Class
        descriptive_stat_class = DescriptiveStatistics(self.data)
        self.numerical_cols, self.categorical_cols = descriptive_stat_class.get_numerical_and_categorical_vars()

    def show_histogram(self, columns):
        """
        Create the Histograms for specific list of columns
        :return: None
        """
        def plot_histogram_as_subplots(__columns, color=None):
            i = 0
            while i < len(__columns):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if i >= len(__columns):
                        break

                    if color is None:
                        fig = px.histogram(self.data, x=columns[i],
                                           title="Histogram for '{}'".format(get_proper_column_name(columns[i])))
                        j.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.histogram(self.data, x=columns[i], color=color,
                                           title="Histogram for '{}'".format(get_proper_column_name(columns[i])))
                        j.plotly_chart(fig, use_container_width=True)

                    i += 1

        if 'classification' in columns:
            plot_histogram_as_subplots(columns, color='classification')
        else:
            plot_histogram_as_subplots(columns)

    def calculateOutliersInliers(self, numerical_col):
        """
        Calculate the Percentage of Outlier and Inliers for given numerical columns
        :return: pandas.core.frame.DataFrame
        """
        q1 = self.data[numerical_col].quantile(0.25)
        q3 = self.data[numerical_col].quantile(0.75)
        IQR = q3 - q1
        outliers = self.data[numerical_col][
            (self.data[numerical_col] < (q1 - 1.5 * IQR)) | (self.data[numerical_col] > (q3 + 1.5 * IQR))]
        inliers = self.data[numerical_col][
            ~((self.data[numerical_col] < (q1 - 1.5 * IQR)) | (self.data[numerical_col] > (q3 + 1.5 * IQR)))]
        return (round(len(outliers) / len(self.data) * 100, 3), round(len(inliers) / len(self.data) * 100, 3))

    def show_box_plot(self, columns):
        """
        Create Box Plot for given list of Columns
        :return: None
        """
        show_categories = st.checkbox(
            "Tick the checkbox to show Box plots separately for each category of CKD and non-CKD.")
        if show_categories:
            classification_color = "classification"
        else:
            classification_color = None

        if set(columns).issubset(set(self.numerical_cols)):
            i = 0
            while i < len(columns):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if i >= len(columns):
                        break

                    fig = px.box(self.data, y=columns[i], points="all", notched=False, color=classification_color,
                                 color_discrete_map={"ckd": "red",
                                                     "notckd": "green"},
                                 title="Box Plot for '{}'".format(get_proper_column_name(columns[i])))
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
        """
        Create Violin Plot for given list of columns 
        :return: None
        """
        if set(columns).issubset(set(self.numerical_cols)):
            i = 0
            while i < len(columns):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if i >= len(columns):
                        break

                    fig = px.violin(self.data, y=columns[i], points="all", box=True,
                                    title="Violin Plot for '{}'".format(get_proper_column_name(columns[i])))
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1
        else:
            st.warning(
                "Categorical columns are not allowed for Violin Plots. Remove the categorical columns : %s" % str(
                    self.categorical_cols))

    def calculate_null_values_perc(self, columns):
        """
        Calculate the Percentage Null Values for given list of columns    
        :return: pandas.core.frame.DataFrame
        """
        null_values_df = pd.DataFrame(self.data[columns].isna().sum()).reset_index()
        null_values_df.columns = ['Column', 'Number of Null Values']
        null_values_df['Null Values Percent'] = (null_values_df[
                                                     'Number of Null Values'] / self.data.isna().sum().sum()) * 100
        null_values_df.sort_values(by='Null Values Percent', ascending=True, inplace=True)
        return null_values_df

    def perform_visualization(self, operation, columns):
        """
        Perform the visualization for a given operation (like ploting distribution, missing values, scatter plot etc) and list of columns
        :return: None
        """
        if operation == 'Distributions':
            self.show_histogram(columns)
        if operation == "Missing Values":
            st.info("The white horizontal patches indicates that value is missing at specific row number.")
            fig = plt.figure()
            msno.matrix(self.data, figsize=(25, 5), fontsize=12)
            st.pyplot()

            null_values_df = self.calculate_null_values_perc(columns=columns)
            fig = px.bar(data_frame=null_values_df, x='Null Values Percent', y='Column', orientation='h',
                         color='Null Values Percent',
                         hover_name='Number of Null Values',
                         title="Null Values per Column")
            st.plotly_chart(fig, use_container_width=True)

        if operation == "Scatter Plot":
            st.write('''**The scatter diagram graphs pairs of numerical data, with one variable on each axis, to look for a relationship between them.
             If the variables are correlated, the points will fall along a line or curve. 
             The better the correlation, the tighter the points will hug the line.**''')
            show_trend_line = st.checkbox("Tick the checkbox to show Trend Line(s).")
            if show_trend_line:
                trendline = "lowess"
            else:
                trendline = None

            if len(columns) == 2 and set(columns).issubset(set(self.numerical_cols)):

                fig = px.scatter(self.data, x=columns[0], y=columns[1],
                                 color="classification",
                                 size_max=60, height=1000,
                                 trendline=trendline,
                                 color_discrete_map={"ckd": "red",
                                                     "notckd": "green"},
                                 marginal_x="histogram", marginal_y="rug",
                                 symbol="classification",
                                 title="Scatter Plot between '{}' and '{}'".format(get_proper_column_name(columns[0]),
                                                                              get_proper_column_name(columns[1])))
                st.plotly_chart(fig, use_container_width=True)
            elif len(columns) > 2:
                st.warning("Only two numerical columns are allowed for Scatter plot")
            elif not set(columns).issubset(set(self.numerical_cols)):
                st.warning("Selected column should only be Numerical. Categorical columns are not allowed")
        if operation == "Box Plots":
            st.write('''
            **A box plot is a statistical representation of the distribution of a variable through its quartiles. 
            The ends of the box represent the lower and upper quartiles, while the median (second quartile) is marked by a line inside the box.
            Box plot is also used for visualizing the potential outliers.
            When reviewing a box plot, an outlier is defined as a data point that is located outside the whiskers of the box plot.**
            ''')
            self.show_box_plot(columns)

        if operation == "Violin Plots":
            st.write('''**A violin plot is a statistical representation of numerical data. 
            It is similar to a box plot, with the addition of a rotated kernel density (KDE) plot on each side.**''')
            self.show_violin_plot(columns)
