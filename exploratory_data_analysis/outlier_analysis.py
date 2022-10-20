import plotly.express as px
import plotly.graph_objects as go
from descriptive_analysis import DescriptiveStatistics
import numpy as np
import streamlit as st


class OutliersAnalysis:
    def __init__(self,data):
        self.data = data
        # Remove inconsistencies in data
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

    def plotBoxPlot(self,numerical_col):
        """
        Create Box plot for visualizing outliers
        :return: Plotly Figure
        """
        boxplot_fig = px.box(self.data, y=numerical_col)
        return boxplot_fig

    def calculateOutliersInliers(self,numerical_col):
        """
        Calculate the percentage of outliers and inliers
        :return: pandas.core.frame.DataFrame
        """
        q1 = self.data[numerical_col].quantile(0.25)
        q3 = self.data[numerical_col].quantile(0.75)
        IQR = q3 - q1
        outliers = self.data[numerical_col][(self.data[numerical_col]<(q1-1.5*IQR)) | (self.data[numerical_col]>(q3+1.5*IQR))]
        inliers = self.data[numerical_col][~((self.data[numerical_col]<(q1-1.5*IQR)) | (self.data[numerical_col]>(q3+1.5*IQR)))]
        return (round(len(outliers)/len(self.data)*100,3),round(len(inliers)/len(self.data)*100,3))

    def perform_outlier_analysis(self,columns):
        """
        Perform the outlier analysis for given columns
        :return: None
        """
        if set(columns).issubset(set(self.numerical_cols)):
            i = 0
            while i < len(columns):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if i >= len(columns):
                        break
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=self.data[columns[i]],
                        name="Suspected Outliers",
                        boxpoints='suspectedoutliers',  # only suspected outliers
                        marker=dict(
                            color='rgb(8,81,156)',
                            outliercolor='rgba(219, 64, 82, 0.6)',
                            line=dict(
                                outliercolor='rgba(219, 64, 82, 0.6)',
                                outlierwidth=5)),
                        line_color='rgb(8,81,156)'
                    ))
                    st.plotly_chart(fig)
                    perc_inliers, perc_outliers = self.calculateOutliersInliers(columns[i])
                    st.info(
                        "Column: '{}' has {}% outliers and {}% inliers".format(columns[i], perc_inliers, perc_outliers))
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1
        else:
            st.warning("Categorical columns are not allowed for outlier analysis. Remove the categorical columns : %s" % str(
                self.categorical_cols))
