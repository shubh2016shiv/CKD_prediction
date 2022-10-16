import pandas as pd
from statsmodels.graphics.gofplots import qqplot
import plotly.graph_objs as go
import streamlit as st
from scipy.stats import shapiro, normaltest, kstest
import numpy as np

class NomalityAnalysis:
    def __init__(self, data: pd.core.frame.DataFrame):
        self.data = data.copy(deep=True)

        self.data['pcv'] = self.data['pcv'].replace(['\t?'], np.nan).replace(['\t43'], '43')
        self.data['pcv'] = self.data['pcv'].apply(lambda x: float(x))

        self.data['wc'] = self.data['wc'].replace(['\t?'], np.nan).replace(['\t6200'], '6200').replace(['\t8400'],
                                                                                                       '6200')
        self.data['wc'] = self.data['wc'].apply(lambda x: float(x))

        self.data['rc'] = self.data['rc'].replace(['\t?'], np.nan)
        self.data['rc'] = self.data['rc'].apply(lambda x: float(x))

        self.data['dm'] = self.data['dm'].replace(['\tno'], 'no').replace(['\tyes'], 'yes').replace([' yes'], 'yes')

        self.data['cad'] = self.data['cad'].replace(['\tno'], 'no')

        self.data.fillna(self.data.mean(), inplace=True)

    def plot_qq_chart(self, column):
        qqplot_data = qqplot(self.data[column], line='s').gca().lines
        fig = go.Figure()

        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[0].get_xdata(),
            'y': qqplot_data[0].get_ydata(),
            'mode': 'markers',
            'marker': {
                'color': '#19d3f3'
            }
        })

        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[1].get_xdata(),
            'y': qqplot_data[1].get_ydata(),
            'mode': 'lines',
            'line': {
                'color': '#636efa'
            }

        })

        fig['layout'].update({
            'title': 'Quantile-Quantile Plot',
            'xaxis': {
                'title': 'Theoritical Quantities',
                'zeroline': False
            },
            'yaxis': {
                'title': 'Sample Quantities'
            },
            'showlegend': False,
            'width': 800,
            'height': 700,
        })
        st.subheader("QQ Plot")
        st.plotly_chart(fig, use_container_width=True)

    def perform_shapiro_test(self, column):
        stat_shapiro, p_shapiro = shapiro(self.data[column])
        # interpret
        alpha = 0.05
        if p_shapiro > alpha:
            msg = 'Sample looks Gaussian (fail to reject H0)'
        else:
            msg = 'Sample does not look Gaussian (reject H0)'

        return stat_shapiro, p_shapiro, msg

    def perform_D_Agostino_K_squared_test(self, column):
        stat_normal, p_normal = normaltest(self.data[column])

        # interpret
        alpha = 0.05
        if p_normal > alpha:
            msg = 'Sample looks Gaussian (fail to reject H0)'
        else:
            msg = 'Sample does not look Gaussian (reject H0)'

        return stat_normal, p_normal, msg

    def perform_KS_test(self, column):
        stat_ks, p_ks = kstest(self.data[column], 'norm')

        # interpret
        alpha = 0.05
        if p_ks > alpha:
            msg = 'Sample looks Gaussian (fail to reject H0)'
        else:
            msg = 'Sample does not look Gaussian (reject H0)'

        return stat_ks, p_ks, msg

    def perform_statistical_normality_test(self,column):
        st.subheader("Statistical Normality Test")
        stat_shapiro, p_shapiro, msg_shapiro = self.perform_shapiro_test(column)
        stat_ks, p_ks, msg_ks = self.perform_KS_test(column)
        stat_normal, p_normal, msg_normal = self.perform_D_Agostino_K_squared_test(column)

        statistical_normality_test_df = pd.DataFrame.from_dict({
            'Normality Test Type':['Shapiro',"D'Agostino's K-squared","Kolmogorov–Smirnov"],
            'Length of the sample data':[len(self.data[column]),len(self.data[column]),len(self.data[column])],
            'Significance Level':[0.05,0.05,0.05],
            'Test Statistic':[stat_shapiro,stat_normal,stat_ks],
            'p-value':["{:e}".format(p_shapiro),"{:e}".format(p_normal),"{:e}".format(p_ks)],
            'Comments':[msg_shapiro,msg_normal,msg_ks]
        })

        st.dataframe(statistical_normality_test_df,use_container_width=True)