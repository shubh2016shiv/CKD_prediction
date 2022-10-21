# Streamlit Imports
import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode
from streamlit_yellowbrick import st_yellowbrick
from streamlit_embedcode import github_gist
import streamlit_authenticator as stauth

# Utility Function
from utility import HelperFunctions

# Numerical Calculation Libraries
import pandas as pd
import numpy as np

# Visualization Library
import matplotlib.pyplot as plt

# Library for Exploratory Data Analysis
from exploratory_data_analysis.descriptive_analysis import DescriptiveStatistics
from exploratory_data_analysis.normality_analysis import NomalityAnalysis
from exploratory_data_analysis.charts import Visualization
from exploratory_data_analysis.correlation_analysis import CorrelationAnalysis

# Library for managing the Machine Learning Life Cycle
from ml_steps.import_data import ImportData
from ml_steps.split_data import SplitData
from ml_steps.feature_engineering import FeatureEngineering
from ml_steps.feature_selection import FeatureSelection
from ml_steps.ml_pipeline import MachineLearningPipeline
from ml_steps.train_ml_pipeline import TrainMLPipeline
from ml_steps.evaluate import Evaluate
from ml_steps.predict import Predict

# Miscellaneous Imports
from sklearn.tree import DecisionTreeClassifier
from os.path import exists

st.set_page_config(layout="wide")
helper_functions = HelperFunctions()

# Start the MongoDB connection
helper_functions.initiate_mongodb_connection()

# Retrieve the application user authentication credentials stored on MongoDB 
app_credentials = helper_functions.mongodb.client['chronic_kidney_disease_prediction'].user_auth.find({}).next()

# Perform the application authentication
authenticator = stauth.Authenticate(credentials=app_credentials,cookie_name='ckd_cookie',key='CKD',cookie_expiry_days=30)
name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]: # If the app authentication is successful 
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{st.session_state["name"]}*')
    # Blog UI Code
    st.title("Chronic Kidney Disease Prediction using Machine Learning")
    st.write("-" * 10)
    
    ########################
    # Introduction Section #
    ########################
    st.header("Introduction")
    intro_text = helper_functions.write_text_to_blog(file_name="introduction.txt")
    st.write(intro_text)

    ###################
    # Dataset Section #
    ###################
    st.header("Dataset")
    dataset_info_text = helper_functions.write_text_to_blog(file_name='dataset_info.txt')
    st.write(dataset_info_text)

    dataset_info_df = pd.read_excel('resources/blog_text_content/dataset_information.xlsx')

    st.dataframe(dataset_info_df.style.applymap(helper_functions.color_type, subset=['Type']), use_container_width=True)
    
    st.write("-" * 10)
    
    ##############################################
    # MongoDB Data Security Explaination Section #
    ##############################################
    st.header("MongoDB Database and Data Security")
    st.write("**Below video shows the use of MongoDB for storing the 'Chronic Kidney Disease' dataset as collection inside the database**")
    with st.expander("Expand to see the video"):
        st.video("https://youtu.be/FALXfWdllfw")
    st.write("**Following steps are taken for enhancing the security for accessing this project application** :")
    with st.expander("Expand to see the steps taken for securing MongoDB database and the application security"):
        st.write("1. There are two separate authentications - *MongoDB Database Authentication* and *Application Authentication* ")
        st.write("2. For accessing the MongoDB database, additional user has been created called '*assessment_user*', which is given 'read-only access'")
        img_column_1, img_column_2 = st.columns(2)
        with img_column_1:
            st.image('security_related_snapshots/MongoDB authentication(GUI).png', caption="MongoDB Authentication GUI using Read-Only User")
        with img_column_2:
            st.image('security_related_snapshots/mongoDB read only user.png', caption="Read-Only user on MongoDB database")
        st.write('''3. Connection is made to MongoDB using connection string specific to the 'read-only' user and 
                    password is stored in encrypted format, which is decrypted using a key, everytime the connection is made for authentication''')
        st.image('security_related_snapshots/key_password_connection-string.png',caption="Encrypted password, key (for decryption) and connection string")
        st.write('''
                4. Once database access is authenticated, application access requires another authentication and the credentials of application access are stored on same database but in another collection called *user_auth* 
                ''')
        st.image('security_related_snapshots/application authentication(GUI).png',caption='Application Authentication')
        st.write('''
                5. The password of application user is hashed and authentication is done using the hashed password for the specific user.   
                ''')
        st.image('security_related_snapshots/user hashed password.png',caption='Hashed Application password stored as collection - "user_auth" on MongoDB Database')


    st.write("-" * 10)

    #####################################
    # Exploratory Data Analysis Section #
    #####################################
    st.header("Exploratory Data Analysis")
    dataframe_tab, descriptive_stat_tab, normality_test_tab, charts_tab, corr_tab = \
        st.tabs(["Dataframe", "Descriptive Analysis", 'Normality Test Analysis', "Charts", "Correlation Analysis"])

    with dataframe_tab:
        st.subheader("Dataframe")
        st.dataframe(pd.read_csv('kidney_disease.csv'), use_container_width=True)

    ##################################
    # Start the descriptive analysis #
    ##################################
    with descriptive_stat_tab:
        st.subheader("Descriptive Analysis")
        
        # Read the whole data from MongoDB and get it in dataframe format
        data = helper_functions.get_data_from_mongodb()
        
        # Remove any inconsistencies in dataframe
        data['classification'] = data['classification'].apply(lambda x: x.replace("\t", ""))

        data['pcv'] = data['pcv'].replace(['\t?'], np.nan).replace(['\t43'], '43')
        data['pcv'] = data['pcv'].apply(lambda x: float(x))

        data['wc'] = data['wc'].replace(['\t?'], np.nan).replace(['\t6200'], '6200').replace(['\t8400'], '8400')
        data['wc'] = data['wc'].apply(lambda x: float(x))

        data['rc'] = data['rc'].replace(['\t?'], np.nan)
        data['rc'] = data['rc'].apply(lambda x: float(x))

        data['dm'] = data['dm'].replace(['\tno'], 'no').replace(['\tyes'], 'yes').replace([' yes'], 'yes')

        data['cad'] = data['cad'].replace(['\tno'], 'no')

        # Instantiate the Descriptive Statistics Class that will help in performing the descriptive analysis on data
        descriptive_statistics = DescriptiveStatistics(data=data)

        numerical_variables, _ = descriptive_statistics.get_numerical_and_categorical_vars()
        selected_numerical_vars = st.multiselect(label="Numerical Variables", options=numerical_variables)

        if selected_numerical_vars:
            descriptive_operations = st.multiselect(label="Calculate", options=["Mean", "Median", "Mode", "Sum",
                                                                                "Std. Deviation", "Variance",
                                                                                "Minimum", "Maximum", "Range",
                                                                                "Quartile 1",
                                                                                "Quartile 3", "Skew", "Kurtosis",
                                                                                "95% Confidence Interval of Mean"])
            if descriptive_operations:
                descriptive_statistics.perform_descriptive_analysis(operations=descriptive_operations,
                                                                    columns=selected_numerical_vars)
                st.dataframe(descriptive_statistics.descriptive_stats_df, use_container_width=True)

    ############################
    # Start the normality test #
    ############################
    with normality_test_tab:
        numerical_variables, _ = descriptive_statistics.get_numerical_and_categorical_vars()
        column_selection = st.selectbox(label="Select Columns from the Data for analysing Normality of their distribution",
                                        options=numerical_variables)
        normality_test = NomalityAnalysis(data=data)

        normality_test.perform_statistical_normality_test(column=column_selection)
        normality_test.plot_qq_chart(column=column_selection)
        
    ###############################
    # Perform Data Visualizations #
    ###############################
    with charts_tab:
        st.subheader("Charts")
        viz_selection = st.selectbox(label="Select the type of Charts", options=['Distributions',
                                                                                 'Missing Values',
                                                                                 'Scatter Plot',
                                                                                 'Box Plots',
                                                                                 'Violin Plots'
                                                                                 ])
        if (viz_selection == 'Scatter Plot') or (viz_selection == 'Box Plots') or (viz_selection == 'Violin Plots'):
            column_selection = st.multiselect(label="Select Numerical Columns from the Data", options=numerical_variables)
        else:
            column_selection = st.multiselect(label="Select Columns from the Data", options=data.columns)

        visualization = Visualization(data=data)
        visualization.perform_visualization(operation=viz_selection, columns=column_selection)

    ####################################
    # Perform the correlation Analysis #
    ####################################
    with corr_tab:
        st.subheader("Correlation Analysis")
        corr_analysis = CorrelationAnalysis(data=data, target_var='Dataset')
        corr_analysis.corr_between_numerical_vars() # Correlation analysis amaong numerical variables 
        corr_analysis.corr_between_numerical_and_categorical() # Correlation analysis between numerical and categorical variables


    st.write("-" * 5)
    
    #######################################
    # Machine Learning Life Cycle Section #
    #######################################
    st.header("Machine Learning Life Cycle Steps")
    data_loading_expander = st.expander(label="Step: Data Loading")
    training_test_split_expander = st.expander(label="Step: Split the Data into training and test set")
    feature_engineering_expander = st.expander(label="Step: Feature Engineering")
    feature_selection_expander = st.expander(label="Step: Feature Selection")
    machine_learning_pipeline_expander = st.expander(
        label="Step: Create Machine Learning pipeline for machine learning model")
    pipeline_run_and_model_training_expander = st.expander(
        label="Step: Run custom pipeline using data directly from MongoDB and machine learning model")
    pipeline_optimization_expander = st.expander(label="Step: Optimizing Decision Tree Model using Hyper-parameter tuning",
                                                 expanded=True)
    evaluation_expander = st.expander(label="Step: Evaluate different Models")
    performing_prediction_expander = st.expander(label="Step: Perform Prediction on Unknown Data", expanded=True)
    
    #############################
    # Data Loading from MongoDB #
    #############################
    with data_loading_expander:
        code_column, result_column = st.columns(2)
        with code_column:
            st.subheader("Code")
            github_gist("https://gist.github.com/shubh2016shiv/71b548149cc59ef98bf1d5c87453b422",width=800)
        with result_column:
            st.subheader("Data After Loading from MongoDB")
            import_data = ImportData(mongoDB_connection=helper_functions.mongodb)
            st.dataframe(import_data.get_data_from_mongoDB())

    ##################
    # Data Splitting #
    ##################
    with training_test_split_expander:
        code_column, result_column = st.columns(2)
        with code_column:
            st.subheader("Code")
            github_gist(link='https://gist.github.com/shubh2016shiv/dc82d8c439d77adbc03aa43abc1b3e39', width=800,
                        height=1000)
        with result_column:
            X = data.drop(['classification'], axis=1)
            y = data['classification']
            split_data_step = SplitData(X, y, train_size=0.8, stratify=True, random_state=100)
            X_train, X_test, y_train, y_test = split_data_step.perform_split()
            st.subheader("Before Split")
            st.write("Feature Variables:")
            st.dataframe(X, height=200, use_container_width=True)
            st.write("Target Variable:")
            st.dataframe(y, height=200, use_container_width=True)
            st.subheader("After Split")
            st.write("Training Feature Variables:")
            st.dataframe(X_train, height=200, use_container_width=True)
            st.write("Training Target Variable:")
            st.dataframe(y_train, height=200, use_container_width=True)
            st.write("Test Feature Variables:")
            st.dataframe(X_test, height=200, use_container_width=True)
            st.write("Test Target Variable:")
            st.dataframe(y_test, height=200, use_container_width=True)

    #######################
    # Feature Engineering #
    #######################
    with feature_engineering_expander:
        code_column, result_column = st.columns(2)
        with code_column:
            st.subheader("Code")
            github_gist("https://gist.github.com/shubh2016shiv/a958cb18d545a5e16dfc15b1033a8462", width=800, height=2000)
        with result_column:
            feature_engineering_step = FeatureEngineering(X_train)

            st.subheader("Before Feature Engineering")
            st.write("Training Feature Variables:")
            st.dataframe(X_train, use_container_width=True, height=300)
            st.write("Training Target Variable:")
            st.dataframe(y_train, use_container_width=True, height=300)
            st.write("Test Feature Variables:")
            st.dataframe(X_test, use_container_width=True, height=300)
            st.write("Test Target Variable:")
            st.dataframe(y_test, use_container_width=True, height=300)

            X_train = feature_engineering_step.perform_feature_engineering(X_train)
            helper_functions.save_dataframe_to_csv(X_train, "Train_data_after_feature_engineering.csv")

            numerical_variables, categorical_variables = feature_engineering_step.get_numerical_and_categorical_columns()

            # impute the null values in numerical columns
            X_test = feature_engineering_step.numerical_column_imputer_transformer.transform(X_test)
            X_test = pd.DataFrame(X_test, columns=numerical_variables + categorical_variables)
            # impute the null values in categorical columns
            X_test = feature_engineering_step.categorical_column_imputer_transformer.transform(X_test)
            X_test = pd.DataFrame(X_test, columns=categorical_variables + numerical_variables)
            # normalize the values in numerical columns
            X_test = feature_engineering_step.scaler_column_transformer.transform(X_test)
            X_test = pd.DataFrame(X_test, columns=numerical_variables + categorical_variables)
            # handle the categorical columns by encoding them
            X_test = feature_engineering_step.handle_categorical_variables(X_test)
            helper_functions.save_dataframe_to_csv(X_test, "Test_data_after_feature_engineering.csv")

            y_train = y_train.map({'ckd': 1, 'notckd': 0})
            y_test = y_test.map({'ckd': 1, 'notckd': 0})

            helper_functions.save_dataframe_to_csv(y_train, "Train_Label_data_after_feature_engineering.csv")
            helper_functions.save_dataframe_to_csv(y_test, "Test_Label_data_after_feature_engineering.csv")
            # y_train.to_csv('Labels_After_Feature_Engineering.csv',index=False)

            st.subheader("After Feature Engineering")
            st.write("Training Feature Variables:")
            st.dataframe(X_train, use_container_width=True, height=300)
            st.write("Training Target Variable:")
            st.dataframe(y_train, use_container_width=True, height=300)
            st.write("Test Feature Variables:")
            st.dataframe(X_test, use_container_width=True, height=300)
            st.write("Test Target Variable:")
            st.dataframe(y_test, use_container_width=True, height=300)

    #####################        
    # Feature Selection #
    #####################
    with feature_selection_expander:
        code_column, result_column = st.columns(2)

        with code_column:
            st.subheader("Analysis")
            st.write("-" * 5)
            st.subheader(
                "Statistics based relationship strength of Numerical and Categorical Variables with Target Variable")
            feature_selection_step = FeatureSelection(X=X_train, y=y_train)
            fig = feature_selection_step.anova_test_feature_scores()
            st.plotly_chart(fig, use_container_width=True)

            fig = feature_selection_step.chi_square_test_feature_scores()
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Model Based Feature Importance Analysis")
            fig = feature_selection_step.analyse_feature_importance()
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Performance Comparison Before and After Feature Selection")
            performance_result = feature_selection_step.compare_performance_before_after_feat_sel()

            fig = plt.figure()
            color = {"whiskers": "black", "medians": "black", "caps": "black"}
            performance_result["fit_time"].plot.box(color=color, vert=False)
            plt.xlabel("Elapsed time (s)")
            plt.title("Time to fit the model")
            st.pyplot()

            fig = plt.figure()
            performance_result["score_time"].plot.box(color=color, vert=False)
            plt.xlabel("Elapsed time (s)")
            plt.title("Time to make prediction")
            st.pyplot()

            fig = plt.figure()
            performance_result["test_score"].plot.box(color=color, vert=False)
            plt.xlabel("Accuracy score")
            plt.title("Test score via cross-validation")
            st.pyplot()

            st.subheader("Code")
            github_gist("https://gist.github.com/shubh2016shiv/40670d819c6ebe28ac0bb810ca168e48", width=850)

        with result_column:
            st.subheader("Before Feature Selection")
            st.dataframe(X_train, use_container_width=True, height=1000)

            st.subheader("After Feature Selection")
            X_train = feature_selection_step.perform_feature_selection(X_train)

            helper_functions.save_dataframe_to_csv(X_train, "Train_data_after_feature_selection.csv")
            st.dataframe(X_train, use_container_width=True, height=1000)
            X_test = feature_selection_step.perform_feature_selection(X_test)

            helper_functions.save_dataframe_to_csv(X_test, "Test_data_after_feature_selection.csv")

    ########################################################        
    # Create Machine Learning Pipeline using Decision Tree #
    ########################################################
    with machine_learning_pipeline_expander:
        code_column, result_column = st.columns(2)

        with code_column:
            github_gist('https://gist.github.com/shubh2016shiv/adb8656297c4ef0d7f5af3b0ab850fc7', width=850)
        with result_column:
            st.subheader("Custom Machine Learning Pipeline")

            decision_tree_classifier = DecisionTreeClassifier(random_state=100, class_weight='balanced')
            ml_pipeline = MachineLearningPipeline(machine_learning_model=decision_tree_classifier) # Create Machine Learning pipeline using Decision Tree
            _ = ml_pipeline.create_custom_machine_learning_pipeline()
            st.write("**Below Pipeline Diagram and its individual elements can be clicked to expand for more details.**")
            p = open("resources/pipeline_diagram/pipeline.html")
            components.html(p.read(), width=1000,
                            height=500,
                            scrolling=True)

    ###############################################################        
    # Run Machine Learning pipeline directly from data in MongoDB #
    ###############################################################
    with pipeline_run_and_model_training_expander:
        code_column, result_column = st.columns(2)

        with code_column:
            github_gist('https://gist.github.com/shubh2016shiv/89c8e9b8b03c637e9ab0885bb422bda0', width=850)
        with result_column:
            # Get Data from MongoDB
            importData = ImportData(mongoDB_connection=helper_functions.mongodb)
            raw_data_from_mongoDB = importData.get_data_from_mongoDB()

            # Get the custom machine learning pipeline
            pipeline = ml_pipeline.create_custom_machine_learning_pipeline()

            # Training pipeline has two inputs - 1. Data from MongoDB and 2. pipeline
            train_pipeline_step = TrainMLPipeline(data=raw_data_from_mongoDB,
                                                  machine_learning_pipeline=pipeline)

            # Initiate the training without optimizing it
            train_pipeline_step.perform_train(optimize=False)
            decision_tree_diagram_path = 'resources/decision_trees/un-optimized decision tree model.png'
            st.image(image=decision_tree_diagram_path,
                     caption='Un-optimized Decision Tree after training')
            
            # Save the un-optimized decision tree model as image
            with open(decision_tree_diagram_path, "rb") as file:
                btn = st.download_button(
                    label="Download Decision Tree",
                    data=file,
                    file_name="Decision Tree without optimization.png",
                    mime="image/png")

    #############################################           
    # Perform Model Optimization using Pipeline #
    #############################################
    with pipeline_optimization_expander:
        optimization_column, result_column = st.columns(2)
        optimized_pipeline = None
        with optimization_column:
            st.subheader('Run Optimization')
            st.write('-' * 5)
            scoring_parameter = st.selectbox(label="Select the scoring parameter for which Pipeline has to be optimized",
                                             options=[
                                                 'f1 (default)',
                                                 'accuracy',
                                                 'precision',
                                                 'recall',
                                                 'roc_auc'
                                             ])

            # Select one of three optimization techniques - 1. Bayesian Optimization 2. Grid Search optimization 3. Random Search Optimization
            optimization_type = st.radio(label="Select the type of Optimization", options=['Bayesian Optimization',
                                                                                           'Grid Search Optimization',
                                                                                           'Random Search Optimization'])

            if st.button("Click to optimize"):
                with st.spinner(
                        "Relax ☕! {} is optimizing the pipeline with '{}' as scoring parameter...".format(optimization_type,
                                                                                                          scoring_parameter)):
                    # Perform the model training with optimization
                    optimized_pipeline = train_pipeline_step.perform_train(optimize=True,
                                                                           optimization_type=optimization_type,
                                                                           scoring_parameter=scoring_parameter)
                    st.success("Optimization completed 😮‍💨 🥳🥂!")

        with result_column:
            st.subheader('Results of Pipeline Optimization (Hyper-Parameter Tuning)')
            st.write("-" * 5)
            st.info(
                '''This section shows the result of Machine Learning Pipeline Optimization in real time. The result will be shown everytime when optimization is done for a specific scoring parameter. Click on 'Click to optimize' **(on left side)** to proceed with optimizing the pipeline!
                ''')
            scoring_parameter = scoring_parameter.replace(" (default)", "")
            if optimized_pipeline:
                if optimization_type == 'Bayesian Optimization':
                    st.subheader('Optimization Pipeline')
                    p = open("resources/pipeline_diagram/bayesian_optimized_pipeline.html")
                    components.html(p.read(), width=1000,
                                    height=500,
                                    scrolling=True)
                    st.subheader('Optimized Hyper-Parameters')
                    st.write(optimized_pipeline.best_params_)

                    bayesian_optimized_decision_tree_image_path = "resources/decision_trees/bayesian-optimized decision tree model (optimized for {}).png".format(
                        scoring_parameter)
                    if exists(bayesian_optimized_decision_tree_image_path):
                        st.subheader("Bayesian Optimized Decision Tree")
                        st.image(bayesian_optimized_decision_tree_image_path)
                elif optimization_type == 'Grid Search Optimization':
                    st.subheader('Optimization Pipeline')
                    p = open("resources/pipeline_diagram/grid_search_pipeline.html")
                    components.html(p.read(), width=1000,
                                    height=500,
                                    scrolling=True)
                    st.subheader('Optimized Hyper-Parameters')
                    st.write(optimized_pipeline.best_params_)
                    grid_search_optimized_decision_tree_image_path = "resources/decision_trees/grid_search-optimized decision tree model (optimized for {}).png".format(
                        scoring_parameter)
                    if exists(grid_search_optimized_decision_tree_image_path):
                        st.subheader("Grid Search Optimized Decision Tree")
                        st.image(grid_search_optimized_decision_tree_image_path)
                elif optimization_type == 'Random Search Optimization':
                    st.subheader('Optimization Pipeline')
                    p = open("resources/pipeline_diagram/random_search_optimized_pipeline.html")
                    components.html(p.read(), width=1000,
                                    height=500,
                                    scrolling=True)
                    st.subheader('Optimized Hyper-Parameters')
                    st.write(optimized_pipeline.best_params_)
                    random_search_optimized_decision_tree_image_path = "resources/decision_trees/random_search-optimized decision tree model (optimized for {}).png".format(
                        scoring_parameter)
                    if exists(random_search_optimized_decision_tree_image_path):
                        st.subheader("Random Search Optimized Decision Tree")
                        st.image(random_search_optimized_decision_tree_image_path)

    ################################                    
    # Perform the Model Evaluation #
    ################################
    with evaluation_expander:
        evaluation_model_selection = st.radio("Select the Decision Tree model to be evaluated.",
                                              options=['Un-Optimized Model',
                                                       'Bayesian Optimized Model',
                                                       'Grid-Search Optimized Model',
                                                       'Random-Search Optimized Model'])
        trained_pipeline = None
        if evaluation_model_selection == 'Un-Optimized Model':
            unoptimized_pipeline_path = 'resources/pipeline_diagram/un-optimized_pipeline.pkl'
            if exists(unoptimized_pipeline_path):
                st.info("Evaluating the Un-optimised Decision Tree Model..")
                unoptimized_model_pipeline = helper_functions.load_pickle_file(
                    unoptimized_pipeline_path.replace('.pkl', ''))
                trained_pipeline = unoptimized_model_pipeline
            else:
                st.error("Un-Optimized Decision Tree Model is not found.")

        if evaluation_model_selection == 'Bayesian Optimized Model':
            bayesian_optimized_pipeline_path = 'resources/pipeline_diagram/bayesian_optimized_pipeline.pkl'
            if exists(bayesian_optimized_pipeline_path):
                st.info("Evaluating the Bayesian Optimised Decision Tree Model..")
                try:
                    bayesian_optimized_pipeline = helper_functions.load_pickle_file(
                        bayesian_optimized_pipeline_path.replace('.pkl', ''))
                    trained_pipeline = bayesian_optimized_pipeline
                except (Exception,) as ex:
                    st.error("An Exception has occurred while loading the pipeline. Please run the optimization pipeline again from previous step for scoring metrics of your choice.")
            else:
                st.error(
                    "Bayesian Optimized Decision Tree Model is not found. Run the optimization pipeline from previous step.")

        if evaluation_model_selection == 'Grid-Search Optimized Model':
            grid_search_optimized_pipeline_path = 'resources/pipeline_diagram/grid_search_optimized_pipeline.pkl'
            if exists(grid_search_optimized_pipeline_path):
                st.info("Evaluating the Grid-Search Optimised Decision Tree Model..")
                try:
                    grid_search_optimized_pipeline = helper_functions.load_pickle_file(
                        grid_search_optimized_pipeline_path.replace('.pkl', ''))
                    trained_pipeline = grid_search_optimized_pipeline
                except (Exception,) as ex:
                    st.error("An Exception has occurred while loading the pipeline. Please run the optimization pipeline again from previous step for scoring metrics of your choice.")
            else:
                st.error(
                    "Grid-Search Optimized Decision Tree Model is not found. Run the optimization pipeline from previous step.")

        if evaluation_model_selection == 'Random-Search Optimized Model':
            random_search_optimized_pipeline_path = 'resources/pipeline_diagram/random_search_optimized_pipeline.pkl'
            if exists(random_search_optimized_pipeline_path):
                st.info("Evaluating the Random-Search Optimised Decision Tree Model..")
                try:
                    random_search_optimized_pipeline = helper_functions.load_pickle_file(
                        random_search_optimized_pipeline_path.replace('.pkl', ''))
                    trained_pipeline = random_search_optimized_pipeline
                except (Exception,) as ex:
                    st.error("An Exception has occurred while loading the pipeline. Please run the optimization pipeline again from previous step for scoring metrics of your choice.")
            else:
                st.error(
                    "Random-Search Optimized Decision Tree Model is not found. Run the optimization pipeline from previous step.")

        code_column, result_column = st.columns(2)

        with code_column:
            st.subheader("Code")
            github_gist("https://gist.github.com/shubh2016shiv/915350f06fd3d61b0d867d77cae271ed", width=850, height=1500)
        with result_column:
            if trained_pipeline is not None:
                evaluate_step = Evaluate(trained_pipeline=trained_pipeline,
                                         train_data=train_pipeline_step.X_train,
                                         test_data=train_pipeline_step.X_test,
                                         train_label=train_pipeline_step.y_train,
                                         test_label=train_pipeline_step.y_test)
                
                # Exception handling added because it can throw error in case evaluation is getting done on stale pipeline
                try:
                
                    st.subheader("Confusion Matrix")
                    confusion_matrix_visualizer, model_score = evaluate_step.get_confusion_matrix()
                    st_yellowbrick(confusion_matrix_visualizer)
                
                    st.subheader("Accuracy")
                    st.write(f"#### {model_score*100} %")

                    st.subheader("Classification Report")
                    classification_report_visualizer = evaluate_step.get_classification_report()
                    st_yellowbrick(classification_report_visualizer)

                    st.subheader("Receiver Operating Characteristic/Area Under the Curve")
                    roc_auc_curve_visualizer = evaluate_step.get_ROCAUC_curve()
                    st_yellowbrick(roc_auc_curve_visualizer)

                    st.subheader("Precision-Recall Curve")
                    pr_curve_visualizer = evaluate_step.get_Precision_Recall_curve()
                    st_yellowbrick(pr_curve_visualizer)
                except (Exception,) as ex:
                    st.error("An Exception has occurred while evaluating the pipeline. Please run the same optimization pipeline once again from previous step **(Run Optimization)**, so that new optimized model can be created from pipeline")

    ##########################                
    # Perform the prediction #
    ##########################
    with performing_prediction_expander:
        st.subheader("Prediction for Chronic Kidney Disease")
        st.write("-"*5)
        st.write("**Random Test Samples**:")
        sample_test_values_df = pd.read_csv(r'resources/sample_test_data.csv')
        gb = GridOptionsBuilder.from_dataframe(sample_test_values_df)
        gb.configure_pagination()
        gb.configure_selection(selection_mode="single", use_checkbox=True, suppressRowDeselection=False)
        gridOptions = gb.build()

        sample = AgGrid(sample_test_values_df, gridOptions=gridOptions, theme='material',
                        enable_enterprise_modules=True,
                        allow_unsafe_jscode=True, update_mode=GridUpdateMode.SELECTION_CHANGED, height=300)

        st.write("-" * 5)

        numeric_input_column, categorical_input_column, result_column = st.columns(3)
        numerical_readings = None
        categorical_readings = None

        if len(sample['selected_rows']) == 0:
            st.info(
                "⚠️Select any of the above given sample by marking the check box in front of row. Same values will be automatically filled for respective measurement from the selected row and then click the prediction button. The values can also be manually edited later.")

        else:
            with numeric_input_column:
                age = float(st.number_input("How old is the person? (Years)", min_value=2, max_value=90, value=int(sample['selected_rows'][0]['age'])))
                blood_pressure = float(st.number_input("Blood Pressure (mm/Hg) : ", min_value=50, max_value=180,value=int(sample['selected_rows'][0]['bp'])))
                blood_glucose_random = float(st.number_input("Blood Glucose Random (mgs/dl) : ", min_value=22, max_value=490,value=int(sample['selected_rows'][0]['bgr'])))
                blood_urea = float(st.number_input("Blood Urea (mgs/dl) : ", min_value=1.5, max_value=391.0,value=float(sample['selected_rows'][0]['bu'])))
                serum_creatinine = float(st.number_input("Serum Creatinine (mgs/dl) : ", min_value=0.4, max_value=76.0,value=float(sample['selected_rows'][0]['sc'])))
                sodium = float(st.number_input("Sodium (mgs/dl) : ", min_value=4.5, max_value=163.0,value=float(sample['selected_rows'][0]['sod'])))
                potassium = float(st.number_input("Potassium (mgs/dl) : ", min_value=2.5, max_value=47.0,value=float(sample['selected_rows'][0]['pot'])))
                hemoglobin = float(st.number_input("Hemoglobin (gms) : ", min_value=3.1, max_value=17.8,value=float(sample['selected_rows'][0]['hemo'])))
                packed_cell_volume = float(st.number_input("Packed Cell Volume : ", min_value=9.0, max_value=54.0,value=float(sample['selected_rows'][0]['pcv'])))
                white_blood_cell_count = float(
                    st.number_input("White Blood Cell Count (cells/cumm) : ", min_value=2200.0, max_value=26400.0,value=float(sample['selected_rows'][0]['wc'])))
                red_blood_cell_count = float(
                    st.number_input("Red Blood Cell Count (millions/cmm) : ", min_value=2.1, max_value=8.0,value=float(sample['selected_rows'][0]['rc'])))
                numerical_readings = {
                    'age': [age],
                    'bp': [blood_pressure],
                    'bgr': [blood_glucose_random],
                    'bu': [blood_urea],
                    'sc': [serum_creatinine],
                    'sod': [sodium],
                    'pot': [potassium],
                    'hemo': [hemoglobin],
                    'pcv': [packed_cell_volume],
                    'wc': [white_blood_cell_count],
                    'rc': [red_blood_cell_count]
                }
            with categorical_input_column:
                specific_gravity = float(st.select_slider("Specific Gravity", options=[1.005, 1.01, 1.015, 1.02, 1.025],value=float(sample['selected_rows'][0]['sg'])))
                albumin = float(st.select_slider("Albumin", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],value=float(sample['selected_rows'][0]['al'])))
                sugar = float(st.select_slider("Sugar", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],value=float(sample['selected_rows'][0]['su'])))
                red_blood_cells = st.select_slider("Reb Blood Cells", options=['normal', 'abnormal'],value=sample['selected_rows'][0]['rbc'])
                pus_cells = st.select_slider("Pus Cells", options=['normal', 'abnormal'],value=sample['selected_rows'][0]['pc'])
                pus_cell_clumps = st.select_slider("Pus Cell Clumps", options=['notpresent', 'present'],value=sample['selected_rows'][0]['pcc'])
                bacteria = st.select_slider("Bacteria", options=['notpresent', 'present'],value=sample['selected_rows'][0]['ba'])
                hypertension = st.select_slider("Hypertension", options=['no', 'yes'],value=sample['selected_rows'][0]['htn'])
                diabetes_mellitus = st.select_slider("Diabetes Mellitus", options=['no', 'yes'],value=sample['selected_rows'][0]['dm'])
                coronary_artery_disease = st.select_slider("Coronary Artery Disease", options=['no', 'yes'],value=sample['selected_rows'][0]['cad'])
                appetite = st.select_slider("Appetite", options=['good', 'poor'],value=sample['selected_rows'][0]['appet'])
                pedal_edema = st.select_slider("Pedal Edema", options=['no', 'yes'],value=sample['selected_rows'][0]['pe'])
                anemia = st.select_slider("Anemia", options=['no', 'yes'],value=sample['selected_rows'][0]['ane'])

                categorical_readings = {
                    'sg': [specific_gravity],
                    'al': [albumin],
                    'su': [sugar],
                    'rbc': [red_blood_cells],
                    'pc': [pus_cells],
                    'pcc': [pus_cell_clumps],
                    'ba': [bacteria],
                    'htn': [hypertension],
                    'dm': [diabetes_mellitus],
                    'cad': [coronary_artery_disease],
                    'appet': [appetite],
                    'pe': [pedal_edema],
                    'ane': [anemia]
                }
            readings = {**numerical_readings, **categorical_readings}

            with result_column:
                st.subheader("Result")
                st.write("-"*5)
                new_data = pd.DataFrame.from_dict(readings)
                # st.dataframe(pd.DataFrame.from_dict(new_data))
                trained_pipeline = None
                pipeline_selection = st.radio("Select the Decision Tree model for prediction.",
                                              options=['Un-Optimized Model',
                                                       'Bayesian Optimized Model',
                                                       'Grid-Search Optimized Model',
                                                       'Random-Search Optimized Model'])
                unoptimized_pipeline_path = 'resources/pipeline_diagram/un-optimized_pipeline.pkl'
                bayesian_optimized_pipeline_path = 'resources/pipeline_diagram/bayesian_optimized_pipeline.pkl'
                grid_search_optimized_pipeline_path = 'resources/pipeline_diagram/grid_search_optimized_pipeline.pkl'
                random_search_optimized_pipeline_path = 'resources/pipeline_diagram/random_search_optimized_pipeline.pkl'
                if pipeline_selection == 'Un-Optimized Model' and exists(unoptimized_pipeline_path):
                    trained_pipeline = helper_functions.load_pickle_file(unoptimized_pipeline_path.replace('.pkl', ''))
                elif pipeline_selection == 'Bayesian Optimized Model' and exists(bayesian_optimized_pipeline_path):
                    trained_pipeline = helper_functions.load_pickle_file(bayesian_optimized_pipeline_path.replace('.pkl', ''))
                elif pipeline_selection == 'Grid-Search Optimized Model' and exists(grid_search_optimized_pipeline_path):
                    trained_pipeline = helper_functions.load_pickle_file(
                        grid_search_optimized_pipeline_path.replace('.pkl', ''))
                elif pipeline_selection == 'Random-Search Optimized Model' and exists(random_search_optimized_pipeline_path):
                    trained_pipeline = helper_functions.load_pickle_file(
                        random_search_optimized_pipeline_path.replace('.pkl', ''))
                else:
                    st.error("Model does not exist. Run the specific optimization pipeline from the above step")

                if st.button("Click to Predict Chronic Kidney Disease (CKD)"):
                    predict_step = Predict(pipeline=trained_pipeline, new_data=new_data)
                    prediction = predict_step.show_prediction()
                    if prediction == 'Chronic Kidney Disease NOT Detected':
                        st.success(prediction)
                    elif prediction == 'Chronic Kidney Disease Detected':
                        st.error(prediction)
elif st.session_state["authentication_status"] == False: # In case the application authentication fails
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None: # In case, either password or username is not entered 
    st.warning('Please enter your username and password')
