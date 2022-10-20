import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
import pickle
import sklearn
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.utils import estimator_html_repr


class TrainMLPipeline:
    def __init__(self, data: pd.core.frame.DataFrame, machine_learning_pipeline: sklearn.pipeline.Pipeline):
        self.data = data
        self.machine_learning_pipeline = machine_learning_pipeline

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def save_decision_tree_diagram(self, decision_tree, feature_names, class_names, out_file_name):
        """
        Save the decision tree as image file (.png)
        :return: None
        """
        export_graphviz(decision_tree,
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True, rounded=True,
                        special_characters=True,
                        out_file='resources/decision_trees/' + out_file_name + ".dot")
        (graph,) = pydot.graph_from_dot_file('resources/decision_trees/' + out_file_name + ".dot")
        graph.write_png('resources/decision_trees/' + out_file_name + '.png')

    def save_pipeline(self, pipeline, pipeline_name):
        """
        Save the pipeline as serialized object
        :return: None
        """
        pickle_out = open('resources/pipeline_diagram/' + pipeline_name + ".pkl", "wb")
        pickle.dump(pipeline, pickle_out)
        pickle_out.close()

    def bayesian_optimization(self, scoring):
        """
        Perform the Bayesian optimization on Decision Tree model pipeline for a given scoring parameter
        :return: Optimized Model
        """
        # Defining the Hyper-Parameter Space for optimization
        params = {
            'estimator__max_depth': Integer(2, 20, prior="log-uniform"),
            'estimator__min_samples_leaf': Integer(5, 500, prior="log-uniform"),
            'estimator__criterion': Categorical(["gini", "entropy"]),
            "estimator__splitter": Categorical(["best", "random"]),
            "estimator__min_weight_fraction_leaf": Real(1e-1, 5e-1, prior="log-uniform"),
            "estimator__max_leaf_nodes": Integer(10, 500, prior="log-uniform")
        }

        # Cross-Validation using Stratified K Fold (3-fold)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
        
        # Start the Bayesian Optimization on Decision Tree Model Pipeline
        opt = BayesSearchCV(
            self.machine_learning_pipeline,
            params,
            n_iter=15,
            random_state=100,
            scoring=scoring,
            cv=cv
        )

        # saving the pipeline in html format
        with open(r'resources/pipeline_diagram/bayesian_optimized_pipeline.html', 'w', encoding='utf-8') as f:
            f.write(estimator_html_repr(opt))

        return opt

    def grid_search_optimization(self, scoring):
        """
        Perform the Grid Search optimization on Decision Tree model pipeline for a given scoring parameter
        :return: Optimized Model
        """
        # Defining the Hyper-Parameter Space for optimization
        params = {
            'estimator__max_depth': [2, 3, 5, 10, 20],
            'estimator__min_samples_leaf': [5, 10, 20, 50, 100, 200, 300, 500],
            'estimator__criterion': ["gini", "entropy"],
            "estimator__splitter": ["best", "random"],
            "estimator__min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
            "estimator__max_leaf_nodes": [10, 50, 100, 200, 300, 500]
        }

        # Cross-Validation using Stratified K Fold (3-fold)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)

        # Start the Grid Search Optimization on Decision Tree Model Pipeline
        grid_search = GridSearchCV(estimator=self.machine_learning_pipeline,
                                   param_grid=params, cv=cv,
                                   scoring=scoring,
                                   verbose=1, n_jobs=-1)

        # saving the pipeline in html format
        with open(r'resources/pipeline_diagram/grid_search_pipeline.html', 'w', encoding='utf-8') as f:
            f.write(estimator_html_repr(grid_search))

        return grid_search

    def random_search_optimization(self, scoring):
        """
        Perform the Random Search optimization on Decision Tree model pipeline for a given scoring parameter
        :return: Optimized Model
        """
        # Defining the Hyper-Parameter Space for optimization
        params = {
            'estimator__max_depth': [2, 3, 5, 10, 20],
            'estimator__min_samples_leaf': [5, 10, 20, 50, 100, 200, 300, 500],
            'estimator__criterion': ["gini", "entropy"],
            "estimator__splitter": ["best", "random"],
            "estimator__min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
            "estimator__max_leaf_nodes": [10, 50, 100, 200, 300, 500]
        }

        # Cross-Validation using Stratified K Fold (3-fold)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)

        # Start the Random Search Optimization on Decision Tree Model Pipeline
        random_search_search = RandomizedSearchCV(estimator=self.machine_learning_pipeline,
                                                  param_distributions=params, cv=cv,
                                                  scoring=scoring,
                                                  verbose=1, n_jobs=-1, random_state=100)

        # saving the pipeline in html format
        with open(r'resources/pipeline_diagram/random_search_optimized_pipeline.html', 'w', encoding='utf-8') as f:
            f.write(estimator_html_repr(random_search_search))

        return random_search_search

    def perform_train(self, optimize: bool = False, optimization_type='Bayesian Optimization', scoring_parameter='f1'):
        """
        Train the Decision Tree Model Pipeline
        :return: Optimizer
        """
        X = self.data[[column for column in self.data.columns if column != 'classification']]
        y = self.data['classification']
        y = y.map({'ckd': 1, 'notckd': 0})

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.8, random_state=100,
                                                                                stratify=y)
        if not optimize:
            self.machine_learning_pipeline.fit(self.X_train, self.y_train)
            #if not exists('resources/pipeline_diagram/un-optimized_pipeline.pkl'):
            self.save_pipeline(self.machine_learning_pipeline, "un-optimized_pipeline")

            # Save the decision tree after training
            self.save_decision_tree_diagram(decision_tree=self.machine_learning_pipeline[2],
                                            feature_names=self.machine_learning_pipeline[1].fit_transform(
                                                self.X_train).columns,
                                            class_names=['notckd', 'ckd'],
                                            out_file_name='un-optimized decision tree model')
        else:
            if scoring_parameter == 'f1 (default)': scoring_parameter = 'f1'
            if optimization_type == 'Bayesian Optimization':
                bayesian_optimizer = self.bayesian_optimization(scoring=scoring_parameter)
                bayesian_optimizer.fit(self.X_train, self.y_train)
                #if not exists('resources/pipeline_diagram/bayesian_optimized_pipeline.pkl'):
                self.save_pipeline(bayesian_optimizer, "bayesian_optimized_pipeline")

                self.save_decision_tree_diagram(decision_tree=bayesian_optimizer.best_estimator_[2],
                                                feature_names=bayesian_optimizer.best_estimator_[1].fit_transform(
                                                    self.X_train).columns,
                                                class_names=['notckd', 'ckd'],
                                                out_file_name='bayesian-optimized decision tree model (optimized for {})'.format(scoring_parameter))
                return bayesian_optimizer
            elif optimization_type == 'Grid Search Optimization':
                grid_search_optimizer = self.grid_search_optimization(scoring=scoring_parameter)
                grid_search_optimizer.fit(self.X_train, self.y_train)
                #if not exists('resources/pipeline_diagram/grid_search_optimized_pipeline.pkl'):
                self.save_pipeline(grid_search_optimizer, "grid_search_optimized_pipeline")

                self.save_decision_tree_diagram(decision_tree=grid_search_optimizer.best_estimator_[2],
                                                feature_names=grid_search_optimizer.best_estimator_[1].fit_transform(
                                                    self.X_train).columns,
                                                class_names=['notckd', 'ckd'],
                                                out_file_name='grid_search-optimized decision tree model (optimized for {})'.format(
                                                    scoring_parameter))
                return grid_search_optimizer
            elif optimization_type == 'Random Search Optimization':
                random_search_optimizer = self.random_search_optimization(scoring=scoring_parameter)
                random_search_optimizer.fit(self.X_train, self.y_train)
                #if not exists('resources/pipeline_diagram/random_search_optimized_pipeline.pkl'):
                self.save_pipeline(random_search_optimizer, "random_search_optimized_pipeline")

                self.save_decision_tree_diagram(decision_tree=random_search_optimizer.best_estimator_[2],
                                                feature_names=random_search_optimizer.best_estimator_[1].fit_transform(
                                                    self.X_train).columns,
                                                class_names=['notckd', 'ckd'],
                                                out_file_name='random_search-optimized decision tree model (optimized for {})'.format(
                                                    scoring_parameter))
                return random_search_optimizer
