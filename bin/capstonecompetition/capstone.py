# -------------------------------------------------------------------------
# Author: Adam de Zoete
# Imperial College Business School
# Professional Certificate in Machine Learning and Artificial Intelligence
# Date: 07 January 2024
# ID: 428
# Function: CapstoneObjective
# Usage: Intended for the Capstone Competition and used within the Jupyter
# notebooks for HPO and assessment of the f(x) for exploration and
# exploitation of the unknown spaces.
# -------------------------------------------------------------------------
# from itertools import product
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, RationalQuadratic, Matern, ExpSineSquared
from tabulate import tabulate

CAP_FUNC_DIMENSIONS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
CAP_FUNC_HEADERS = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
CAP_NUM_FOLDS = 10
CAP_GRID_MAX = 1500000
CAP_CONFIDENCE_INTERVAL = 1.96  # default
CAP_GRID_FOCUS_MARGIN = 0.005 # reduced when approaching exploitation
CAP_GRID_PIN_MARGIN = 0.00005 # reduced when approaching exploitation

class CapstoneObjective(object):
    def __init__(self, x, y, fn):
        """
        Init for this objective class takes on the current observations and function number
        :param x: X values from files
        :param y: Y values from files
        :param fn: Function Number
        """
        super(CapstoneObjective, self).__init__()
        # -------- Features --------
        self._fn = fn
        self._num_features = self.get_feature_num_for_func(self._fn)
        self._fn_headers = self.get_headers_for_func()
        # -------- Data --------
        self._X = np.float64(x)
        self._Y = np.float64(y)
        self._obsX = self.set_best_observe_x()
        self._obsY = self.set_best_observe_y()

    def __call__(self, trial) -> float:
        """
        Objective Optuna function for Bayesian Optimization
        """
        model = self.create_model(trial)

        # Run cross validation
        scores = {"RMSE": [], "F2": []}
        for i in range(CAP_NUM_FOLDS):
            x_train, x_test, y_train, y_test = train_test_split(self._X, self._Y,
                    test_size=0.2, shuffle=True, random_state=42)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores['RMSE'].append(self._rmse(y_pred, y_test))
            scores['F2'].append(metrics.r2_score(y_pred, y_test))

        # Retrain the model on the whole data to get a singlar LML
        model.fit(self._X, self._Y)
        return model.log_marginal_likelihood(model.kernel_.theta), np.min(scores['RMSE']), np.max(scores['F2'])

    @staticmethod
    def _rmse(prediction, true):
        return np.sqrt(np.mean(np.square(prediction - true)))

    def __callOLD__(self, trial) -> float:
        """
        Objective Optuna function for Bayesian Optimization
        """
        model = self.create_model(trial)
        model.fit(self._X, self._Y)
        error = model.log_marginal_likelihood(model.kernel_.theta)
        return error

    def create_model(self, trial) -> GaussianProcessRegressor:
        """
        The HBO for use within the objective function
        ConstantKernel(1e-20, (1e-25, 1e-15))* RBF(length_scale=1)
        """
        k_settings = ["rbf", "rbf-white", "const", "quad", "matern", "sineexp"]
        k_lbound = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        k_rbound = [1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1]
        # if self._fn in [5]:
        #     k_settings = ["rbf", "rbf-white", "const", "quad"]
        kernel_type = trial.suggest_categorical("kernel", k_settings)
        kernel = 1 ** 2 * RBF(length_scale=1)
        match kernel_type:
            case "rbf-white":
                a_rbf_lbound = trial.suggest_categorical("a_rbf_lbound", k_lbound)
                a_rbf_rbound = trial.suggest_categorical("a_rbf_rbound", k_rbound)
                a_wht_lbound = trial.suggest_categorical("a_wht_lbound", k_lbound)
                a_wht_rbound = trial.suggest_categorical("a_wht_rbound", k_rbound)
                a_rbf_lscale = trial.suggest_float("a_rbf_lscale", 0.001, 10.0)
                a_wht_noise = trial.suggest_float("a_wht_noise", 0.1, 10.0)
                kernel = 1 ** 2 * RBF(length_scale=a_rbf_lscale,
                                      length_scale_bounds=(a_rbf_lbound, a_rbf_rbound)) + WhiteKernel(noise_level=a_wht_noise,
                                                                                                  noise_level_bounds=(
                                                                                                      a_wht_lbound,
                                                                                                      a_wht_rbound))
            case "rbf":
                # https://stats.stackexchange.com/questions/445484/does-length-scale-of-the-kernel-in-gaussian-process-directly-relates-to-correlat
                # 0.61  – a decent amount of correlation.
                b_rbf_lbound = trial.suggest_categorical("b_rbf_lbound", k_lbound)
                b_rbf_rbound = trial.suggest_categorical("b_rbf_rbound", k_rbound)
                b_rbf_lscale = trial.suggest_float("b_rbf_lscale", 0.001, 10.0)
                kernel = 1.0 ** 2.0 * RBF(length_scale=b_rbf_lscale, length_scale_bounds=(b_rbf_lbound, b_rbf_rbound))
            case "const":
                c_cnst_lbound = trial.suggest_categorical("c_cnst_lbound", k_lbound)
                c_cnst_rbound = trial.suggest_categorical("c_cnst_rbound", k_rbound)
                c_cnst_lscale = trial.suggest_float("c_cnst_lscale", 0.1, 12.0)
                c_rbf_lscale = trial.suggest_float("c_rbf_lscale", 0.1, 12.0)
                kernel = ConstantKernel(c_cnst_lscale, (c_cnst_lbound, c_cnst_rbound)) * RBF(length_scale=c_rbf_lscale)
            case "quad":
                d_quad_lbound = trial.suggest_categorical("d_quad_lbound", k_lbound)
                d_quad_rbound = trial.suggest_categorical("d_quad_rbound", k_rbound)
                d_quad_lscale = trial.suggest_float("d_quad_lscale", 0.1, 10.0)
                d_quad_wht_noise = trial.suggest_float("d_quad_wht_noise", 0.05, 10.0)
                kernel = (RationalQuadratic(length_scale=d_quad_lscale, alpha_bounds=(
                    d_quad_lbound, d_quad_rbound)) * ConstantKernel() + ConstantKernel() + WhiteKernel(noise_level=d_quad_wht_noise))
            case "matern":
                e_mat_lbound = trial.suggest_categorical("e_mat_lbound", k_lbound)
                e_mat_rbound = trial.suggest_categorical("e_mat_rbound", k_rbound)
                e_mat_lscale = trial.suggest_float("e_mat_lscale", 0.01, 3.0)
                e_mat_nu = trial.suggest_float("e_mat_nu", 0.5, 3.5)
                kernel = 1.0 * Matern(length_scale=e_mat_lscale,
                                      length_scale_bounds=(e_mat_lbound, e_mat_rbound), nu=e_mat_nu)
            case "sineexp":
                f_sine_lbound = trial.suggest_categorical("f_sine_lbound", k_lbound)
                f_sine_rbound = trial.suggest_categorical("f_sine_rbound", k_rbound)
                f_sine_lscale = trial.suggest_float("f_sine_lscale", 0.001, 10.0)
                f_sine_p = trial.suggest_float("f_sine_p", 0.001, 4.0)
                kernel = 1.0 * ExpSineSquared(length_scale=f_sine_lscale,
                                      length_scale_bounds=(f_sine_lbound, f_sine_rbound), periodicity=f_sine_p)

        # params = {"max_iter": 2e05}
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
        return model

    def get_pareto_trail(self, study):
        """
        Gets the Pareto Frontier of the optimum study
        :param study: An Optuna study
        :return: Prints the Pareto Frontier
        """
        trial_with_highest_lml = max(study.best_trials, key=lambda t: t.values[0])
        return trial_with_highest_lml

    def run_model(self, study, x_grid, label="") -> None:
        """
        Returns the results from a study on a grid
        :param study: An Optuna frozen study
        :param x_grid: A grid (see get_grid_for_func)
        :param label: Title for the report output
        :return: Prints the output in a tabulated format
        """
        optuna_frozen_trail = self.get_pareto_trail(study)
        gpr = self.create_model(optuna_frozen_trail)
        post_mean, post_std, best_observed = self._fit_surrogate(gpr, x_grid)

        headers = [' ', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
        # col_header = [[' UCB '], [' +/- '], [' PI '], [' +/- '], [' EI '], [' +/- ']]
        col_header = [[' UCB 95% '], [' UCB 90% '], [' UCB 0% '], [' PI '], [' EI ']]
        results = []

        # Observations
        last_best_obs = self.get_best_observe_x()

        # UCB
        metric_ucb = self._get_upper_confidence_bound(post_mean, post_std)
        ucb_next_point = x_grid[np.argmax(metric_ucb)]
        results.append(ucb_next_point)
        # results.append(self.point_distance(ucb_next_point, last_best_obs))

        # UCB
        metric_ucb = self._get_upper_confidence_bound(post_mean, post_std, 1.91)
        ucb_next_point = x_grid[np.argmax(metric_ucb)]
        results.append(ucb_next_point)

        # UCB
        metric_ucb = self._get_upper_confidence_bound(post_mean, post_std, 1.0)
        ucb_next_point = x_grid[np.argmax(metric_ucb)]
        results.append(ucb_next_point)

        # PI
        metric_pi = self._get_probability_of_improvement(post_mean, post_std, best_observed)
        pi_next_point = x_grid[np.argmax(metric_pi)]
        results.append(pi_next_point)
        # results.append(self.point_distance(pi_next_point, last_best_obs))

        # EI
        metric_ei = self._get_exp_improvement(post_mean, post_std, best_observed)
        ei_next_point = x_grid[np.argmax(metric_ei)]
        results.append(ei_next_point)
        # results.append(self.point_distance(ei_next_point, last_best_obs))

        col_header.append([' BEST '])
        results.append(self._obsX)
        results = np.hstack([col_header, results])

        if label:
            print(f'-f{self._fn}----------------------- {label} {x_grid.shape} -----------------------f{self._fn}-')
        print(tabulate(results, headers[0:self._num_features + 1], floatfmt=".19f"))

        # print(f'------------------ Positioning ------------------')
        results = []
        col_header = [[' UCB '], [' PI '], [' EI ']]
        headers = [' ', 'dist(max(x))', 'dist(x)']

        if label == "Grid Search":
            results.append([self.get_distance(ucb_next_point, last_best_obs).min(axis=1)[0],
                            self.get_distance(ucb_next_point, self._X).min(axis=1)[0]])
            results.append([self.get_distance(pi_next_point, last_best_obs).min(axis=1)[0],
                            self.get_distance(pi_next_point, self._X).min(axis=1)[0]])
            results.append([self.get_distance(ei_next_point, last_best_obs).min(axis=1)[0],
                            self.get_distance(ei_next_point, self._X).min(axis=1)[0]])
            results = np.hstack([col_header, results])
            print(tabulate(results, headers, floatfmt=".19f"))

        trial_values = optuna_frozen_trail.values
        if label == "Grid Search":
            print(f'----------------------- Kernel -----------------------')
            print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
            print(f"Trial with highest log marginal likelihood: ")
            print(f"\tnumber: {optuna_frozen_trail.number}")
            print(f"\tparams: {optuna_frozen_trail.params}")
            print(f"\tvalues: {optuna_frozen_trail.values}")
            print(f"\tgpr: {gpr.kernel_.get_params()}")
            print(f"\tgpr: {gpr}")

        print(f'----------------------- Trial {optuna_frozen_trail._trial_id} Report -----------------------')
        print(f'Previous Best Point: {self._obsY}')
        print(f'\tTrail LML: {trial_values[0]}')
        print(f'\tRMSE: {trial_values[1]}')
        print(f'\tF2 SCORE: {trial_values[2]}')
        print(f'\t\033[1;34mBest Observed Point: {best_observed}')
        print(f'\tGPR Log Marginal Likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta)}\033[0m')

        if label:
            print(f'-f{self._fn}----------------------- {label} -----------------------f{self._fn}-')

    def _fit_surrogate(self, model, x_test) -> tuple[np.array, np.array, float]:
        """
        GP posterior Mean and Std
        :param model: model
        :param x_test: grid for prediction
        :return: model, posterior mean + std dev and best observed of Y
        """
        model.fit(self._X, self._Y)
        post_mean, post_std = model.predict(x_test, return_std=True)
        best_observed = np.max(model.predict(x_test))
        return post_mean, post_std, best_observed

    def get_studyresults(self, study) -> None:
        """
        Brief study results
        :param study: An Optuna study
        :return: Prints various key outputs from a study
        """
        optuna_frozen_trail = self.get_pareto_trail(study)
        # print(f'Best parameter names and values: {optuna_frozen_trail.best_params}')
        # print(f'Best observed log_marginal_likelihood is: {optuna_frozen_trail.values}')
        print(f"Choosen Trial: ")
        for key, value in optuna_frozen_trail.__dict__.items():
            print("\t{}: {}".format(key, value))

    def get_grid_for_func(self, focus=None, limit=CAP_GRID_MAX) -> np.array:
        """
        Returns a grid based various criteria
        :param focus: Whether to focus the grid on an area
        :return: Returns numpy array of points (limited in size)
        """
        num_features = self._num_features
        grid_size = int(np.round(limit ** (1 / num_features)))
        start, stop = zip(*[[0, 1]]*num_features)
        if focus is not None:
            start, stop = self.grid_pin(focus) # self.grid_focus(focus) - replaced to increase exploitation
        # print(f'Grid Size: {start} - {stop} = {grid_size}')
        features = []
        for i in range(num_features):
            features.append(np.linspace(start[i], stop[i], grid_size))
        x_grid = np.meshgrid(*features)
        x_grid = np.vstack([x_grid[i].ravel() for i in range(num_features)]).T
        return np.unique(x_grid, axis=0)

    def set_best_observe_y(self) -> float:
        """
        :return: Returns best Y observation (1 dimension)
        """
        return np.max(self._Y)

    def get_fn(self) -> np.array:
        """
        :return: Returns Function number (int)
        """
        return self._fn

    def get_x(self) -> np.array:
        """
        :return: Returns all X observations (np.array in n dimensions)
        """
        return self._X

    def get_y(self) -> np.array:
        """
        :return: Returns all Y observations (np.array in 1 dimension)
        """
        return self._Y

    def set_best_observe_x(self) -> np.array:
        """
        :return: Returns X of Y's best observation (np.array in n dimensions)
        """
        return self._X[np.argmax(self._Y)]

    def get_best_observe_y(self) -> np.array:
        return self._obsY

    def get_best_observe_x(self) -> np.array:
        return self._obsX

    @staticmethod
    def get_feature_num_for_func(fn) -> int:
        """
        :param fn: Function number
        :return: Number of dimensions in function (int)
        """
        return CAP_FUNC_DIMENSIONS[fn]

    def get_headers_for_func(self) -> list:
        """
        :return: Returns a list of headers for a function
        """
        return CAP_FUNC_HEADERS[:self.get_feature_num_for_func(self._fn)]

    @staticmethod
    def _get_probability_of_improvement(mu, sigma, best_observed, xi=0.01) -> np.array:
        gamma = (mu - (best_observed + xi)) / (sigma + 1E-19)
        pi = norm.cdf(gamma)
        return pi

    @staticmethod
    def _get_exp_improvement(mean, std, best_observed, xi=0.01) -> np.array:
        with np.errstate(divide='warn'):
            improvement = mean - best_observed - xi
            z = improvement / (std + 1E-19)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0  # Handle points with zero standard deviation
        return ei

    @staticmethod
    def _get_upper_confidence_bound(mean, std, kappa=CAP_CONFIDENCE_INTERVAL) -> np.array:
        ucb = mean + kappa * std
        #return np.clip(ucb, 0, 0.9999999999999999999)
        return ucb

    @staticmethod
    def grid_focus(obs, margin=CAP_GRID_FOCUS_MARGIN) -> tuple[np.array, np.array]:
        """
        Focuses a grid on a point
        :param obs: the point to focus on
        :param margin: percentage margin
        :return: Returns a tuple of lower and upper bounds
        """
        lb = np.array(obs) * (1 - margin)
        rb = ((1 - np.array(obs)) * margin) + np.array(obs)
        return np.maximum(lb, 0.), np.minimum(rb, 0.999999)

    @staticmethod
    def grid_pin(obs, margin=CAP_GRID_PIN_MARGIN) -> tuple[np.array, np.array]:
        """
        Focuses a grid on a point
        :param obs: the point to focus on
        :param margin: incremental margin
        :return: Returns a tuple of lower and upper bounds
        """
        lb = np.array(obs) - margin
        rb = np.array(obs) + margin
        return np.maximum(lb, 0.), np.minimum(rb, 0.999999)

    @staticmethod
    def point_distance(obs_a, obs_b) -> np.array:
        obs_c = np.array(obs_a) - np.array(obs_b)
        return obs_c

    @staticmethod
    def get_distance(obs_a, obs_b) -> np.array:
        """
        Gets the distance between arrays of points
        :param obs_a: array of points
        :param obs_b: array of points
        :return: Returns the distance between the points
        """
        obs_c = cdist(np.atleast_2d(obs_a), np.atleast_2d(obs_b))
        # obs_c = np.linalg.norm(np.array(obs_a) - np.array(obs_b), axis=1) #replaced with cdist
        return obs_c
