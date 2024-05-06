# -------------------------------------------------------------------------
# Author: ©Adam de Zoete
# Imperial College Business School
# Professional Certificate in Machine Learning and Artificial Intelligence
# Date: 02 February 2024
# ID: 428
# Function: CapstoneVisual
# Usage: Intended for the Capstone Competition and used with the CapstoneObjective
# class to provide visualisation of the data along with the Voronoi diagrams.
# -------------------------------------------------------------------------
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pandas.plotting import scatter_matrix
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, Delaunay, ConvexHull
from scipy.spatial import voronoi_plot_2d
from scipy.optimize import linprog
from shapely.geometry import Point
pd.set_option('display.precision', 10)

CAP_SEABORN_PALETTE = 'Reds'


class CapstoneVisual:
    """
    Extended functionality for CapstoneObjective
    Keeps some functionality outside of the Dask Cluster, which acheives smaller footprint
    and better efficency inside the Optuna objective function.
    """

    def __init__(self, objective):
        """
        init the visuals
        :param objective: CapstoneObjective class
        """
        self.objective = objective

    def return_as_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(self.objective.get_x(), columns=self.objective.get_headers_for_func())
        df['Y'] = self.objective.get_y()
        return df

    def return_scatterpairplot(self) -> None:
        """
        Plots a scatter pair plot of X against the quantiles of Y for improved visabiliy.
        :return:
        """
        data = self.return_as_pandas()
        # Min / Max normalisation
        normalized_y = (data['Y'] - min(data['Y'])) / (max(data['Y']) - min(data['Y']))
        data['Y'] = pd.qcut(normalized_y, q=[0, .25, .5, .75, 1.],
                            labels=['<0.25', '0.25 - 0.5', '0.5 - 0.75', '>0.75'])
        enmax_palette = ['#d9d9fc', '#b5b5ff', '#fc9090', '#c80000']
        sns.set_style("white", {'axes.grid': False})
        g = sns.PairGrid(data, hue="Y", palette=enmax_palette, diag_sharey=False)
        g.map_upper(sns.scatterplot, marker='.')
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.kdeplot)
        # g.set(xticklabels=[0,0.5,1], yticklabels=[0,0.5,1])
        g.set(ylim=(0, 1), xlim=(0, 1))
        g.fig.set_size_inches(14, 14)
        g.fig.suptitle(f"Function {self.objective.get_fn()} - Distribution of X / Quantiles of Y",
                       horizontalalignment='center', y=1.03, x=0.45, fontsize=18)
        g.add_legend(loc='upper center', bbox_to_anchor=(0.45, 1.01), ncol=4, title='')
        plt.show()

    def return_parellelmap(self,data=None) -> None:
        """
        Plots a parallel map of the X dimensions against Y
        :return:
        """
        if data is None:
            data = self.return_as_pandas()

        # Min / Max normalisation quantiles
        normalized_y = (data['Y'] - min(data['Y'])) / (max(data['Y']) - min(data['Y']))
        qnts = np.quantile(normalized_y, [0., 0.25, 0.5, 0.75, 1.]).round(1)
        cscale_palette = [[max(0., qnts[0] - 0.1), '#d9d9fc'], [qnts[1], '#b5b5ff'], [qnts[2], 'gold'],
                          [qnts[3], '#fc9090'], [min(1., qnts[4] + 0.1), '#c80000']]

        # Build Dimensions - range is specific to whether data is supplied
        dims = []
        for i in self.objective.get_headers_for_func():
            dims.append(dict(range=([0, 1] if data is None else [min(data[i]),max(data[i])]), label=i, values=data[i]))
        dims.append(dict(range=[qnts[0], qnts[4]], label='Y', values=normalized_y))

        fig = go.Figure(data=go.Parcoords(line=dict(color=data['Y'], colorscale=cscale_palette), dimensions=dims))
        fig.update_layout(height=800)
        # fig = px.parallel_coordinates(data, color="Y", labels=self.objective.get_headers_for_func(),
        # color_continuous_scale=enmax_palette, color_continuous_midpoint=2, height=800)
        fig.show()

    def plot_distribution(self) -> None:
        """
        Plot the propensity of Y in quantiles in relation to number of observations for each dimension.
        :return: seaborn plot
        """
        labels = ['<0.25', '0.25 - 0.5', '0.5 - 0.75', '>0.75']
        data = self.return_as_pandas()
        # Min / Max normailse Y
        normalized_y = (data['Y'] - min(data['Y'])) / (max(data['Y']) - min(data['Y']))
        data.sort_values(by=['Y'], inplace=True)
        # Slice into quantiles
        data['Y'] = pd.qcut(normalized_y, q=[0, .25, .5, .75, 1.],
                            labels=labels)
        headers = self.objective.get_headers_for_func()
        cols = 2
        plt.figure(figsize=(14, (len(headers) // 2) * 5))
        for idx, column in enumerate(headers):
            plt.subplot(np.sum(divmod(len(headers), cols)), cols, idx + 1)
            ax = sns.histplot(x=column, stat='count', element='step', hue='Y', data=data, bins=40,
                              palette=CAP_SEABORN_PALETTE,
                              multiple='stack', edgecolor='white')
            sns.move_legend(ax, "upper left")
            plt.title(f"{column} Propensity of Y / No of X Observations")
            plt.tight_layout()

    def plot_2d(self, x_=None, y_=None, area=None) -> None:
        # final plot and display
        import matplotlib.patches as mpatches

        if x_ is None:
            x_ = self.objective.get_x()
        if y_ is None:
            y_ = self.objective.get_y()
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.scatter(x_[:, 0], x_[:, 1], c=y_, marker=('x' if area is None else ','), label='Observations', s=(100 if area is None else 2))
        if area is not None:
            ax.add_patch(mpatches.Polygon(area, closed=True, edgecolor='gray', fill=False))
        else:
            for x, y in zip(x_, y_):
                plt.annotate("{:.2f}".format(y), xy=(x[0] + 0.01, x[1] + 0.01))
        obs_x = self.objective.get_best_observe_x()
        ax.scatter(obs_x[0], obs_x[1], c='r', marker='*', label='Best')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        if area is None:
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
        ax.set_title('Capstone Competition - Function %d' % self.objective.get_fn(), fontsize=16)
        ax.legend()
        plt.show()

    def plot_3d(self) -> None:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel('Z')
        ax.scatter(self.objective.get_x()[:, 0], self.objective.get_x()[:, 1],
                   self.objective.get_x()[:, 2], c=self.objective.get_y())
        for x, y in zip(self.objective.get_x(), self.objective.get_y()):
            ax.text(x[0] + 0.01, x[1] + 0.01, x[2] + 0.01, '%.2f' % y, size=9, zorder=1, color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.suptitle('Capstone Competition - Function %d' % self.objective.get_fn(), fontsize=16,
                     verticalalignment='baseline', y=0.84)
        plt.show()

    def in_polygon(polygon, points_) -> list:
        """
        Check if a point is inside a polygon (constraint of 3d)
        :param polygon: vertices from the voronoi diagram
        :param points_: points to check
        :return: boolean array
        """
        return [polygon.contains(Point(x)) for x in points_]

    def in_hull(self, simplex, x):
        """
        Check if a point is inside a convex hull
        :param simplex: hull
        :param x: point
        :return:
        """
        n_points = len(simplex)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[simplex.T, np.ones((n_dim, n_points))]
        b = np.r_[x, np.ones(n_dim)]
        lp = linprog(c, A_eq=A, b_eq=b)
        return lp.success

    def in_delaunay_hull(self, simplex, x):
        return Delaunay(simplex).find_simplex(x) >= 0

    def get_voronoi_polytopes(self, vor_) -> list:
        """
        Get the voronoi polygons from the vertices
        (Generates a lot of data in high dimensions and contrained by notebook memory, requires:
        jupyter server --generate-config and c.ServerApp.iopub_data_rate_limit = 1.0e10
        or jupyter notebook --ServerApp.iopub_data_rate_limit=1.0e10)
        :param vor_: voronoi diagram
        :return: list of points for each polygon
        """
        ply = []
        for n in vor_.regions:
            if -1 not in n and n != []:
                v = [vor_.vertices[i].tolist() for i in n]
                ply.append(v)
        #return [x for x in ply if np.all(np.array(x) >= 0.00) & np.all(np.array(x) < 1.00)]
        return ply

    def get_polytope(self, verts, point_):
        """
        Get the polytope around the point
        :param verts: vertices
        :param point_: point of interest
        :return:
        """
        if len(verts) > 20000:
            return []
        placements = []
        # Skim off the vertices that are outside the bounds of the space
        for n in verts:
            if np.all(np.array(n) >= 0.00) & np.all(np.array(n) < 1.00):
                if self.in_delaunay_hull(np.array(n), point_) == True:
                    placements.append(n)
        if len(placements) != 1:
            fn = self.objective.get_fn()
            warnings.warn(
                "WARNING: The optimal polyhedron for function %s in %s dimensions is not unique and there were %s complete simplexes found." % (fn, self.objective.get_feature_num_for_func(fn), len(placements)))
            return []
        return placements[0]

    def in_poly_hull_multi(self, poly, points):
        hull = ConvexHull(poly)
        res = []
        for p in points:
            new_hull = ConvexHull(np.concatenate((poly, [p])))
            res.append(np.array_equal(new_hull.vertices, hull.vertices))
        return res[0]

    def create_pulsargrid(self, study, poi_, scale=[0., 1.]) -> np.array:
        """
        Create a grid from the best trial by contracting the vertices in towards the POI
        using the convariance matrix to create some non-linearities
        :param study: Best Optuna Study
        :param poi_: Point of Interest
        :return:
        """
        # Get the polygons
        polyhulls = self.get_voronoi_polytopes(self.vor)
        # Get the polygons around the simplex
        simplex = self.get_polytope(polyhulls, poi_)
        # Create pulsations with some non-linearities
        if len(simplex) == 0:
            warnings.warn("WARNING: There is no simplex from Voronoi vertices. Utilising get_grid_for_func(%s)." % (poi_))
            # print(polyhulls)
            vrt = self.objective.get_grid_for_func(poi_)
        else:
            vrt = self.create_pulsation(np.array(simplex), poi_, scale)
        # Slice off out of range vertices
        tol = 0.00
        r, _ = np.where(np.logical_or(vrt < (0.00 - tol), vrt > (1.00 + tol)))
        vrt = np.delete(vrt, r, axis=0)

        optuna_frozen_trail = self.objective.get_pareto_trail(study)
        model = self.objective.create_model(optuna_frozen_trail)
        model.fit(self.objective.get_x(), self.objective.get_y())

        # tidy up before prediction
        vrt = np.round(vrt, 6)
        vrt = np.unique(vrt, axis=0)

        pred_y = model.predict(vrt)

        return self.set_voronoi_table(vrt, pred_y), simplex

    def create_pulsation(self, verts, poi, shell, pulses=10) -> np.array:
        """
        Create a pulsar from the vertices
        :param verts:
        :param poi:
        :param shell:
        :param pulses:
        :return:
        """
        grid_max = 50000
        pulses = int(np.round(grid_max /verts.shape[0]))
        nvert = verts.copy()
        rad = np.linspace(shell[0], shell[1], pulses)
        # Get the feature propensity to create some non-linearities
        f_weights = self.get_feature_propensity_y()
        min_move = 1e-5
        f_weights = f_weights * (1-min_move) + min_move
        for i, n in enumerate(rad):
            if i % 2:
                beam = np.array(nvert - poi) * n + poi
            else:
                beam = (np.array(nvert - poi) * n * f_weights) + poi
            verts = np.vstack([verts, beam])
        return np.unique(verts, axis=0)

    def get_feature_propensity_y(self) -> np.array:
        """
        Get the feature propensity and min/max normalise
        :return: np.array covariance matrix for y
        """
        cov_matrix = self.get_covariance()[:-1, -1]
        cov_matrix = (cov_matrix - min(cov_matrix)) / (max(cov_matrix) - min(cov_matrix))
        return cov_matrix

    def get_vertices(self) -> np.array:
        """
        Returns the vertices of the Voronoi diagram. Removes the vertices that are outside the bounds of the space
        :return: trimmed vertices
        """
        tol = 0.00
        v = self.vor.vertices
        r, _ = np.where(np.logical_or(v < (0.00 - tol), v > (1.00 + tol)))
        v = np.delete(v, r, axis=0)
        return v

    def get_voronoi_table(self, study) -> pd.DataFrame :
        """
        Returns the Voronoi data for the given study, disance from the observed point, and the predicted Y value
        :param self:
        :param study: best optuna study
        :return: pandas dataframe with voronoi data
        """
        # Fit the Voronoi to _X
        vrt, pred_y = self.set_voronoi(study)
        return self.set_voronoi_table(vrt, pred_y)

    def set_voronoi(self, study) -> tuple:
        # Fit the Voronoi to _X
        self.vor = Voronoi(self.objective.get_x())
        # Fit the best trial and predict the vertices
        optuna_frozen_trail = self.objective.get_pareto_trail(study)
        model = self.objective.create_model(optuna_frozen_trail)
        model.fit(self.objective.get_x(), self.objective.get_y())
        vrt = self.get_vertices()
        pred_y = model.predict(vrt)
        return vrt, pred_y

    def set_voronoi_table(self, vrt, pred_y) -> pd.DataFrame:
        """
        Returns the Voronoi data for the given study, disance from the observed point, and the predicted Y value
        :param self:
        :param study: best optuna study
        :return: pandas dataframe with voronoi data
        """
        # Build an output dataFrame
        data = pd.DataFrame(vrt, columns=self.objective.get_headers_for_func())
        # Calculate the distance from the best observed point
        obs_x = self.objective.get_best_observe_x()
        data['dist(max(x))'] = np.linalg.norm(vrt - obs_x, axis=1)
        # Calculate the distance from all other points
        # data['Obs Distance'] = np.linalg.norm(vrt - self._X, axis=1)
        data['dist(x)'] = cdist(vrt, self.objective.get_x()).min(axis=1)
        sort_field = 'dist(x)'
        # Note: above 6 dimensions, Vonoroi becomes very expensive to compute
        if self.objective.get_x().shape[1] < 4 and False:
            centroid = self.voronoi_volumes(self.vor)
            mec_distance = np.linalg.norm(vrt - np.array(centroid), axis=1)
            data['dist(MEC)'] = mec_distance
            sort_field = 'dist(MEC)'
        data['Y'] = pred_y
        return data.sort_values(sort_field)

    @staticmethod
    def voronoi_volumes(v) -> np.array:
        data = pd.DataFrame(columns=['reg_num', 'volume', 'x'])
        for i, reg_num in enumerate(v.point_region):
            indices = v.regions[reg_num]
            if -1 in indices:  # regions with -1 index are outside
                continue
            else:
                hull = ConvexHull(v.vertices[indices])
                centr = np.zeros(v.vertices.shape[1])
                for x in range(v.vertices.shape[1]):
                    centr[x] = np.mean(hull.points[hull.vertices, x])
                if np.any(centr < 0) or np.any(centr > 1):
                    continue
                else:
                    data.loc[i] = {
                        'volume': hull.volume,
                        'reg_num': reg_num,
                        'x': centr
                    }
        return data.sort_values('volume', ascending=False)['x'].values[0]

    def plot_voronoi(self) -> None:
        plt.rcParams['figure.figsize'] = [15, 15]
        voronoi_plot_2d(self.vor)
        plt.title('Voronoi Diagram - Function 2', fontsize=16)
        for x, y in zip(self.objective.get_x(), self.objective.get_y()):
            plt.annotate("{:.2f}".format(y), xy=(x[0] + 0.005, x[1] + 0.001))
        plt.show()

    def get_covariance(self) -> np.array:
        data_y = self.objective.get_y()
        if max(data_y) > 1 or min(data_y) < 0:
            data_y = (data_y - min(data_y)) / (max(data_y) - min(data_y))
        cov_matrix = np.cov(np.concatenate((self.objective.get_x(), data_y.reshape(-1, 1)), axis=1).T)
        return cov_matrix

    def plot_covariance(self)-> None:
        cov_matrix = self.get_covariance()
        axis_labels = self.objective.get_headers_for_func()
        axis_labels.append('Y')
        xticklabels = axis_labels
        yticklabels = axis_labels
        plt.rcParams['figure.figsize'] = [15, 15]
        sns.heatmap(np.round(cov_matrix, 2), annot=True, fmt='g', xticklabels=xticklabels, yticklabels=yticklabels,
                    cmap=CAP_SEABORN_PALETTE)
        plt.show()

    def return_scatterplot(self) -> None:
        data = self.return_as_pandas()
        alpha_normalised = (data['Y'] - min(data['Y'])) / (max(data['Y']) - min(data['Y']))
        scatter_matrix(data.iloc[:, :-1], alpha=alpha_normalised, figsize=(15, 15), diagonal='kde')
        plt.suptitle('Capstone Competition - Function ' + str(self.objective.get_fn()), fontsize=16,
                     verticalalignment='baseline',
                     y=0.90)
        plt.show()
