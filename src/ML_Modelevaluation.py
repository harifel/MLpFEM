import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from numpy.polynomial.polynomial import Polynomial

#error functions
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import PredictionErrorDisplay


######################## Define the text size of each plot globally ###########
SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "Arial"
######################## Define the text size of each plot globally ###########

# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
# })
cm = 1/2.54  # centimeters in inches

class SoilModel:
    path = '../data/'
    file_ending = '.npy'
    seperator = ','

    def __init__(self, modelname: str='', CellPressure: int = None, cellpressure_value: int = None):
        """
        Parameters
        ----------
        modelname : str
            important information - name of constitutive model.
        CellPressure : float, optional
            Takes the column where cell pressure values is saved. The default is None.
        cellpressure_value : float, optional
            Prescribe the desired cell pressure value. The default is None.

        Returns
        -------
        None.
        """
        self.modelname = modelname
        self.path_file =  SoilModel.path + modelname + SoilModel.file_ending #path where the files are saved

        self.CellPressure = CellPressure
        self.cellpressure_value = cellpressure_value

    def import_data(self, file_ending = file_ending, columns = None):
        if file_ending == '.csv':
            df = pd.read_csv(self.path_file, sep = SoilModel.seperator)  #reads the dataframe
            df = df[df[self.CellPressure].isin(self.cellpressure_value)]  #classifies the dataframe for desired cell pressure values

            df_nan = df[df.isna().any(axis=1)]  # DataFrame with NaN values
            df = df.dropna()  # DataFrame without NaN values


        elif file_ending == '.npy':
            df = np.load(self.path_file, allow_pickle=True)
            df = pd.DataFrame(df, columns=columns)
            df = df[df[self.CellPressure].isin(self.cellpressure_value)]  #classifies the dataframe for desired cell pressure values

            df_nan = df[df.isna().any(axis=1)]  # DataFrame with NaN values
            df = df.dropna()  # DataFrame without NaN values

        return df, df_nan

    def PolynomialFeatures(self, x_true: np.ndarray, y_true: np.ndarray, x_check: np.ndarray, degree:int) -> np.ndarray:
        """
        Parameters
        ----------
        x_true : np.ndarray
            Input array in x-direction.
        y_true : np.ndarray
            Input array in y-direction.
        x_check : np.ndarray
            Input array used for prediction.
        degree : float
            Degree of regression.

        Returns
        -------
        y_pred : TYPE
            Predicted array based on input array and used degree for regression. No intercept is used for this method.
        """

        x_true = x_true[:, np.newaxis]
        x_check = x_check[:, np.newaxis]

        model = make_pipeline(PolynomialFeatures(degree = degree, include_bias=False), Ridge(alpha=1, fit_intercept = False))
        model.fit(x_true, y_true)

        y_pred = model.predict(x_check)

        return y_pred

    def interpolation_numpy(self, x: np.ndarray,y: np.ndarray,degree:float) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            Input array in x-direction.
        y : np.ndarray
            Input array in y-direction.
        degree : float
            Degree of regression.

        Returns
        -------
        y_pred : TYPE
            Predicted array based on input array and used degree for regression.
        """

        p = Polynomial.fit(x, y, degree)
        y_pred = p(x)

        return y_pred

    def interpolation_scikit(self, x_true: np.ndarray, y_true: np.ndarray, x_check: np.ndarray, degree: int, include_bias: bool = False)-> np.ndarray:
        """
        Parameters
        ----------
        x_true : np.ndarray
            Input array in x-direction.
        y_true : np.ndarray
            Input array in y-direction.
        x_check : np.ndarray
            Input array used for prediction.
        degree : float
            Degree of regression.
        include_bias : bool, optional
            Intercept for regression. The default is False.

        Returns
        -------
        y_pred : TYPE
            Predicted array based on input array and used degree for regression. No intercept is used for this method.
        """

        polynomial_features = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only= False)

        linear_regression = LinearRegression(fit_intercept = False)

        pipeline = Pipeline(
            [("polynomial_features", polynomial_features),
             ("linear_regression", linear_regression),])

        # Fit the pipeline with reshaped data
        pip = pipeline.fit(x_true[:, np.newaxis], y_true)

        # Predict y-values using the polynomial
        y_pred = pip.predict(x_check[:, np.newaxis])

        return y_pred


    def eval_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> (float, float):
        """
        Parameters
        ----------
        y_true : np.ndarray
            True (or ground truth) array.
        y_pred : np.ndarray
            Predicted array.

        Returns
        -------
        score_r2 : float
            Calculated coefficient of determination based on both input arrays.
        score_mse : float
            Calculated mean squared error based on both input arrays.
        """
        score_r2 = r2_score(y_true, y_pred, force_finite=False)
        score_mse = mean_squared_error(y_true, y_pred)

        return score_r2, score_mse


    def eval_error_col(self, y_true_col: np.ndarray, y_pred_col: np.ndarray, column_name: str = None) -> (float, float):
        """
        Parameters
        ----------
        y_true : np.ndarray
            True (or ground truth) array.
        y_pred : np.ndarray
            Predicted array.
        column_name : str, optional
            Column name. The default is None.

        Returns
        -------
        score_r2 : float
            Calculated coefficient of determination based on both input arrays.
        score_mse : float
            Calculated mean squared error based on both input arrays.
        """
        if len(y_true_col) == len(y_pred_col):
            score_r2 = r2_score(y_true_col, y_pred_col, force_finite=False)
            score_mse = mean_squared_error(y_true_col, y_pred_col)
            print(f'R2 Error for {column_name}: {round(score_r2, 3)}, MSE: {round(score_mse, 3)}.')

        else:
            print("Input columns have unequal length. Please, check your input again!")


    def prediction_error_plot(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Parameters
        ----------
        y_true : np.ndarray
            True (or ground truth) array.
        y_pred : np.ndarray
            Predicted array.

        Returns
        -------
        None
        """

        # Define plot structure
        fig, axs = plt.subplots(ncols=1, figsize=(5*cm, 3*cm), dpi=500)

        # Create an instance of PredictionErrorDisplay
        ped = PredictionErrorDisplay.from_predictions(y_true=y_true,
                                                      y_pred=y_pred,
                                                      kind="actual_vs_predicted",
                                                      subsample=1000,
                                                      ax=axs,
                                                      random_state=0)

        # Set the x and y labels of the PredictionErrorDisplay plot
        ped.ax_.set_xlabel("Best output")  # Set x label
        ped.ax_.set_ylabel("Interpolated data (at $x_{true}$)")  # Set y label

        # Add grid
        ped.ax_.grid()

    def error_plot(self,  y_true: np.ndarray, y_pred: np.ndarray, title= str):
        """
        Parameters
        ----------
        y_true : np.ndarray
            True (or ground truth) array.
        y_pred : np.ndarray
            Predicted array.
        title : str
            Titel name in the figure.

        Returns
        -------
        None
        """
        # Define plot structure
        fig, axs = plt.subplots(ncols=1, figsize=(4*cm, 4*cm), dpi=500)

        # Create an instance of PredictionErrorDisplay
        ped = PredictionErrorDisplay.from_predictions(y_true=y_true,
                                                      y_pred=y_pred,
                                                      kind="actual_vs_predicted",
                                                      subsample=1000,
                                                      ax=axs,
                                                      random_state=0)

        # Set the x and y labels of the PredictionErrorDisplay plot
        ped.ax_.set_xlabel("Predicted Data")  # Set x label
        ped.ax_.set_ylabel("Actual Data")  # Set y label
        ped.ax_.set_title(title)  # Set title

        # Add grid
        ped.ax_.grid()

        # Set title and adjust its position
        title_text = ped.ax_.set_title(title)  # Set title
        title_pos = title_text.get_position()
        title_text.set_position((title_pos[0] - 0.09, title_pos[1]))


    def global_error_plot(self, nrows: int, ncols: int,
                        figsize: tuple,
                        y_true: np.ndarray, y_pred: np.ndarray,
                        skip_data_column: list,
                        input_column_label: np.ndarray, score: np.ndarray,
                        save_path: str):
        """
        Parameters
        ----------
        y_true : np.ndarray
            True (or ground truth) array.
        y_pred : np.ndarray
            Predicted array.

        Returns
        -------
        None
        """

        # Define plot structure
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=500)

        if skip_data_column:
            y_true = np.delete(y_true, skip_data_column, axis=1)
            y_pred = np.delete(y_pred, skip_data_column, axis=1)
            input_column_label = np.delete(input_column_label, skip_data_column)
            score = np.delete(score, skip_data_column)

        for i in range(y_pred.shape[1]):
            y_true_col = y_true[:, i]
            y_pred_col = y_pred[:, i]
            # Create an instance of PredictionErrorDisplay
            ped = PredictionErrorDisplay.from_predictions(y_true=y_true_col,
                                                        y_pred=y_pred_col,
                                                        kind="actual_vs_predicted",
                                                        subsample=1000,
                                                        ax=axs[i // ncols, i % ncols],
                                                        random_state=0)

            # Set the x and y labels of the PredictionErrorDisplay plot
            ped.ax_.set_xlabel("Predicted Data")  # Set x label
            ped.ax_.set_ylabel("Actual Data")  # Set y label

            # Add grid
            ped.ax_.grid()

            # Set title and adjust its position
            title_text = ped.ax_.set_title(f'Parameter {input_column_label[i]}, $R^2$ = {score[i]:.3f}')  # Set title
            title_pos = title_text.get_position()
            title_text.set_position((title_pos[0] - 0.09, title_pos[1]))

            # Customize the subplot
            ax = ped.ax_
            ax.set_xticks(ax.get_xticks())
            ax.tick_params(axis='x', rotation=45)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=1000)


    def plot_data(x_true: np.ndarray, y_true: np.ndarray, label_true: str, color_true: str,
                  x_fit: np.ndarray, y_fit: np.ndarray, label_fit: str, color_fit: str,
                  x_label: str, y_label: str,
                  x_inter:np.ndarray = None, y_inter:np.ndarray = None, color_inter: str = None, label_inter:str = None):
        """
        Parameters
        ----------
        x_true : np.ndarray
            True (or ground truth) array in x-direction.
        y_true : np.ndarray
            True (or ground truth) array in y-direction.
        label_true : str
            Labeling (string) of the true data.
        color_true : str
            Defines the color (string) of the true data.
        x_fit : np.ndarray
            Fitted array in x-direction.
        y_fit : np.ndarray
            Fitted array in y-direction.
        label_fit : str
            Labeling (string) of the fitted data.
        color_fit : str
            Defines the color (string) of the fitted data.
        x_label : str
            Defines the x-label of the figure.
        y_label : str
            Defines the y-label of the figure.
        x_inter : np.ndarray, optional
            Possiblity to plot an additional array in x-direction. The default is None.
        y_inter : np.ndarray, optional
            Possiblity to plot an additional array in y-direction. The default is None.
        color_inter : str, optional
            Defines the color (string) of additional data. The default is None.
        label_inter : str, optional
            Labeling (string) of additional data. The default is None.


        Returns
        -------
        None.

        """

        #define plot structure
        plt.figure(figsize = (5*cm,3*cm), dpi = 500)

        #Plot the actual data points
        plt.plot(x_true, y_true, color_true, label=label_true, linewidth = 1, markersize = 2)

        #Plot the best data
        plt.plot(x_fit, y_fit, color_fit, label=label_fit, linewidth = 1,markersize = 2)

        # Plot the interpolated data if available
        if x_inter is not None and y_inter is not None:
            #plt.scatter(x_inter, y_inter, color_inter, label=label_inter, s=20)
            plt.scatter(x_inter, y_inter, c=color_inter, label=label_inter, s=20)


        #Add labels, title, and legend
        plt.grid()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

class HSdata_process(SoilModel):
    def __init__(self, input_column: list, triaxial_column: list, odoemeter_column: list, **kwargs):
        """
        Parameters
        ----------
        input_column : list
            List of Hardening soil parameters (as strings) which are used for targets in predictions.
        triaxial_column : list
            List of stress-strain curve parameters (as strings) of a triaxial test which are used for labels in predictions.
        odoemeter_column : list
            List of stress-strain curve parameters (as strings) of a oedometer test which are used for labels in predictions.

        Returns
        -------
        None.
        """
        super().__init__(**kwargs) #import from mother class
        self.input_column = input_column #
        self.triaxial_column = triaxial_column
        self.odoemeter_column = odoemeter_column


    def data_processing(self):
        columns = [
                    'E50ref', 'Eoedref','Eurref','phi','cref',
                    'psi','m','nu','Rf','K0NC','CellPressure',
                    'q','eps_y','eps_v','p',
                    'sig_y','eps_y_oed']

        df, df_drop = self.import_data(columns = columns)

        input_parameters = df[self.input_column]
        output_triaxial = df[self.triaxial_column]
        output_odometer = df[self.odoemeter_column]

        input_parameters_nan = df_drop[self.input_column]

        return input_parameters, output_triaxial, output_odometer, input_parameters_nan


    def data_input_error(self, test_df: pd.DataFrame, test_df_y_column: int, test_df_x_column: int,
                         y_true: np.ndarray = None, x_true: np.ndarray = None, degree: int = None) -> np.ndarray:
        """
        Parameters
        ----------
        test_df : pd.DataFrame
            Dataframe of tested data.
        x_true : np.ndarray
            True (or ground truth) array in x-direction.
        y_true : np.ndarray
            True (or ground truth) array in y-direction.
        degree : float
            Degree of regression.

        Returns
        -------
        An array with calculated error for all columns in the dataframe (x_best_output, y_best_output, x_true, y_true, score_list, y_true_interp).
        """

        highest_error = -float('inf')  # Initialize with a low value
        best_results = None
        index = None
        score_list = []

        for i in range(len(test_df)):
                # Set corresponding input parameters
                y_input = test_df.iloc[i, test_df_y_column]
                x_input = test_df.iloc[i, test_df_x_column]
                # Convert to numpy arrays
                x_input = np.array(eval(x_input))
                y_input = np.array(eval(y_input))
                if np.array_equal(x_input, x_true):
                    score_r2, score_mse = self.eval_error(y_true, y_input) #error: first term is true, the second is interpolated/or to be check with
                    score_list.append((score_r2, score_mse,test_df.index[i]))
                    if score_r2 > highest_error:
                        if np.max(abs(y_input)) > np.max(abs(y_true)):
                            pass
                        else:
                            highest_error = score_r2
                            mse = score_mse
                            index = test_df.index[i]
                            y_true_interp = None
                            best_results = (x_input, y_input, x_true, y_true, score_list, y_true_interp)
                else:
                    if i == 0:
                        print(f'!! Interpolation for test data needed, polynomial degree = {degree} !!')
                        ## check the regression accuracy of lab data itsself
                        y_check = self.interpolation_scikit(x_true=x_true, y_true=y_true, x_check = x_true ,degree=degree)
                        score_r2, score_mse = self.eval_error(y_true, y_check) #error: first term is true, the second is interpolated/or to be check with
                        print(f'R2 score test data: {score_r2:.4f}, MSE test data: {score_mse:.4f}')

                        SoilModel.plot_data(
                            x_true = x_true, y_true=y_true, label_true = 'True (input) data', color_true = 'b-o',
                            x_fit = x_true, y_fit =y_check, label_fit = 'Check output', color_fit = 'r-x',
                            x_label = '$\epsilon_1$', y_label = 'q',
                            )
                        self.prediction_error_plot(y_true, y_check) # plot of first regression curve
                        plt.title('Curve fitting test data')
                    ## predict the values at location of articial data
                    y_true_interp = self.interpolation_scikit(x_true=x_true, y_true=y_true, x_check=x_input, degree=degree)
                    score_r2, score_mse = self.eval_error(y_true_interp, y_input) #error: first term is true, the second is interpolated/or to be check with
                    score_list.append((score_r2, score_mse, test_df.index[i]))
                    if score_r2 > highest_error:
                        if np.max(abs(y_input)) > np.max(abs(y_true)):
                            pass
                        else:
                            y_true_interp = self.interpolation_scikit(x_true=x_true, y_true=y_true, x_check=x_input, degree=degree)
                            highest_error = score_r2
                            mse = score_mse
                            index = test_df.index[i]
                            best_results = (x_input, y_input, x_true, y_true, score_list, y_true_interp)
        try:
            print(f'Highest R2 score: {highest_error:.4f}, MSE = {mse:.4f}, Index = {index}')
            if best_results:
                x_best_output, y_best_output, x_true, y_true, score_list, y_true_interp = best_results
        except Exception:
            print('\n\n\n Error message:\n No corresponding data found, please check your input!')
            sys.exit()

        return best_results


    def plot_additional_figures(self,
                                score_df: pd.DataFrame,
                                x_true: np.ndarray, y_true: np.ndarray, label_true: str, color_true: str,
                                label_fit: str, color_fit: str,
                                x_label: str, y_label: str,
                                num_figures: int, test_df_y_column: int, test_df_x_column: int,
                                color_inter: np.ndarray = None, label_inter: np.ndarray = None,
                                test_df: pd.DataFrame = None, degree: int = None):
        """
        Parameters
        ----------
        score_df : pd.DataFrame
            Dataframe of of all metric scores for each investigated parameter.
        x_true : np.ndarray
            True (or ground truth) array in x-direction.
        y_true : np.ndarray
            True (or ground truth) array in y-direction.
        label_true : str
            Labeling (string) of the true data.
        color_true : str
            Defines the color (string) of the true data.
        label_fit : str
            Labeling (string) of the best output data.
        color_fit : str
            Defines the color (string) of the best output data.

        x_label : str
            Defines the x-label of the figure.
        y_label : str
            Defines the y-label of the figure.

        num_figures : int
            Number of figures which are produced in addition.

        test_df : pd.DataFrame, optional
            Investigated dataframe. The default is None.
        test_df_x_column : np.ndarray, optional
            Investigated columns.The default is None.
        test_df_y_column : np.ndarray, optional
            Investigated columns.The default is None.
        label_inter : str, optional
            Labeling (string) of the true data.The default is None.
        color_inter : str, optional
            Defines the color (string) of the true data.The default is None.


        degree : float
            Degree of regression.


        Returns
        -------
        An array with calculated error for all columns in the dataframe (x_best_output, y_best_output, x_true, y_true, score_list, y_true_interp).
        """

        if test_df is None:
            raise ValueError("Output data must be provided.")

        # Get the top N indexes with highest scores
        top_indexes = score_df.nlargest(num_figures, 'R2')['Index'].values[1:] #skip the first value
        top_scores = score_df.nlargest(num_figures, 'R2')['R2'].values[1:] #skip the first value
        top_mse = score_df.nlargest(num_figures, 'R2')['MSE'].values[1:] #skip the first value
        i = 0
        for index in top_indexes:
            # Get the corresponding data from output_triaxial based on index
            y_input = test_df.loc[index, test_df_y_column] #obtain the desired data for the index
            x_input = test_df.loc[index, test_df_x_column] #obtain the desired data for the index
            # Convert to numpy arrays
            y_input = np.array(eval(y_input))
            x_input = np.array(eval(x_input))
            #define plot structure
            plt.figure(figsize = (5,3), dpi = 500)
            #Plot the actual data points
            plt.plot(x_true, y_true, color_true, label=label_true, linewidth = 1, markersize = 2)
            #Plot the best data
            plt.plot(x_input, y_input, color_fit, label=label_fit, linewidth = 1,markersize = 2)
            # Plot the interpolated data if available
            if np.array_equal(x_input, x_true):
                pass
            else:
                y_true_interp = self.interpolation_scikit(x_true=x_true, y_true=y_true, x_check = x_input, degree=degree)
                plt.scatter(x_input, y_true_interp, c = color_inter, label=label_inter, s=20)
            #Add labels, title, and legend
            plt.grid()
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"Score = {top_scores[i]:.4f}, MSE = {top_mse[i]:.4f}, Index = {index:.0f}", size = 10)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{i+10}_{index}_plot.png', dpi = 500)
            print(f'Score = {top_scores[i]:.4f}, MSE = {top_mse[i]:.4f}, Index = {index:.0f}')
            i += 1
