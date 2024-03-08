import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import pickle
import os
from statsmodels.base.model import GenericLikelihoodModel

from statsmodels.miscmodels.ordinal_model import OrderedModel

# data = pd.read_excel(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\Roschlau Fellowship\THATS '
#                    r'Survey Outputs\1. Collected Data\Sign-Up and Daily Surveys\Sign-Up Survey_September 1_modified.xlsx')
# data = pd.read_excel(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\Sign-Up_ordered_wsltur.xlsx')
# Opening pickled file
# data = pd.read_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\Sign-Up_ordered_wsltur.pkl')

results_folder = r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\Results'
scenario_name = 'Scenario_1 with percentage, popn_density and median year, distance to scotiabank tower and log entropy 1km area' \
                'and land uses classified and POIs within radius included'

# data = pd.read_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\df_persons_wsltur_with_distances09_12b.pkl')
data = pd.read_pickle(r'C:\Users\mwendwa.kiko\Documents\Personal_Kiko\UofT\Research\WSTLUR 2023\df_persons_wsltur_with_entropy4.pkl')
data.dropna(subset=['Household_size', 'Number_cars'], inplace=True)
# Get dummies for column Q8
data = pd.get_dummies(data, prefix='Q8', drop_first=True, columns=['Q8'])
data[['Q8_Male', 'Q8_Other']].astype('int64')
data['Length_of_cycle_lanes_within_radius'].fillna(data['Length_of_cycle_lanes_within_radius'].median(), inplace=True)
data['Distance_to_closest_bus_station'].fillna(data['Distance_to_closest_bus_station'].median(), inplace=True)
data['Popn_density'].fillna(data['Popn_density'].median(), inplace=True)
data['Apartment percentage'].fillna(data['Apartment percentage'].median(), inplace=True)
data['Median_year'].fillna(data['Median_year'].median(), inplace=True)
data['Median_building_age'] = 2023 - data['Median_year']
data['Distance_to_closest_subway_station_log'] = np.log(data['Distance_to_closest_subway_station'])
data['Distance_to_closest_bus_station_log'] = np.log(data['Distance_to_closest_bus_station'])
data['Distance_to_scotiabank_tower_sqrt'] = np.sqrt(data['Distance_to_scotiabank_tower'])
data['Distance_to_scotiabank_tower_squared'] = ((data['Distance_to_scotiabank_tower'])/100) ** 2
data['Distance_to_scotiabank_tower_log'] = np.log(data['Distance_to_scotiabank_tower'])
data['Log_popn_density'] = np.log(data['Popn_density'])
data['Log_entropy'] = np.log(data['entropy'])
data = data.rename({'Q14_12': 'Age'}, axis=1)
data['Walk_mode_share'].replace(0, 1e-5, inplace=True)
exog = data[['Household_size', 'Distance_to_closest_subway_station_log', 'Length_of_cycle_lanes_within_radius',
             'Log_income', 'pc1', 'pc2', 'pc3', 'Log_popn_density', 'Apartment percentage'
             , 'Median_building_age', 'Distance_to_scotiabank_tower_log', 'Log_entropy', 'num_pois_within_radius']]
exog_inflation = data[['Household_size', 'Distance_to_closest_subway_station_log', 'Log_income',
                       'pc1', 'pc2', 'pc3', 'Log_popn_density', 'Apartment percentage',
                       'Median_building_age', 'Distance_to_scotiabank_tower_log', 'Log_entropy', 'num_pois_within_radius']]
exog_inflation = sm.add_constant(exog_inflation, prepend=False)
data['Q43.1'] = np.where(data['Number_cars'] == '0 (no cars in my household)', 0, data['Number_cars'])
data['Q43.1'] = np.where(data['Number_cars'] > 2, 3, data['Number_cars'])
# data['Q43.1'].fillna(1, inplace=True)
# data['Q43.1'] = np.where(data['Q43.1'] == '0 (no cars in my household)', 0, 1)
endog = data['Number_cars']

# exog = sm.add_constant(exog, prepend=False)

class MyOrdProbit(GenericLikelihoodModel):
    def __init__(self, endog, exog1, exog2, extra_params_names=None, **kwds):
        super(MyOrdProbit, self).__init__(endog, exog1, extra_params_names=extra_params_names, kwds=kwds)
        self.exog_inflation = exog2
    # t1, t2 = 0, 0
    # extra_params_names = 't1', 't2'
    def loglike(self, params):
        """The structure of the params list is [original params, threshholds, zero-inflation params]"""
        exog = self.exog
        endog = self.endog
        exog_inflation = self.exog_inflation.to_numpy()   # I created two different exog datasets so that I
        # can have different independent vars for zero inflation and for fleet size.
        no_col = exog.shape[1]
        no_levels = 2
        return_val = np.where(endog == 0, stats.norm.cdf(-np.dot(exog_inflation, params[no_col + no_levels:])).T,
                              np.where(endog == 1, stats.norm.cdf(np.dot(exog_inflation, params[no_col + no_levels:])).T * (stats.norm.cdf(params[no_col] - np.dot(exog, params[:no_col]))),
                                       np.where(endog == 2, stats.norm.cdf(np.dot(exog_inflation, params[no_col + no_levels:])).T * (stats.norm.cdf(params[no_col + 1] - np.dot(exog, params[:no_col])) - stats.norm.cdf(params[no_col] - np.dot(exog, params[:no_col]))),
                                                stats.norm.cdf(np.dot(exog_inflation, params[no_col + no_levels:])).T * (1 - stats.norm.cdf(params[no_col + 1] - np.dot(exog, params[:no_col])))
                                                )))
        return_val = np.where(return_val == 0, 1e-10, return_val)    # This is to avoid log(0) errors.
        # return_val = np.where(endog == 1, stats.norm.logcdf(np.dot(exog, params)),
        #                       stats.norm.logcdf(-np.dot(exog, params)))
        return np.log(return_val).sum()

# def iterate_over_params(exog, exog_inflation):
#     """This function iterates over the parameters and returns the log likelihood for each iteration."""
#     with open('params.pkl', 'rb') as f:
#         start_params_list = pickle.load(f)
#     exog = sm.add_constant(exog, prepend=False)
#     exog_inflation = sm.add_constant(exog_inflation, prepend=False)
#     extra_params = ['t2', 't3'] + [str(col) + '_inflation' for col in exog_inflation.columns]
#     sm_ord_probit_manual = MyOrdProbit(endog.astype(int), exog.astype(int), exog_inflation.astype(int),
#                                        extra_params_names=extra_params, method='bfgs', maxiter=10000, gtol=1e-20)
#     sm_ord_probit_manual_fit = sm_ord_probit_manual.fit(start_params=start_params_list)
#     print(sm_ord_probit_manual_fit.summary())
#     with open('params.pkl', 'wb') as f:
#         pickle.dump(sm_ord_probit_manual_fit.params, f)
#     return sm_ord_probit_manual_fit.llf, sm_ord_probit_manual_fit.params

# with open(os.path.join(results_folder,
#                        f'Scenario_1 with pc1, pc2 and pc3 in both models and distance to scotiabank tower - '
#                        f'New dataset_params.pkl'), 'rb') as f:
#      start_params_list = pickle.load(f)
# # # Add an extra element of value = 0.0174 at the 8th index of the list.
# # start_params_list = np.insert(start_params_list, 8, 0.0174)
# # start_params_list = np.insert(start_params_list, 8, 0.0182)

start_params_list = [0.1] * exog.shape[1] + [0.3, 0.7] + [0.1] * exog_inflation.shape[1]
with open('params.pkl', 'wb') as f:
     pickle.dump(start_params_list, f)
#
# for i in range(3):
#     iterate_over_params(exog, exog_inflation)

for i in range(4):
    # Create list of the column names from the exog_inflation dataset with the suffix '_inflation' added to them.
    extra_params = ['t2', 't3'] + [str(col) + '_inflation_model' for col in exog_inflation.columns]
    sm_ord_probit_manual = MyOrdProbit(endog.astype(int), exog.astype(int), exog_inflation.astype(int), extra_params_names=extra_params, method='bfgs', maxiter=10000, gtol=1e-20)

    # Create a list of 0.1 as long as the number of columns in exog.
    # This is the starting point for the optimization algorithm.
    # start_params_list = [0.1] * exog.shape[1] + [0.3, 0.7] + [0.1] * exog_inflation.shape[1]
    with open('params.pkl', 'rb') as f:
        start_params_list = pickle.load(f)
    sm_ord_probit_manual_fit = sm_ord_probit_manual.fit(start_params=start_params_list, maxiter=50000)
    print(sm_ord_probit_manual_fit.summary())
    #
    # sm_ordered_probit_built_in = OrderedModel(endog, data['Q3_11'], distr='probit').fit(method='nm', maxiter=1000)
    # print(sm_ordered_probit_built_in.summary())
    with open('params.pkl', 'wb') as f:
        pickle.dump(sm_ord_probit_manual_fit.params, f)

    # Append the sm_ord_probit_manual_fit.summary() to a text file.
    with open(os.path.join(results_folder, f'{scenario_name}_results.txt'), 'a+') as f:
        f.write(sm_ord_probit_manual_fit.summary().as_text())
        f.write('\n')


with open(os.path.join(results_folder, f'{scenario_name}_params.pkl'), 'wb') as f:
    pickle.dump(sm_ord_probit_manual_fit.params, f)

# Make a dataframe with model params, std errors, p values, and confidence intervals for each parameter.
# Make the index the parameter names.
params_df = pd.DataFrame({'params': sm_ord_probit_manual_fit.params,
                            'std_err': sm_ord_probit_manual_fit.bse,
                            'p_values': sm_ord_probit_manual_fit.pvalues,
                            'conf_int_lower': sm_ord_probit_manual_fit.conf_int()[:, 0],
                            'conf_int_upper': sm_ord_probit_manual_fit.conf_int()[:, 1]},
                         index=sm_ord_probit_manual.exog_names)
params_df.to_excel(os.path.join(results_folder, f'{scenario_name}_params.xlsx'))


