import numpy as np
import matplotlib.pyplot as plt
import polars as pl

def ignore_nans_list(series:list):

    mask = np.all([~np.isnan(s) for s in series], axis=1)

    return [s[mask] for s in series]

def ignore_nans(x, y):

    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))

    return x[mask], y[mask], mask

def mae(y_true, y_pred, ignore_nan=True):
    if ignore_nan:
        y_true, y_pred,_ = ignore_nans(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred, ignore_nan=True):
    if ignore_nan:
        y_true, y_pred,_ = ignore_nans(y_true, y_pred)
    return np.mean(np.square(y_true - y_pred))

def rmse(y_true, y_pred, ignore_nan=True):
    if ignore_nan:
        y_true, y_pred,_ = ignore_nans(y_true, y_pred)
    return np.sqrt(mse(y_true, y_pred))

def corr(y_true, y_pred, ignore_nan=True):
    if ignore_nan:
        y_true, y_pred,_ = ignore_nans(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]

def pred_fitted_plot(y_true, y_pred, title='', xlabel='', ylabel='', ignore_nan=True, save=False, save_title='pred_vs_fitted', MAE=None, RMSE=None, Cor=None, participant=None):

    if ignore_nan:
        y_true, y_pred, _ = ignore_nans(y_true, y_pred)

    if participant is not None:
        save_title = f'P{participant}_' + save_title

    textstr = '\n'.join([
        f'MAE: {MAE:.2f}',
        f'RMSE: {RMSE:.2f}',
        f'Cor: {Cor:.2f}'])


    plt.figure()
    plt.plot(y_true, y_pred, 'o')
    plt.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'k--')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    plt.text(40, 120, textstr,
             verticalalignment='top', bbox=props)

    plt.ylim(30, 140)
    plt.xlim(30, 140)
    if participant is not None:
        plt.title(f'P{participant}: Ground Truth vs. Predicted HR')
    else:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    if save:
        plt.savefig(f'./figures/{save_title}.png')

    plt.show()

def preds_over_time(t, y_true, y_pred, title='', xlabel='Time (h)', ylabel='HR (BPM)', ignore_nan=True, times = None, save=False, save_title='preds_over_time', MAE=None, RMSE=None, Cor=None, participant=None):

    plt.figure()

    if ignore_nan:
        y_true, y_pred, mask = ignore_nans(y_true, y_pred)

    if times is not None:
        if isinstance(times, pl.Series):
            times = np.array(times.to_numpy())
        t = times / 3600

    if participant is not None:
        save_title = f'P{participant}_' + save_title

    plt.plot(t[mask], y_true, '.', label='Ground Truth HR')
    plt.plot(t[mask], y_pred, '.', label='Predicted HR')
    plt.ylim(30, 140)

    textstr = '\n'.join([
        f'MAE: {MAE:.2f}',
        f'RMSE: {RMSE:.2f}',
        f'Cor: {Cor:.2f}'])

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    plt.text(0.05, 130, textstr,
            verticalalignment='top', bbox=props)

    if participant is not None:
        plt.title(f'P{participant}: Ground Truth vs. Predicted HR over time')
    else:
        plt.title(f'Ground Truth vs. Predicted HR over time')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    if save:
        plt.savefig(f'./figures/{save_title}.png')

    plt.show()

# Load results of any algorithm for a particular participant
def load_std_prediction_df(subject_id, run=0, dataset='wristbcg', method='std'):

    if run == 0:
        results_folder = 'results'
    else:
        results_folder = f'results_{run}'

    if dataset == 'wristbcg':
        file_pre = f'p{subject_id}'

    elif dataset == 'aw':
        file_pre = f'aw_{subject_id}'

    elif dataset == 'ntnu':
        file_pre = f'ntnu_{subject_id}'

    return pl.read_csv(f'./{results_folder}/{file_pre}_{method}_df.csv')

def load_bioinisghts_prediction_df(subject_id, run=0, dataset='wristbcg', method='bioinsights'):

    if run == 0:
        results_folder = 'results'
    else:
        results_folder = f'results_{run}'

    if dataset == 'wristbcg':
        file_pre = f'p{subject_id}'

    elif dataset == 'aw':
        file_pre = f'aw_{subject_id}'

    elif dataset == 'ntnu':
        file_pre = f'ntnu_{subject_id}'

    return pl.read_csv(f'./{results_folder}/{file_pre}_{method}_df.csv')

def evaluate(y_true, y_pred, ignore_nan=True, plot=False, print_errors = False, times = None, save = False, participant=None):

    if isinstance(y_true, pl.Series):
        y_true = np.array(y_true.to_numpy())

    if isinstance(y_pred, pl.Series):
        y_pred = np.array(y_pred.to_numpy())

    return_dict = {'MAE': mae(y_true, y_pred, ignore_nan),
            'MSE': mse(y_true, y_pred, ignore_nan),
            'RMSE': rmse(y_true, y_pred, ignore_nan),
            'Corr': corr(y_true, y_pred, ignore_nan)}

    if plot:
        pred_fitted_plot(y_true, y_pred, ignore_nan=ignore_nan, title='Ground Truth vs. Predicted HR', xlabel='Ground Truth HR', ylabel='Predicted HR', save=save, participant=participant, MAE=return_dict['MAE'], RMSE=return_dict['RMSE'], Cor=return_dict['Corr'])
        preds_over_time(np.arange(len(y_true)), y_true, y_pred, ignore_nan=ignore_nan, times=times, save=save, MAE=return_dict['MAE'], RMSE=return_dict['RMSE'], Cor=return_dict['Corr'],participant=participant)

    if print_errors:
        print(f'MAE: {mae(y_true, y_pred, ignore_nan)}')
        print(f'MSE: {mse(y_true, y_pred, ignore_nan)}')
        print(f'RMSE: {rmse(y_true, y_pred, ignore_nan)}')
        print(f'Corr: {corr(y_true, y_pred, ignore_nan)}')

    return return_dict

# GO FROM A DICTIONARY OF EVALUATIONS TO AVERAGES ACROSS ALL PARTICIPANTS
def get_average_eval(eval_dict):
    mae_list = []
    corr_list = []
    rmse_list = []

    for key in eval_dict:
        mae_list.append(eval_dict[key]['MAE'])
        rmse_list.append(eval_dict[key]['RMSE'])
        corr_list.append(eval_dict[key]['Corr'])

    return np.mean(mae_list), np.mean(rmse_list), np.mean(corr_list)