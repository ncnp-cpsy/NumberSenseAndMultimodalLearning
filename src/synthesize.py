import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


def make_run_ids_dict(experiment_dir):
    print('run_ids_dict was automatically constructed.')
    run_ids_dict = {}
    # model_names = [
    #     os.path.split(model_dir)[1] \
    #     for model_dir in glob.glob(experiment_dir + '/*')
    # ]
    model_names = [
        'VAE_CMNIST',
        'VAE_OSCN',
        'MMVAE_CMNIST_OSCN',
    ]
    for model_name in model_names:
        model_dir = os.path.join(experiment_dir, model_name)
        run_ids = [
            os.path.split(run_id_dir)[1] \
            for run_id_dir in glob.glob(model_dir + '/*')
        ]
        run_ids_dict.update({model_name: run_ids})
    return run_ids_dict

def make_synthesized_data(experiment_dir,
                          run_ids_dict=None,
                          ):
    results = []
    for model_name in run_ids_dict.keys():
        for run_id in run_ids_dict[model_name]:
            for target_modality in [0, 1]:
                fname = os.path.join(
                    experiment_dir,
                    model_name,
                    run_id,
                    'analyse',
                    str(target_modality) + '_1',
                    'analyse_result.csv',
                )
                if os.path.exists(fname):
                    rslt = pd.read_csv(
                        fname,
                        header=0,
                    )
                    results.extend(rslt.to_dict(orient='records'))
                    print(fname, 'was loaded.')
    results = pd.DataFrame(results)
    return results

def plot_box(df_wide,
             target_column,
             fname,
             ):
    df = df_wide
    vals, xs = [], []
    palette = ['r', 'g', 'b', 'y']
    names = df.columns
    # names = ['Multi-modal', 'Single-modal']

    # boxplot
    for i, col in enumerate(df.columns):
        vals.append(df[col].values)
        # adds jitter to the data points - can be adjusted
        xs.append(
            np.random.normal(i + 1, 0.02, df[col].values.shape[0]))
    plt.boxplot(
        vals,
        labels=names,
        widths=(0.6, 0.6),
    )

    # scatter
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.9, color=c)
    plt.tick_params(labelsize=15)
    plt.savefig(fname, format='svg')

    # pandas
    # df.plot.box(grid=True)
    # plt.savefig(fname, format='svg')

    plt.close()
    return

def perform_anova(df,
                  target_modality,
                  target_column,
                  output_dir='./',
                  ):
    print(
        'Start ANOVA.',
        '\ntarget_modality:', target_modality,
        '\ntarget_column:', target_column,
    )
    mmvae_name = 'MMVAE_CMNIST_OSCN'
    if target_modality == 0:
        vae_name = 'VAE_CMNIST'
    elif target_modality == 1:
        vae_name = 'VAE_OSCN'
    else:
        Exception

    # Extract
    query_1 = '(model_name in [@mmvae_name])'
    query_2 = '(target_modality == @target_modality)'
    query_3 = '(model_name in [@vae_name])'
    query = '(' + query_1 + ' and ' + query_2 + ') or ' + query_3
    print(query)
    df = df.query(query)[['id', 'model_name', target_column]]
    df['id'] = df['id'].str.replace(
        'mmvae_cmnist_oscn_', ''
    ).str.replace(
        'vae_oscn_', ''
    ).str.replace(
        'vae_cmnist_', ''
    )
    print(df)

    df_wide = pd.pivot(
        data=df,
        index='id',
        columns='model_name',
        values=target_column,
    )
    print(df_wide)

    # Plot
    fname = output_dir + '/anova_' + str(target_modality) + '_' + target_column + '.svg'
    plot_box(
        df_wide=df_wide,
        target_column=target_column,
        fname=fname
    )

    # ANOVA using stats
    fvalue, pvalue = stats.f_oneway(df_wide[mmvae_name], df_wide[vae_name])
    print(
        'F-value:', fvalue,
        '\np-value:', pvalue,
    )

    # ANOVA using ols and anova_lm
    script = target_column + '~C(model_name)'
    model = ols(script, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    return

def synthesize(args,
               run_ids_dict=None):
    experiment_dir = os.path.join('./rslt/' + args.experiment)

    if type(run_ids_dict) is type(None):
        if 'run_ids_dict' in dir(args):
            run_ids_dict = args.run_ids_dict
        else:
            run_ids_dict = make_run_ids_dict(
                experiment_dir=experiment_dir,
            )
            print('run_ids_dict for synthesized:', run_ids_dict)
    try:
        pprint.pprint(run_ids_dict)
    except Exception as e:
        print(run_ids_dict)

    # convert to pandas dataframe
    results = make_synthesized_data(
        experiment_dir=experiment_dir,
        run_ids_dict=run_ids_dict,
    )

    # Statistical test and plot
    for target_modality in [0, 1]:
        columns_selected = [
            'model_name',
            'id',
            'reconst_' + str(target_modality) + 'x' + str(target_modality) + '_avg',
            'cross_' + str(1 - target_modality) + 'x' + str(target_modality) + '_avg',
            'cluster_avg',
            'magnitude_avg',
        ]
        columns_selected = results.columns
        analyse_synthesized_data(
            synthesized=results,
            columns=columns_selected,
            target_modality=target_modality,
            suffix='_' + str(target_modality),
            output_dir=args.output_dir,
        )
    return

def analyse_synthesized_data(
        synthesized,
        columns=[],
        target_modality=None,
        suffix='',
        output_dir='./',
):
    rslt = {}

    print('---')
    print('analyzed data (all):')
    print(synthesized)
    synthesized.to_csv(output_dir + '/synthesized.csv')

    print('---')
    print('analyzed data (selected):')
    if target_modality is None:
        synthesized = synthesized[columns]
    else:
        query = '(target_modality == @target_modality and model_name in ["MMVAE_CMNIST_OSCN"]) or (not model_name in ["MMVAE_CMNIST_OSCN"])'
        print(query)
        synthesized = synthesized.query(query)[columns]
    print(synthesized)

    print('---')
    cols = [
        'magnitude_avg',
        'cluster_avg',
        'mathematics_avg',
    ]
    for col in cols:
        perform_anova(
            df=synthesized,
            target_modality=target_modality,
            target_column=col,
            output_dir=output_dir,
        )

    print('---')
    print('pairplot')
    fname = output_dir + '/pairplot' + suffix + '.png'
    sns.pairplot(
        synthesized,
        hue='model_name',
    ).savefig(fname, format='png')
    plt.close()

    return rslt

if __name__ == '__main__':
    from src.config import config_synthesizer as args
    synthesize(args=args)
