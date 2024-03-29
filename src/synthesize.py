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

def plot_scatter(df,
                 fname,
                 model_name=None,
                 target_modality=None,
                 col_x='',
                 col_y='',
                 ):
    if model_name is not None:
        df = df.loc[df['model_name'] == model_name]
    if target_modality is not None:
        df = df.loc[df['target_modality'] == target_modality]
    plt.scatter(df[col_x], df[col_y])
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.savefig(fname, format='svg')
    plt.close()
    return

def plot_box(df_wide,
             fname,
             target_columns=None,
             names=None,
             widths=None,
             ):
    df = df_wide
    vals, xs = [], []
    palette = ['r', 'g', 'b', 'y', 'orange']
    if target_columns is None:
        target_columns = df.columns
    if names is None:
        names = df.columns
    if widths is None:
        widths = [1.2 / len(df.columns) for num in range(len(df.columns))]

    # boxplot
    for i, col in enumerate(target_columns):
        vals.append(df[col].values)
        # x = np.random.normal(i + 1, 0.02, df[col].values.shape[0])  # with jitter
        x = [i + 1 for n in range(df[col].values.shape[0])]  # without jitter
        xs.append(x)
    plt.boxplot(
        vals,
        labels=names,
        widths=widths,
    )

    # scatter
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.9, color=c)
    plt.tick_params(labelsize=15)
    plt.xticks(rotation=0)
    plt.savefig(fname, format='svg')

    # pandas
    # df.plot.box(grid=True)
    # plt.savefig(fname, format='svg')

    plt.close()
    print(fname, 'was saved.')

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
    print('---')
    print(df)

    df_wide = pd.pivot(
        data=df,
        index='id',
        columns='model_name',
        values=target_column,
    )
    print('---')
    print(df_wide)
    print('---')
    print(df_wide.describe())

    # Plot
    fname = output_dir + '/anova_' + str(target_modality) + '_' + target_column + '.svg'
    plot_box(
        df_wide=df_wide,
        fname=fname,
        names=['Multi-modal', 'Single-modal'],
    )

    # ANOVA using stats
    try:
        fvalue, pvalue = stats.f_oneway(df_wide[mmvae_name], df_wide[vae_name])
        print(
            '\n---',
            '\nF-value:', fvalue,
            '\np-value:', pvalue,
        )
    except Exception as e:
        print('ANOVA was skipped because', e)

    # ANOVA using ols and anova_lm
    try:
        script = target_column + '~C(model_name)'
        model = ols(script, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
    except Exception as e:
        print('ANOVA was skipped because', e)

    # Welch's t-test
    try:
        print(stats.ttest_ind(
            df_wide[mmvae_name],
            df_wide[vae_name],
            equal_var=False,
            # alternative='greater', not supported due to old sicpy version
        ))
    except Exception as e:
        print('t-test was skipped because', e)
    return

def perform_anova_logtrans(df,
                           target_modality,
                           output_dir
                           ):
    model_names = [
        'MMVAE_CMNIST_OSCN',
        'VAE_CMNIST',
        'VAE_OSCN',
    ]
    cols = [
        'magnitude_avg',
        'magnitude_logmin_mean_avg',
        # 'magnitude_log_mean_avg',
        'magnitude_exp_mean_avg',
        'magnitude_pow_mean_avg',
        # 'magnitude_logmin_dist_avg',
        # # 'magnitude_log_dist_avg',
        # 'magnitude_exp_dist_avg',
        # 'magnitude_pow_dist_avg',
    ]

    def perform_anova_logtrans_model(df_wide,
                                     model_name,
                                     target_modality,
                                     output_dir):
        # Plot
        fname = output_dir + '/anova_logtrans_' + model_name + str(target_modality) + '.svg'
        plot_box(
            df_wide=df_wide,
            fname=fname,
            names=['original', 'log', 'exp', 'pow']
        )

        # ANOVA using stats
        try:
            fvalue, pvalue = stats.f_oneway(*[df_wide[col] for col in cols])
            print(
                '\nF-value:', fvalue,
                '\np-value:', pvalue,
            )
        except Exception as e:
            print('ANOVA was skipped because', e)

        # ANOVA using ols and anova_lm
        try:
            script = target_column + '~C(model_name)'
            model = ols(script, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)
        except Exception as e:
            print('ANOVA was skipped because', e)
        return

    for model_name in model_names:
        print('\n---')
        # Extract
        if model_name == 'MMVAE_CMNIST_OSCN':
            query = '(model_name in [@model_name] and target_modality == @target_modality)'
        elif model_name in ['VAE_CMNIST', 'VAE_OSCN']:
            query = 'model_name in [@model_name]'
        else:
            Exception
        df_wide = df.query(query)[['id', 'model_name'] + cols]
        df_wide['id'] = df_wide['id'].str.replace(
            'mmvae_cmnist_oscn_', ''
        ).str.replace(
            'vae_oscn_', ''
        ).str.replace(
            'vae_cmnist_', ''
        )
        print(df_wide)
        print(df_wide.describe())
        perform_anova_logtrans_model(
            df_wide=df_wide[cols],
            model_name=model_name,
            target_modality=target_modality,
            output_dir=output_dir,
        )

    return

def perform_one_sample(df,
                       target_modality,
                       output_dir):
    def perform_one_sample_model(df_wide,
                                 model_name,
                                 target_column,
                                 output_dir):
        # Plot
        fname = output_dir + '/anova_1sample_' + model_name + '_' + target_column + '.svg'
        plot_box(
            df_wide=df_wide,
            fname=fname,
            target_columns=[target_column],
            names=[target_column],
            widths=[1.0],
        )

        # t-test
        try:
            print(stats.ttest_1samp(
                a=df_wide[target_column],
                popmean=1/9,
                # alternative='greater',
            ))
        except Exception as e:
            print('One sample test was skipped because', e)
        return

    cols = [
        'reconst_' + str(target_modality) + 'x' + str(target_modality) + '_avg',
        'cross_' + str(1 - target_modality) + 'x' + str(target_modality) + '_avg',
        'mathematics_avg',
    ]
    target_conditions = {
        'MMVAE_CMNIST_OSCN': cols,
    }
    if target_modality == 0:
        target_conditions.update({
            'VAE_CMNIST': [
                'reconst_' + str(target_modality) + 'x' + str(target_modality) + '_avg',
                'mathematics_avg',
            ]})
    if target_modality == 1:
        target_conditions.update({
            'VAE_OSCN': [
                'reconst_' + str(target_modality) + 'x' + str(target_modality) + '_avg',
                'mathematics_avg',
        ]})

    for model_name in target_conditions.keys():
        print('\n---')
        if model_name == 'MMVAE_CMNIST_OSCN':
            query = '(model_name in [@model_name] and target_modality == @target_modality)'
        elif model_name in ['VAE_CMNIST', 'VAE_OSCN']:
            query = 'model_name in [@model_name]'
        else:
            Exception
        df_wide = df.query(query)[['id', 'model_name'] + cols]
        df_wide['id'] = df_wide['id'].str.replace(
            'mmvae_cmnist_oscn_', ''
        ).str.replace(
            'vae_oscn_', ''
        ).str.replace(
            'vae_cmnist_', ''
        )
        print(df_wide)
        print(df_wide.describe())
        for target_column in target_conditions[model_name]:
            perform_one_sample_model(
                df_wide=df_wide,
                model_name=model_name,
                target_column=target_column,
                output_dir=output_dir,
            )
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
    print(results.columns)
    print(results.describe())

    # convert `reconst_0x0_avg` value to `1x1` in VAE_OSCN.
    cols = ['id', 'model_name', 'reconst_0x0_avg', 'reconst_1x1_avg']
    print('Before converting:\n', results[cols])
    reconst_1x1_avg = results.loc[results['model_name'] == 'VAE_OSCN', 'reconst_1x1_avg']
    results.loc[results['model_name'] == 'VAE_OSCN', 'reconst_1x1_avg'] = \
        results.loc[results['model_name'] == 'VAE_OSCN']['reconst_0x0_avg']
    results.loc[results['model_name'] == 'VAE_OSCN', 'reconst_0x0_avg'] = reconst_1x1_avg
    print('After converting:\n', results[cols])

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

    print('===')
    print('analyzed data (all):')
    print(synthesized)
    synthesized.to_csv(output_dir + '/synthesized.csv')

    print('===')
    print('Anova between multi- and single-modal models...')
    cols = [
        'magnitude_avg',
        'magnitude_logmin_mean_avg',
        # 'magnitude_log_mean_avg',
        # 'magnitude_exp_mean_avg',
        # 'magnitude_pow_mean_avg',
        # 'magnitude_logmin_dist_avg',
        # 'magnitude_log_dist_avg',
        # 'magnitude_exp_dist_avg',
        # 'magnitude_pow_dist_avg',
        'cluster_avg',
        'mathematics_avg',
        'reconst_' + str(target_modality) + 'x' + str(target_modality) + '_avg'
    ]
    for col in cols:
        print('---')
        perform_anova(
            df=synthesized,
            target_modality=target_modality,
            target_column=col,
            output_dir=output_dir,
        )

    print('===')
    print('Anova for log transformation...')
    perform_anova_logtrans(
        df=synthesized,
        target_modality=target_modality,
        output_dir=output_dir,
    )

    print('===')
    print('One sample t-test...')
    perform_one_sample(
        df=synthesized,
        target_modality=target_modality,
        output_dir=output_dir,
    )

    print('===')
    print('scatter bettween cluster and magnitude')
    fname = output_dir + '/scatter_' + str(target_modality) + '.svg'
    plot_scatter(
        df=synthesized,
        fname=fname,
        model_name='MMVAE_CMNIST_OSCN',
        target_modality=target_modality,
        col_x='magnitude_avg',
        col_y='cluster_avg',
    )

    # print('===')
    # print('pairplot')
    # fname = output_dir + '/pairplot' + suffix + '.png'
    # sns.pairplot(
    #     synthesized,
    #     hue='model_name',
    # ).savefig(fname, format='png')
    # plt.close()

    print('===')
    print('all done.')

    return rslt

if __name__ == '__main__':
    from src.config import config_synthesizer as args
    synthesize(args=args)
