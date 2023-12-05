import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns

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
    print('analyzed data (selected):')
    if target_modality is None:
        synthesized = synthesized[columns]
    else:
        query = '(target_modality == @target_modality and model_name in ["MMVAE_CMNIST_OSCN"]) or (not model_name in ["MMVAE_CMNIST_OSCN"])'
        print(query)
        synthesized = synthesized.query(query)[columns]
    print(synthesized)

    print('---')
    print('pairplot')
    fname = output_dir + '/pairplot' + suffix + '.png'
    sns.pairplot(
        synthesized,
        hue='model_name',
    ).savefig(fname)

    return rslt

if __name__ == '__main__':
    from src.config import config_synthesizer as args
    synthesize(args=args)
