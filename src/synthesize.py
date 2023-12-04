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
        'MMVAE_OSCN',
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
            fname = os.path.join(
                experiment_dir,
                model_name,
                run_id,
                'analyse/0_1/analyse_result.csv'
            )
            print(fname, 'was loaded.')
            rslt = pd.read_csv(
                fname,
                header=0,
            )
            results.extend(rslt.to_dict(orient='records'))
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
    print(results)
    results.to_csv(args.output_dir + '/synthesized.csv')

    # Plot
    columns_selected = [
        'model_name',
        'reconst_avg',
        'cluster_avg',
        'magnitude_avg',
    ]
    sns.pairplot(
        results[columns_selected],
        hue='model_name',
    ).savefig(args.output_dir + '/pairplot.png')
    return

if __name__ == '__main__':
    from src.config import config_trainer_vae_oscn as args
    synthesize(args=args)
