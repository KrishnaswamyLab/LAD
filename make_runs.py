"""
Builds runs table for snakemake
"""
import os
import pandas as pd
import numpy as np

def build_runs():
    seeds = 10
    dataset = 'mnist'
    prefix = '/data/atong/anomaly/'
    model_types = ['rcae', 'conv', 'dcae', 
                   #'lipschitz_gp', 
#                   'shallow_isolation_forest', 
#                   'shallow_ocsvm', 
#                   'shallow_lof', 
#                   'lipschitz_gp_long',
                   #'lipschitz_gp_high_noise',
                   #'lipschitz_gp_higher_noise',
                   #'lipschitz_gp_big_high_noise',
                   #'lipschitz_gp_beta_zero',
                   #'lipschitz_gp_beta_zero_long',
#                   'lipschitz_gp_big',
#                   'lipschitz_gp_dense',
                   #'lipschitz_gp_spectral',
                   #'lipschitz_spectral_conv',
                   #'lipschitz_gp_patches',
                   #'lipschitz_gp_patches_small',
                   #'lipschitz_gp_patches_noise', 
                   'dsvdd',
    ]
    # model_types = ['lipschitz_gp']
    num_sevens = range(0, 501, 100)
    #num_sevens = range(0, 501, 20)
    batch_size = 128
    num_batches = 20000
    data = []
    for i in range(seeds):
        for ns in num_sevens:
            for model_type in model_types:
                path = os.path.join(prefix, dataset, model_type,
                                    str(i), str(ns), 'model.json')
                data.append([5000 - ns, ns, i, dataset, batch_size,
                             num_batches, path])
    columns = ['num_fives', 'num_sevens', 'seed', 'dataset',
               'batch_size', 'num_batches', 'path']
    return pd.DataFrame(data, columns=columns)

def build_runs2():
    seeds = 3
    dataset = 'mnist2'
    prefix = '/home/atong/data/anomaly/'
    model_types = [
                   'shallow_isolation_forest', 
                   'shallow_ocsvm', 
                   'shallow_lof', 
                   'rcae', 
                   'conv', 
                   'dcae', 
    #               'shallow_isolation_forest', 
    #               'shallow_ocsvm', 
    #               'shallow_lof', 
    #               'lipschitz_gp_beta_zero',
                   #'lipschitz_spectral_dense',
                   #'lipschitz_spectral_conv',
    #               'lipschitz_gp_spectral',
                   'lipschitz_gp_patches',
                  # 'lipschitz_spectral_patches', 
                  # 'lipschitz_gp_patches_small',
                  # 'lipschitz_gp_patches_noise', 
                   'dsvdd',
                   'ALOCC',
    ]
    digits = range(10)
    percent_corrupt = np.linspace(0, 0.1, 11)
    batch_size = 128
    num_batches = 20000
    data = []
    for i in range(seeds):
         for pc in percent_corrupt:
         #for pc in [0]:
            for d in digits:
                for model_type in model_types:
                    path = os.path.join(prefix, dataset, model_type,
                                        str(d), str(i), '%0.2f' % pc, 'model.json')
                    data.append([d, pc, i, dataset, batch_size,
                                 num_batches, path])
    columns = ['digit', 'percent_corrupt', 'seed', 'dataset',
               'batch_size', 'num_batches', 'path']
    return pd.DataFrame(data, columns=columns)

def build_runs3():
    seeds = 3
    dataset = 'vacs2'
    prefix = '/home/atong/data/anomaly/'
    model_types = ['lipschitz_gp_dense_vacs',
                   'dsvdd_dense', 
                   'dense_ae',
                   'dense_dae',
                   'dense_rae',
                   'dense_alocc',
                   'shallow_isolation_forest', 
                   'shallow_ocsvm', 
                   'shallow_lof']
    percent_corrupt = np.linspace(0, 0.03, 11)
    batch_size = 128
    num_batches = 20000
    data = []
    d = 0
    for i in range(seeds):
        for pc in percent_corrupt:
        #for pc in [0]:
            for model_type in model_types:
                path = os.path.join(prefix, dataset, model_type,
                                    str(d), str(i), '%0.3f' % pc, 'model.json')
                data.append([d, pc, i, dataset, batch_size,
                             num_batches, path])
    columns = ['digit', 'percent_corrupt', 'seed', 'dataset',
               'batch_size', 'num_batches', 'path']
    return pd.DataFrame(data, columns=columns)

def build_cifar():
    seeds = 3
    dataset = 'cifar2'
    prefix = '/home/atong/data/anomaly/'
    model_types = [
                   'shallow_isolation_forest', 
                   'shallow_ocsvm', 
                   'shallow_lof', 
                   'rcae', 
                   'conv', 
                   'dcae', 
    #               'shallow_isolation_forest', 
    #               'shallow_ocsvm', 
    #               'shallow_lof', 
    #               'lipschitz_gp_beta_zero',
                   #'lipschitz_spectral_dense',
                   #'lipschitz_spectral_conv',
    #               'lipschitz_gp_spectral',
                   'lipschitz_gp_patches',
                  # 'lipschitz_spectral_patches', 
                  # 'lipschitz_gp_patches_small',
                  # 'lipschitz_gp_patches_noise', 
                   'dsvdd',
                   'ALOCC',
    ]
    digits = range(10)
    percent_corrupt = np.linspace(0, 0.1, 11)
    batch_size = 256
    num_batches = 60000
    data = []
    for i in range(seeds):
         for pc in percent_corrupt:
         #for pc in [0]:
            for d in digits:
                for model_type in model_types:
                    path = os.path.join(prefix, dataset, model_type,
                                        str(d), str(i), '%0.2f' % pc, 'model.json')
                    data.append([d, pc, i, dataset, batch_size,
                                 num_batches, path])
    columns = ['digit', 'percent_corrupt', 'seed', 'dataset',
               'batch_size', 'num_batches', 'path']
    return pd.DataFrame(data, columns=columns)

if __name__ == '__main__':
    runs = build_runs2()
    runs.to_csv('runs/spectral_runs2.csv', index=False)
    runs = build_runs3()
    #runs = build_runs()
    print(runs.shape)
    runs.to_csv('runs/vacs_runs.csv', index=False)
    runs = build_cifar()
    runs.to_csv('runs/cifar_runs.csv', index=False)
    print(runs.head())
