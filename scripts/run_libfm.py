import subprocess
import os
pre = './libFM/libfm-1.42.src'

fracs = [
    0.01,
    0.05, .1, .2, .3, .4, .5
]
seeds = [
    0, 1,
    2, 3, 4
]

for frac in fracs:
    for seed in seeds:
        format_call = f'{pre}/scripts/triple_format_to_libfm.pl'
        libfm_call = f'{pre}/bin/libFM'
        for company in [
            'small',
            'large'
        ]:
            scenario = f'{frac}_random{seed}'
            train_data = f'{pre}/data/ml-10m/{scenario}/{company}_train0.csv'
            formatted_train_data = f'{train_data}.libfm'
            test_data = f'{pre}/data/ml-10m/{scenario}/{company}_test0.csv'
            formatted_test_data = f'{test_data}.libfm'
            target_col = 2
            delete_col = 3
            separator = ','
            format_args = [str(x) for x in [
                format_call, "-in", f'{train_data},{test_data}',
                '-target', 2, '-delete_column', 3, '-separator', ',',
            ]]

            print(format_args)
            format_res = subprocess.check_output(format_args)
            for line in format_res.splitlines():
                print(line)

            dim = '1,1,32'
            iterations = 50
            preds_subdir = f'{pre}/preds/ml-10m/{scenario}/{dim}_{iterations}'


            for sub in [
                f'{pre}/preds/',
                f'{pre}/preds/ml-10m/',
                 f'{pre}/preds/ml-10m/{scenario}',
                preds_subdir,
            ]:
                if not os.path.isdir(sub):
                    os.mkdir(sub)


            if not os.path.isdir(preds_subdir):
                os.mkdir(preds_subdir)
            outfile = f'{preds_subdir}/{company}_test0.preds'
            libfm_args = [str(x) for x in [
                libfm_call, '-task', 'r', '-train', formatted_train_data,
                '-test', formatted_test_data, '-dim', dim, '-iter', iterations,
                '-out', outfile,
            ]]
            print(libfm_args)
            libfm_res = subprocess.check_output(libfm_args)
            for line in libfm_res.splitlines():
                print(line)


    

# orig bash version
# for FRAC in 0.01 0.02 0.03 0.04 0.05
# do
#     ${PRE}/bin/triple_format_to_libfm.pl \
#     -in ${PRE}/data/ml-10m/${FRAC}_random/small_train0.csv,../data/ml-10m/${FRAC}_random/small_test0.csv \
#     -target 2 -delete_column 3 -separator ","
# done

# ../bin/libFM -task r \
# -train ../data/ml-10m/${FRAC}_random/${ENTITY}_train0.csv.libfm \
# -test ../data/ml-10m/${FRAC}_random/${ENTITY}_test0.csv.libfm \
# -dim ’1,1,16’ -iter 30 -out ../preds/ml-10m/${FRAC}_random/${ENTITY}_test0.preds
