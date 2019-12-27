# original results
../bin/libFM -task r \
-train ../data/ml-10m/train0.csv.libfm \
-test ../data/ml-10m/test0.csv.libfm \
-dim ’1,1,16’ -iter 30 -out ../preds/ml-10m/test0.preds


# convert to libfm format
../scripts/triple_format_to_libfm.pl \
-in ../data/ml-10m/0.1_random/coop_train0.csv,../data/ml-10m/0.1_random/coop_test0.csv \
-target 2 -delete_column 3 -separator ","

# convert 0.01
for FRAC in 0.01 0.02 0.03 0.04 0.05
do
    ../scripts/triple_format_to_libfm.pl \
    -in ../data/ml-10m/${FRAC}_random/coop_train0.csv,../data/ml-10m/${FRAC}_random/coop_test0.csv \
    -target 2 -delete_column 3 -separator ","
done

# run
for FRAC in 0.01 0.02 0.03 0.04 0.05
do
    ../bin/libFM -task r \
    -train ../data/ml-10m/${FRAC}_random/coop_train0.csv.libfm \
    -test ../data/ml-10m/${FRAC}_random/coop_test0.csv.libfm \
    -dim ’1,1,16’ -iter 30 -out ../preds/ml-10m/${FRAC}_random/coop_test0.preds
done

../bin/libFM -task r \
-train ../data/ml-10m/0.1_random/orig_train0.csv.libfm \
-test ../data/ml-10m/0.1_random/orig_test0.csv.libfm \
-dim ’1,1,16’ -iter 30 -out ../preds/ml-10m/0.1_random/orig_test0.preds


FRAC=0.4
ENTITY=coop
../scripts/triple_format_to_libfm.pl \
    -in ../data/ml-10m/${FRAC}_random/${ENTITY}_train0.csv,../data/ml-10m/${FRAC}_random/${ENTITY}_test0.csv \
    -target 2 -delete_column 3 -separator ","

FRAC=0.4
ENTITY=coop
../bin/libFM -task r \
-train ../data/ml-10m/${FRAC}_random/${ENTITY}_train0.csv.libfm \
-test ../data/ml-10m/${FRAC}_random/${ENTITY}_test0.csv.libfm \
-dim ’1,1,16’ -iter 30 -out ../preds/ml-10m/${FRAC}_random/${ENTITY}_test0.preds