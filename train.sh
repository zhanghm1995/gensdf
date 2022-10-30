set -x

########### the official one
# python train.py -e config/gensdf/semi -b 64 -r last  # remove -r to train from scratch

########### our PU training script
python train.py -e config/genpu -b 32
