#!/bin/bash

# Copyright 2014     Yajie Miao   Carnegie Mellon University       Apache 2.0
# This script trains  Maxout Network  models over fMLLR features. It is to be
# run after run.sh. Before running this, you should already build the initial
# GMM model. This script requires a GPU, and also the "pdnn" toolkit to train
# the DNN. 

# We implement the <Maxout> activation function, based on Kaldi "revision 4960".
# Please follow the following steps:
# 1. Go to /path/to/kaldi/src/nnet and *backup* nnet-component.h, nnet-component.cc, nnet-activation.h
# 2. Download these 3 files from here:
#    http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/nnet-component.h
#    http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/nnet-component.cc
#    http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/nnet-activation.h
# 3. Recompile Kaldi

# For more informaiton regarding the recipes and results, visit the webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn/dnn_maxout
gmmdir=exp/tri3

# Specify the gpu device to be used
gpu=gpu

cmd=run.pl
. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

# At this point you may want to make sure the directory $working_dir is
# somewhere with a lot of space, preferably on the local GPU-containing machine.
if [ ! -d pdnn ]; then
  echo "Checking out PDNN code."
  svn co https://github.com/yajiemiao/pdnn/trunk pdnn
fi

if [ ! -d steps_pdnn ]; then
  echo "Checking out steps_pdnn scripts."
  svn co https://github.com/yajiemiao/kaldipdnn/trunk/steps_pdnn steps_pdnn
fi

if ! nvidia-smi; then
  echo "The command nvidia-smi was not found: this probably means you don't have a GPU."
  echo "(Note: this script might still work, it would just be slower.)"
fi

# The hope here is that Theano has been installed either to python or to python2.6
pythonCMD=python
if ! python -c 'import theano;'; then
  if ! python2.6 -c 'import theano;'; then
    echo "Theano does not seem to be installed on your machine.  Not continuing."
    echo "(Note: this script might still work, it would just be slower.)"
    exit 1;
  else
    pythonCMD=python2.6
  fi
fi

mkdir -p $working_dir/log

! gmm-info $gmmdir/final.mdl >&/dev/null && \
   echo "Error getting GMM info from $gmmdir/final.mdl" && exit 1;

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;

echo =====================================================================
echo "           Data Split & Alignment & Feature Preparation            "
echo =====================================================================
# Split training data into traing and cross-validation sets for DNN
if [ ! -d data/train_tr95 ]; then
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train data/train_tr95 data/train_cv05 || exit 1
fi
# Alignment on the training and validation data
for set in tr95 cv05; do
  if [ ! -d ${gmmdir}_ali_$set ]; then
    steps/align_fmllr.sh --nj 16 --cmd "$train_cmd" \
      data/train_$set data/lang $gmmdir ${gmmdir}_ali_$set || exit 1
  fi
done

# Dump fMLLR features. "Fake" cmvn states (0 means and 1 variance) are applied. 
for set in tr95 cv05; do
  if [ ! -d $working_dir/data/train_$set ]; then
    steps/nnet/make_fmllr_feats.sh --nj 16 --cmd "$train_cmd" \
      --transform-dir ${gmmdir}_ali_$set \
      $working_dir/data/train_$set data/train_$set $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data/train_$set $working_dir/_log $working_dir/_fmllr || exit 1;
  fi
done
for set in dev test; do
  if [ ! -d $working_dir/data/$set ]; then
    steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
      --transform-dir $gmmdir/decode_$set \
      $working_dir/data/$set data/$set $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data/$set $working_dir/_log $working_dir/_fmllr || exit 1;
  fi
done

echo =====================================================================
echo "               Training and Cross-Validation Pfiles                "
echo =====================================================================
# By default, DNN inputs include 11 frames of fMLLR
for set in tr95 cv05; do
  if [ ! -f $working_dir/${set}.pfile.done ]; then
    steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --norm-vars false \
      --splice-opts "--left-context=5 --right-context=5" \
      $working_dir/data/train_$set ${gmmdir}_ali_$set $working_dir || exit 1
    ( cd $working_dir; mv concat.pfile ${set}.pfile; gzip ${set}.pfile; )
    touch $working_dir/${set}.pfile.done
  fi
done
# Rename pfiles to keep consistency
( cd $working_dir;
  ln -s tr95.pfile.gz train.pfile.gz; ln -s cv05.pfile.gz valid.pfile.gz
)

echo =====================================================================
echo "                  DNN Pre-training & Fine-tuning                   "
echo =====================================================================
# Here we use maxout networks. When using maxout, we need to reduce the learning rate. To apply dropout,
# add "--dropout-factor 0.2,0.2,0.2,0.2" and change the value of "--lrate" to "D:0.1:0.5:0.2,0.2:8"
# Check run_timit/RESULTS for the results

# The network structure is set in the way that this maxout network has approximately the same number of
# parameters as the DNN model in run-dnn.sh

feat_dim=$(gunzip -c $working_dir/train.pfile.gz |head |grep num_features| awk '{print $2}') || exit 1;

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/cmds/run_DNN.py --train-data "$working_dir/train.pfile.gz,partition=1000m,random=true,stream=false" \
                          --valid-data "$working_dir/valid.pfile.gz,partition=200m,random=true,stream=false" \
                          --nnet-spec "$feat_dim:625:625:625:625:$num_pdfs" \
                          --activation "maxout:3" \
                          --lrate "D:0.008:0.5:0.2,0.2:8" \
                          --wdir $working_dir --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
fi

echo =====================================================================
echo "                           Decoding                                "
echo =====================================================================
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph
  steps_pdnn/decode_dnn.sh --nj 12 --scoring-opts "--min-lmwt 1 --max-lmwt 8" --cmd "$decode_cmd" \
    $graph_dir $working_dir/data/dev ${gmmdir}_ali_tr95 $working_dir/decode_dev || exit 1;
  steps_pdnn/decode_dnn.sh --nj 12 --scoring-opts "--min-lmwt 1 --max-lmwt 8" --cmd "$decode_cmd" \
    $graph_dir $working_dir/data/test ${gmmdir}_ali_tr95 $working_dir/decode_test || exit 1;
  touch $working_dir/decode.done
fi

echo "Finish !!"
