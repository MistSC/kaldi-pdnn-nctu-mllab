#!/bin/bash

# Copyright 2014    Yajie Miao   Carnegie Mellon University       Apache 2.0
# This is the script that trains DNN system over the filterbank features. It
# is to  be  run after run.sh. Before running this, you should already build
# the initial GMM model. This script requires a GPU card, and also the "pdnn"
# toolkit to train the DNN. The input filterbank features are with mean  and
# variance normalization.

# For more informaiton regarding the recipes and results, visit the webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

KALDI_DIR=~/kaldi-trunk/egs/timit/s5
working_dir=exp_pdnn/dnn_fbank
gmmdir=$KALDI_DIR/exp/tri3

# Specify the gpu device to be used
gpu=gpu

cmd=run.pl
. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

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
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;

echo =====================================================================
echo "                  DNN Pre-training & Fine-tuning                   "
echo =====================================================================
feat_dim=$(gunzip -c $working_dir/train.pfile.gz |head |grep num_features| awk '{print $2}') || exit 1;

echo $feat_dim
echo $num_pdfs

if [ ! -f $working_dir/dnn.ptr.done ]; then
  echo "RBM Pre-training"
  $cmd $working_dir/log/dnn.ptr.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/cmds/run_RBM.py --train-data "$working_dir/train.pfile.gz,partition=1000m,random=true,stream=false" \
                               --nnet-spec "$feat_dim:1024:1024:1024:1024:$num_pdfs" --wdir $working_dir \
                               --ptr-layer-number 4 --param-output-file $working_dir/dnn.ptr || exit 1;
  touch $working_dir/dnn.ptr.done
fi

# For SDA pre-training
#$pythonCMD pdnn/cmds/run_SdA.py --train-data "$working_dir/train.pfile.gz,partition=1000m,random=true,stream=false" \
#                          --nnet-spec "$feat_dim:1024:1024:1024:1024:$num_pdfs" \
#                          --1stlayer-reconstruct-activation "tanh" \
#                          --wdir $working_dir --param-output-file $working_dir/dnn.ptr \
#                          --ptr-layer-number 4 --epoch-number 5 || exit 1;

# To apply dropout, add "--dropout-factor 0.2,0.2,0.2,0.2" and change the value of "--lrate" to "D:0.8:0.5:0.2,0.2:8"
# Check run_timit/RESULTS for the results

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/cmds/run_DNN.py --train-data "$working_dir/train.pfile.gz,partition=1000m,random=true,stream=false" \
                          --valid-data "$working_dir/valid.pfile.gz,partition=200m,random=true,stream=false" \
                          --nnet-spec "$feat_dim:1024:1024:1024:1024:$num_pdfs" \
                          --ptr-file $working_dir/dnn.ptr --ptr-layer-number 4 \
                          --lrate "D:0.08:0.5:0.2,0.2:8" --momentum 0.9 \
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
