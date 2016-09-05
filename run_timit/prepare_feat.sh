# By default, DNN inputs include 11 frames of filterbanks
for set in tr95 cv05; do
  if [ ! -f $working_dir/${set}.pfile.done ]; then
    steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --norm-vars true \
      --splice-opts "--left-context=5 --right-context=5" \
      --do_split true --stage 3 \
      $working_dir/data/train_$set ${gmmdir}_ali_$set $working_dir || exit 1
    ( cd $working_dir; mv concat.pfile ${set}.pfile; gzip ${set}.pfile; )
    touch $working_dir/${set}.pfile.done
  fi
done

