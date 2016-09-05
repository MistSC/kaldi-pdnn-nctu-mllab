import ipdb
import numpy as np
import sys

from io_func.data_io import read_data_args
from io_func.data_io import read_dataset
from io_func.data_io import read_dataset_utt
from io_func.pfile_io import PfileDataRead


(train_pl,train_opts)=read_data_args('/home/shen/kaldipdnn/run_timit/test/train_tr95/*.pfile,random=false,stream=false,utt=true')

ur_train=read_dataset_utt(train_pl,train_opts)

(valid_pl,valid_opts)=read_data_args('/home/shen/kaldipdnn/run_timit/test/train_cv05/*.pfile,random=false,stream=false,utt=true')

ur_valid=read_dataset_utt(valid_pl,valid_opts)



utt_reader=[]
for l in train_pl:
  utt_reader.append(PfileDataRead([l], train_opts))

for ur in utt_reader:
  ur.initialize_read(first_time_reading = True)

def printdim(i):
  print utt_reader[i].feat_mats[0].shape
  print utt_reader[i].label_vecs[0].shape


ipdb.set_trace()
