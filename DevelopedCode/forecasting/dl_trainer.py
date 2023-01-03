import sys
sys.path.append('..')
from utils import *

ids = np.load('id_kept.npy')
              
def gen_cmd(id, gpu):
    cmd = f'python training_cluster.py --ID {id} --GPU {gpu} --CLUSTERING spectral &'
    return cmd
    
import os 

n_training = 5
n_gpu = 2

import time
t = time.time()
for i in range(len(ids)//(n_gpu*n_training)):
    tot_cmd = ''
    #GPU 0
    for j in range(n_training):
        if not os.path.exists(f'res/res_{ids[n_training*i+j]}.npy'):
            tot_cmd += gen_cmd(ids[n_training*i+j], 0)
            tot_cmd += '\n'
    #GPU 1
    for j in range(n_training):
        if not os.path.exists(f'res/res_{ids[len(ids)//2+n_training*i+j]}.npy'):
            tot_cmd += gen_cmd(ids[len(ids)//2+n_training*i+j], 1)
            tot_cmd += '\n'
    #GPU 3
    for j in range(n_training):
        if not os.path.exists(f'res/res_{ids[len(ids)//2+n_training*i+j]}.npy'):
            tot_cmd += gen_cmd(ids[len(ids)//2+n_training*i+j], 2)
            tot_cmd += '\n'
    print(tot_cmd)
    os.system(tot_cmd)
    print('finish training ... wait for others ...')
    time.sleep(60*8)
