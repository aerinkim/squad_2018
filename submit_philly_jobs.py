import os
import argparse
"""Refine the philly submit script/template
Note that specifiy your vc/data path

If you use other dockers, please change the data_dir and other parameters

"""
parser = argparse.ArgumentParser()
parser.add_argument('--template', default='script/template.yaml')
parser.add_argument('--out', default='script/temp.yaml')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--name', type=str, default='san')
parser.add_argument('--main', type=str, default='train.py')
parser.add_argument('--vc', type=str, default='resrchvc')
parser.add_argument('--data_dir', default='data', required=True)
parser.add_argument('--cluster', default='gcr', required=True)
parser.add_argument('--user', type=str, required=True)

args = parser.parse_args()
cluster = args.cluster
vc = args.vc
user = args.user
data_dir = args.data_dir

with open(args.template) as f:
    output_file=open(args.out,'w')
    for line in f:
        write_line = line.replace('VC_NAME',vc).replace('CLUSTER_NAME',cluster)
        write_line = write_line.replace('USER_NAME', user)
        write_line = write_line.replace('DATA_DIR', data_dir)
        write_line = write_line.replace('MAIN_FUNC', args.main)
        if 'extra_args' in line and cluster!='rr1':
            write_line = '{} --batch_size {}\n'.format(write_line[:-2], args.batch_size)
        output_file.write(write_line)
    output_file.close()
input('check file')
cmd = 'pt run {} model_data/run_pht_{}_{}_{}'.format(args.out, cluster, vc, args.name)
print('CMD: {}'.format(cmd))
os.system(cmd)
