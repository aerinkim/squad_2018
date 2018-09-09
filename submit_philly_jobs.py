import os
import argparse
"""Refine the philly submit script/template
Note that specifiy your vc/data path

If you use other dockers, please change the data_dir and other parameters

"""
parser = argparse.ArgumentParser()
parser.add_argument('--template', default='script/template.yaml')
parser.add_argument('--out', default='script/temp.yaml')
parser.add_argument('--epoches', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_size_eval', type=int, default=32)
parser.add_argument('--name', default='dcl_3_1000_200_2nd')
parser.add_argument('--main', default='train_v2.py')
parser.add_argument('--vc', default='pnrsy')
parser.add_argument('--data_dir', default='squad_at/docker', help='based on user root')
parser.add_argument('--model_dir', default='squad_at/docker', help='based on user root')
parser.add_argument('--cluster', default='rr1')
parser.add_argument('--user', default='ahkim')
parser.add_argument('--elmo_on', default='true', help='str "true" if want enable elmo')

args = parser.parse_args()
cluster = args.cluster
vc = args.vc
user = args.user
data_dir = args.data_dir
model_dir = args.model_dir

with open(args.template) as f:
    output_file=open(args.out,'w')
    for line in f:
        write_line = line.replace('VC_NAME',vc).replace('CLUSTER_NAME',cluster)
        write_line = write_line.replace('JOB_NAME', args.name)
        write_line = write_line.replace('USER_NAME', user)
        write_line = write_line.replace('DATA_DIR', data_dir)
        write_line = write_line.replace('MODEL_DIR', model_dir)
        write_line = write_line.replace('MAIN_FUNC', args.main)
        write_line = write_line.replace('EPOCHES', str(args.epoches))
        write_line = write_line.replace('BATCH_SIZE_EVAL', str(args.batch_size_eval))
        write_line = write_line.replace('BATCH_SIZE', str(args.batch_size))
        write_line = write_line.replace('ELMO_ON', '--elmo_on' if args.elmo_on == 'true' else '')
        if 'extra_args' in line and cluster!='rr1':
            write_line = '{} --batch_size {}\n'.format(write_line[:-2], args.batch_size)
        output_file.write(write_line)
    output_file.close()
input('check file')
cmd = 'pt run {} model_data/run_pht_{}_{}_{}'.format(args.out, cluster, vc, args.name)
print('CMD: {}'.format(cmd))
os.system(cmd)
