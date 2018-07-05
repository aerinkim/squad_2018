import os

cluster='rr1'
vc='pnrsy'

with open('test/search.yaml') as f:
    output_file=open('test/temp.yaml','w')
    for line in f:
        write_line=line.replace('VC_NAME',vc).replace('CLUSTER_NAME',cluster)
        if 'extra_args' in line and cluster!='rr1':
            write_line = write_line[:-2]+' --batch_size 24"\n'
        output_file.write(write_line)
        
    output_file.close()
input('check file')
os.system('pt run test/temp.yaml ../model_data/run_pht_%s' % (cluster))
