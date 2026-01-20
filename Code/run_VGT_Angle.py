from VoxelGridTracing import VoxelGridTracing
from AngleCalculation import AngleCalculation
import open3d as o3d
import numpy as np
import pandas as pd
import pathlib
import time
from datetime import timedelta
from tqdm import tqdm

ply_list = []
result_list = []
leaf_count = []
time_list = []

path = pathlib.Path('Sample_Data/Angle/')
plant_list = list(path.glob('*.ply'))

result_array = np.array(['Name','Date','Long','Total','No.'])

for ply_file in tqdm(plant_list):
    plant_name = str(ply_file).split('/')[-1].split('_')[0]
    date = str(ply_file).split('/')[-1].split('_')[1]
    start = time.time()
    pcd = o3d.io.read_point_cloud(str(ply_file))
    VGT = VoxelGridTracing(pcd,Grid_size=0.01)
    VGT.auto_analysis()
    VGT.compile_starting_voxel()
    result_array = np.vstack([result_array,[plant_name,date,round(VGT.longest_length,6),round(VGT.total_length,6),len(VGT.branch_df)]])
    out_tsv = pd.DataFrame(result_array)
    VGT.branch_df.to_csv('Sample_Output/VGT/{}_{}.csv'.format(plant_name,date))

    out_tsv = pd.DataFrame(result_array)
    out_tsv.to_csv('Sample_Output/VGT/Tracing_result_Median_Anglesample.csv',mode='w' ,sep='\t',header=False,index=False)
    try:
        tip_vox = VGT.starting_voxel
    except:
        print('No branch')
        continue

    AGC = AngleCalculation(pcd,Gridsize=0.01,tip_voxel_list=tip_vox)
    AGC.auto_analysis()

    result = AGC.x50

    end = time.time()
    if result != None:
        result_list.append(result)
        leaf_count.append(len(VGT.branch_df)+1)
        time_list.append(timedelta(seconds=end-start))
        ply_list.append(ply_file)
        df = pd.DataFrame(
            {'Name': ply_list,
            'Angle': result_list,
            'Leaf_no': leaf_count,
            'time' : time_list
            })
        df.to_csv('Sample_Output/Angle/VGTAC_result.csv')
        AGC.plot_sigmoid(save=True,savename='Sample_Output/Angle/'+str(pathlib.Path(ply_file).relative_to(path)).split('_')[0].split('/')[-1]+'_'+str(pathlib.Path(ply_file).relative_to(path)).split('_')[1]+'_sigmoidplot.png')
    else:
        continue
