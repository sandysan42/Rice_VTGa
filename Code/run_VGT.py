from VoxelGridTracing import VoxelGridTracing
import open3d as o3d
import numpy as np
import pandas as pd
import pathlib

ply_list = []
path2 = pathlib.Path( 'Sample_Data/VGT')
ply_list = sorted(list(path2.glob('*.ply')))

result_array = np.array(['Name','Date','Long','Total','No.'])

for ply_file in ply_list:
    ply_file = str(ply_file)
    pcd = o3d.io.read_point_cloud(str(ply_file))
    VGT = VoxelGridTracing(pcd,Grid_size=0.01)
    try:
        VGT.auto_analysis()
        print(ply_file)
        plant_name = ply_file.split('/')[-1].split('_')[0]
        date = ply_file.split('/')[-1].split('_')[1]
        result_array = np.vstack([result_array,[plant_name,date,round(VGT.longest_length,6),round(VGT.total_length,6),len(VGT.branch_df)]])
        VGT.branch_df.to_csv('Sample_Output/VGT/{}_{}.csv'.format(plant_name,date))

        
        out_tsv = pd.DataFrame(result_array)
        out_tsv.to_csv('Sample_Output/VGT/Tracing_result_Median.csv',mode='w' ,sep=',',header=False,index=False)
    except:
        print('{} Error'.format(ply_file))
