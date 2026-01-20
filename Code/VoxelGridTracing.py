import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sc_utils import CalVector,counter_cosine_similarity,create_3d_cube
from scipy.spatial import ConvexHull
# from scipy.spatial.qhull import QhullError
from collections import Counter
import networkx as nx
# from tqdm import tqdm

"""
TODO LIST

1. Find origin point of each model?
2. Change result fromt tracing for more compatibility with a networkX library (For branching test) /
3. Add branching capability -> If we use old code , the problem is overlapping will occur again. Need to find new way.
4. ???
5. Profit
"""


class VoxelGridTracing:
    def __init__(self,pcd,Grid_size):
        self.pcd = pcd
        self.Gridsize = Grid_size
        self.traced_list = []
        self.origin_point = [-0.01,-0.01,0.19]
        self.skip_vox = []
        self.unique_vox = []
        self.flattened = []
        self.full_connected_list = []
        self.graph_list = []
        self.end_node_list = []

    def clean_pcd(self,lower_bound=0.2):
        """
        Clean up pointcloud by cropping to lower bound.
        This is to prevent any mulfunction in tracing algorithm
        Input: raw pcd
        output: cropped pcd
        """
        array_point = np.asarray(self.pcd.points)
        array_color = np.asarray(self.pcd.colors)
        array = np.concatenate([array_point,array_color],1)
        crop_bool = (array[:,2] > lower_bound)
        array_crop=array[crop_bool]
        array_crop_points = array_crop[:,0:3]
        array_crop_colors = array_crop[:,3:6]

        #numpy->open3d
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points= o3d.utility.Vector3dVector(array_crop_points)
        self.pcd.colors= o3d.utility.Vector3dVector(array_crop_colors)

        return self
    def set_origin_point(self,lower_bound=0.2,upper_bound=0.21):
        data = pd.DataFrame(np.asarray(self.pcd.points),columns=["x","y",'z'])
        sliced_data = data[data['z'].between(lower_bound,upper_bound,inclusive='left')]
        self.origin_point = (sliced_data.x.mean(),sliced_data.y.mean(),0.19)
    def create_voxel_Grid(self):
        """
        Create Voxel self.Grid form point cloud
        input: o3d pointcloud
        output: voxel self.Grid dataframe
        
        """
        self.Grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd,self.Gridsize)
        full_df = pd.DataFrame(columns=['x','y','z'])
        for i in self.Grid.get_voxels():
            bbs = np.asarray(self.Grid.get_voxel_bounding_points(i.grid_index))
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbs))
            df = pd.DataFrame(np.asarray(self.pcd.crop(aabb).points),columns=["x","y",'z'])
            # print(i.grid_index)
            df['index_x'] = i.grid_index[0]
            df['index_y'] = i.grid_index[1]
            df['index_z'] = i.grid_index[2]

            full_df = pd.concat([full_df,df])
        group = full_df.groupby(['index_x','index_y','index_z'])
        voxel_Grid_list = []
        for i,df in group:
            num_p = len(df)
            row = [i,i[0],i[1],i[2],df.mean()['x'],df.mean()['y'],df.mean()['z'],num_p]
            voxel_Grid_list.append(row)
        self.voxel_Grid_df = pd.DataFrame(voxel_Grid_list,columns=['Index','Ix','Iy','Iz','ux','uy','uz','num_points'])
        self.voxel_Grid_df.sort_values('Iz',ascending=False,inplace=True)
        self.voxel_Grid_df.reset_index(inplace=True,drop=True)

        return self

    def find_starting_point(self):
        """
        Use convex hull to automatically find starting point of tracing
        input: o3d pointcloud
        output: list of starting point (x,y,z tuple)
        """
        self.initial_starting_point = []

        hull, _ = self.pcd.compute_convex_hull()
        hull_vertices = np.asarray(hull.vertices)
        for i in hull_vertices:
            index = self.Grid.get_voxel(i)
            index = tuple(index)
            # print(index)
            if index not in self.initial_starting_point:
                # print(index)
                self.initial_starting_point.append(index)
        return self
    
    def trace_from_point(self,start_point):
        """
        """
        if start_point == 'main':
            search_point_list = self.initial_starting_point
        else:
            search_point_list = self.remaining_starting_point
        for search_point in search_point_list:
            if start_point != 'main':
                search_point = (search_point[0],search_point[1],search_point[2])
            if search_point[2] < 5:
                    continue
            penalty_multiplier = 0.001
            selected_voxel = []
            connect_list = []
            selected_voxel.append(search_point)
            # vector_to_origin = CalVector((i_ux , i_uy, i_uz),self.origin_point)

            while search_point[2] != 0:
                score_list = []
                empty_count = 0
                skip_flag = 0
                current_point = search_point
                best_score = 0
                search_arr = create_3d_cube(search_point[0],search_point[1],search_point[2])
                try:
                    row = self.voxel_Grid_df[self.voxel_Grid_df['Index']==search_point]
                except ValueError:
                    search_point = (search_point[0],search_point[1],search_point[2])
                    row = self.voxel_Grid_df[self.voxel_Grid_df['Index']==search_point]
                    # row = self.voxel_Grid_df[self.voxel_Grid_df['Index']==(search_point[0],search_point[1],search_point[2])]

                i_ux , i_uy, i_uz = row['ux'].values[0],row['uy'].values[0],row['uz'].values[0]

                vector_to_origin = CalVector((i_ux , i_uy, i_uz),self.origin_point)
                for indx in search_arr:
                    df = self.voxel_Grid_df[self.voxel_Grid_df['Index']==indx]
                    if indx == search_point:
                        continue
                    if df.empty:
                        empty_count += 1
                        continue
                    vector = CalVector((i_ux , i_uy, i_uz),(df['ux'].values[0],df['uy'].values[0],df['uz'].values[0]))
                    dDx = abs(vector_to_origin[1]-vector[1])
                    dDy = abs(vector_to_origin[2]-vector[2])
                    dDz = abs(vector_to_origin[3]-vector[3])
                    dDVa = dDx+dDy+dDz
                    score = df['num_points'].values[0] * (1/dDVa)
                    if indx in selected_voxel:
                        count = selected_voxel.count(indx)
                        score *= (penalty_multiplier**count)
                    if score > best_score:
                        best_score = score
                        search_point = indx
                    score_list.append(score)
                if sum(score_list) < 0.000001:
                    skip_flag = 1
                    self.unique_vox += list(set(selected_voxel))
                    break
                if empty_count == 26:
                    skip_flag = 1
                    self.unique_vox += list(set(selected_voxel))
                    self.isolate_vox += 1
                    break
                else:
                    selected_voxel.append(search_point)
                    connect_list.append([current_point,search_point,vector[0],vector[1],vector[2],vector[3]])
            counterA = Counter(selected_voxel)
            # print(len(counterA))
            # Check duplication/subset of leaf            
            if not skip_flag:
                check = 0
                if len(self.traced_list) > 0:
                    for traced_item in self.traced_list:
                        counterB = Counter(traced_item)
                        similarity = counter_cosine_similarity(counterA,
                            counterB)
                        if similarity < 1.0:
                            if set(counterB).issubset(set(counterA)):
                                # traced_list.remove(traced_item)
                                check=0
                            if set(counterA).issubset(set(counterB)):
                                check=0
                                break
                        if similarity > 0.8:
                            #How to proceed?
                            check=0
                            break
                        else:
                            check=1
                else:
                    check=1

                if check:
                    artificial_layer = [connect_list[-1][1],(0,0,-1),0.0001,.1,0.1,0.1]
                    connect_list.append(artificial_layer)
                    self.traced_list.append(selected_voxel)    #print(indx , df['num_points'].values[0],dDVa,score)
                    self.full_connected_list.append(connect_list)
                    self.generate_graph(connect_list=connect_list,selected_voxel=selected_voxel)
                else:
                    continue
            else:
                continue
        return self
    
    def create_flattend_unique_list(self):
        """
        Create unique voxel list
        """
        self.flattened = [val for sublist in self.traced_list for val in sublist]
        # self.flattened.append(new_flattened)
        self.unique_vox += list(set(self.flattened))
        return self
    
    def find_next_starting_point(self):
        """
        Removed traced voxel and find new tracing starting point from 3D model
        """
        remaining_voxel = np.zeros(shape=(3,),dtype=int)
        for i in self.Grid.get_voxels():
            for k in create_3d_cube(i.grid_index[0],i.grid_index[1],i.grid_index[2]):
                if k in self.unique_vox:
                    self.skip_vox.append((i.grid_index[0],i.grid_index[1],i.grid_index[2]))
            if tuple(i.grid_index) in self.unique_vox:
                continue
            elif tuple(i.grid_index) in self.skip_vox:
                continue
            else:
                voxel = np.array([i.grid_index[0],i.grid_index[1],i.grid_index[2]])
                remaining_voxel = np.vstack([remaining_voxel,voxel])
        remaining_voxel = remaining_voxel[1:]
        remain_hull = ConvexHull(remaining_voxel)
        self.remaining_starting_point = remain_hull.points[remain_hull.vertices]
        return self
    
    def generate_graph(self,connect_list,selected_voxel,combine=False):
        """
        Create graph line using networkx fuction
        input: list of connected node with edge of distance and angle information
        output: append network graph to graph list, can combine into full graph if flag is set to true
        """
        coord = pd.DataFrame(connect_list,columns=['A','B','dist','Dx','Dy','Dz'])
        G = nx.from_pandas_edgelist(coord,'A','B','dist')
        T = nx.dfs_tree(G,(0,0,-1))

        endnodes = [x for x in T.nodes() if T.out_degree(x)==0 and T.in_degree(x)==1]
        A = nx.shortest_path(T,selected_voxel[-1],selected_voxel[0])
        for endnode in endnodes:
            if endnode == selected_voxel[0]:
                continue
            else:
                T.remove_nodes_from(list(set(selected_voxel)-set(A)))
                G.remove_nodes_from(list(set(selected_voxel)-set(A)))
        self.graph_list.append(G)
        if combine:
            self.full_graph  = nx.compose_all(self.graph_list)
        return self
       
    def combine_graph(self):
        """
        Combine graph in graph list into 1 network graph
        """
        self.full_graph  = nx.compose_all(self.graph_list)
        return self
    
    def analyse_graph(self):
        """
        Analyse graph to output longest lenght (Main leaf/branch) and sum of all branch lenght as total lenght

        """
        T = nx.dfs_tree(self.full_graph,(0,0,-1))
        self.longest_path = nx.dag_longest_path(T)
        self.longest_length = nx.path_weight(self.full_graph,nx.dag_longest_path(T),weight='dist')
        self.total_length = self.full_graph.size(weight='dist')
        return self
    
    def branching(self):
        """
        From networkX graph, create branching of leaves/stem from directional graph, and result in lenght of each branch 
        """
        branch_array = np.array(['Node','Branch','Length'])
        T = nx.dfs_tree(self.full_graph,(0,0,-1))
        G_dict=nx.dfs_successors(self.full_graph,(0,0,-1))
        predec = nx.dfs_predecessors(self.full_graph,(0,0,-1))

        self.end_node_list = []
        end_node = None
        for k,v in G_dict.items():
            if len(v) > 1:
                for i in v:
                    predecessor = ''
                    max_length = 0
                    longest_branch = []
                    O= list(nx.descendants(T,i))
                    if nx.dag_longest_path(T)[-1] not in O:
                        for node in O:
                            predecessor = predec[node]
                            for target in O:
                                try:
                                    A = nx.shortest_path(T,predec[predecessor],target)
                                    A_len = len(A)
                                except nx.NetworkXNoPath :
                                    continue
                                if A_len > max_length:
                                    max_length = A_len
                                    longest_branch = A
                                    start_node = predec[predecessor]
                                    end_node = target
                                    A_len = len(A)
                    if end_node is not None:
                        if end_node not in self.end_node_list:
                            if predecessor != '':
                                self.end_node_list.append(end_node)
                                branch = np.array([(start_node,end_node),longest_branch,nx.path_weight(self.full_graph,longest_branch,weight='dist')],dtype=object)
                                branch_array = np.vstack([branch_array,branch])

        self.branch_df = pd.DataFrame(branch_array[1:,:],columns=['Node','Branch','Length'])
        return self
    
    def draw_graph(self,save=False,name=''):
        plt.figure(figsize=(10,7))
        pos = nx.kamada_kawai_layout(self.full_graph)
        nx.draw_networkx(self.full_graph,pos,arrows=True,node_size=100,alpha=1,font_size=5)
        if save:
            plt.savefig('{}_graph.jpg'.format(name))
        else:
            plt.show()

    def visualize_voxel_3d(self,figsize=(7,10),a1=0,a2=0,pointsize=0.1):
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection="3d")

        ax.scatter(
                    self.voxel_Grid_df['ux'],self.voxel_Grid_df['uy'],self.voxel_Grid_df['uz'],c='k',s= self.voxel_Grid_df['num_points']*pointsize
                    )
            # aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbs))
        ax.view_init(a1, a2)
        plt.show()

    def visualize_pcd_3d(self,figsize=(7,10),a1=0,a2=0,save=False,name=''):
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection="3d")
        Grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd,self.Gridsize)
        full_df = pd.DataFrame(columns=['x','y','z'])
        skip_vox = []
        for i in Grid.get_voxels():
            for k in create_3d_cube(i.grid_index[0],i.grid_index[1],i.grid_index[2]):
                if k in self.unique_vox:
                    # ax.scatter(i.grid_index[0],i.grid_index[1],i.grid_index[2],c='g',s=10)

                    skip_vox.append((i.grid_index[0],i.grid_index[1],i.grid_index[2]))
                
                    # break
            if tuple(i.grid_index) in self.unique_vox:
                ax.scatter(i.grid_index[0],i.grid_index[1],i.grid_index[2],c='r',s=5)
                # continue
            elif tuple(i.grid_index) in skip_vox:
                ax.scatter(i.grid_index[0],i.grid_index[1],i.grid_index[2],c='g',s=5)
            else:
                bbs = np.asarray(Grid.get_voxel_bounding_points(i.grid_index))

                aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbs))
                df = pd.DataFrame(np.asarray(self.pcd.crop(aabb).points),columns=["x","y",'z'])
                # ax.scatter(df['x'],df['y'],df['z'],s=1)
                ax.scatter(i.grid_index[0],i.grid_index[1],i.grid_index[2],s=1)
        # ax.scatter(i.grid_index[0],i.grid_index[1],i.grid_index[2],s=1)
        # df = pd.DataFrame(np.asarray(pcd.crop(aabb).points),columns=["x","y",'z'])
        # print(i.grid_index)

        try:
            for j in self.remaining_starting_point:
                ax.scatter(j[0],j[1],j[2],c='k',s=10,marker='*')
        except:
            try:
                for j in self.initial_starting_point:
                    ax.scatter(j[0],j[1],j[2],c='k',s=10,marker='*')  
            except:
                pass       

        ax.view_init(a1, 
                    a2)
        if save:
            plt.savefig('{}_graph.jpg'.format(name))
        else:
            plt.show()

    def compile_starting_voxel(self,origin = "origin"):
        """
        This function is for sending voxel list of tips to angle calculator class
        if origin is set to 'origin', the output will send only tips (This is for forage crop such as rice, grass)
        if origin is set to 'branch', the output will send both tips and origin point of branch (This is for standing crop such as soybean) //Not implemented yet//
        """
        if len(self.end_node_list) > 0:
            self.starting_voxel = self.end_node_list
            self.starting_voxel.append(self.longest_path[-1])

            return self
        else:
            print('Cannot compile starting point, Please create branching first')

    def auto_analysis(self):
        loop_flag = 1
        latest_RMP_count = 0
        self.clean_pcd()
        self.set_origin_point()
        self.create_voxel_Grid()
        self.find_starting_point()
        self.trace_from_point('main')
        while loop_flag:
            self.create_flattend_unique_list()
            try:
                self.find_next_starting_point()
            except:
                break
            loop_flag = 0
            if latest_RMP_count != len(self.remaining_starting_point):
                loop_flag = 1
                latest_RMP_count = len(self.remaining_starting_point)
            else:
                latest_RMP_count = len(self.remaining_starting_point)
            self.trace_from_point('next')
        self.combine_graph()
        self.analyse_graph()
        self.branching()


