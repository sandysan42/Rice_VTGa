import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sc_utils import CalVector
from scipy.optimize import curve_fit


class AngleCalculation:
    def __init__(self,pcd,Gridsize,tip_voxel_list,origin_point=None,cut=2,name=None):
        self.pcd = pcd
        self.Gridsize = Gridsize
        self.voxel_list = tip_voxel_list
        self.cut = cut
        self.name= name
        self.origin = origin_point

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

    def Get_coord(self):
        self.coord_list = []
        self.Grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd,self.Gridsize)
        for i in self.voxel_list:
            coord = self.Grid.get_voxel_center_coordinate(i)
            self.coord_list.append(coord)
        return self

    def set_origin_point(self,lower_bound=0.2,upper_bound=0.21,min_z=0.15):
        data = pd.DataFrame(np.asarray(self.pcd.points),columns=["x","y",'z'])
        sliced_data = data[data['z'].between(lower_bound,upper_bound,inclusive='left')]
        self.origin = (sliced_data.x.mean(),sliced_data.y.mean(),min_z)

    def Get_angle(self):
        self.angle_df = pd.DataFrame(columns=['dist','x','y','z'])
        for i in self.coord_list:
            # print(i[0])
            if i[2] < 0.2:
                continue
            angle = CalVector(self.origin,i)
            # print(i[1])
            angle = pd.DataFrame(angle).T
            angle.columns = ['dist','x','y','z']
            self.angle_df = pd.concat([self.angle_df,angle])

        return self

    def Group_angle(self):
        age_groups = ["{0} - {1}".format(i, i + self.cut-1) for i in range(1, 90, self.cut)]
        cats = pd.Categorical(age_groups)
        self.angle_df["Angle range"] = pd.cut(self.angle_df.z, range(0, 91, self.cut), right=False, labels=cats)
        return self

    def sigmoid(self,x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return (y)

    def sigmoid_r(self,L ,x0, k, b):
        x = x0 - (1 / k) * np.log((L - (0.5 - b)) / (0.5 - b))
        return (x)

    def calculate_cdf(self):
        self.stats_df = self.angle_df.groupby('Angle range')['Angle range'].agg('count').pipe(pd.DataFrame).rename(columns = {'Angle range': 'frequency'})
        self.stats_df.reset_index(inplace=True)
        self.stats_df["start_value"] = self.stats_df["Angle range"].str.split(" - ").str[0].astype(int)
        self.stats_df.sort_values(by=["start_value"],inplace=True)
        self.stats_df.reset_index(inplace=True,drop=True)
        # PDF
        self.stats_df['pdf'] = self.stats_df['frequency'] / sum(self.stats_df['frequency'])

        # CDF
        self.stats_df['cdf'] = self.stats_df['pdf'].cumsum()

        return self

    def calculate_y50(self):
        skip = 0
        self.p0 = [max(self.stats_df.cdf), np.median(self.stats_df.start_value),1,min(self.stats_df.cdf)] # this is an mandatory initial guess

        try:
            self.popt, self.pcov = curve_fit(self.sigmoid, self.stats_df.start_value, self.stats_df.cdf,self.p0, method='dogbox')
        except:
            try:
                self.popt,self.pcov = curve_fit(self.sigmoid, self.stats_df.start_value, self.stats_df.cdf,self.p0, method='dogbox',maxfev=5000)
            except:
                try:
                    self.popt, self.pcov = curve_fit(self.sigmoid, self.stats_df.start_value, self.stats_df.cdf,self.p0, method='dogbox',bounds=(0,90))
                except ValueError:
                    self.x50 = None
                    skip = 1
        if not skip:
            self.x50 = self.sigmoid_r(*self.popt)

        return self

    def plot_sigmoid(self,figsize=(6,5),save=False,savename=None):
        plt.figure(figsize=figsize)
        x = np.linspace(0, max(self.stats_df.start_value), 100)
        y = self.sigmoid(x, *self.popt)

        y50 = 0.5
        x50 = self.sigmoid_r(*self.popt)
        # print(x50)

        plt.hlines(0.5,0,x50,colors='k',linestyles=':')
        plt.vlines(x50,0,y50,colors='k',linestyles=':')

        # plt.text(0,0.6,'y50 = {}'.format(round(x50,2)))

        plt.plot(self.stats_df.start_value,  self.stats_df.cdf, 'o', label='data')
        plt.plot(x,y, label='fit')
        plt.ylim(0, 1)
        plt.xlim(0,90)
        # plt.legend(loc='best')
        if self.name != None:
            plt.title(self.name)
        if save:
            plt.savefig(savename)
        else:
            plt.show()

    def auto_analysis(self):
        self.clean_pcd()
        if self.origin == None:
            self.set_origin_point()
        self.Get_coord()
        self.Get_angle()
        self.Group_angle()
        self.calculate_cdf()
        self.calculate_y50()

        return self
