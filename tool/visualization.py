
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

class KMeansVisualizer:
    def __init__(self, poi_gdf_label:gpd):
        """
        KMeansVisualizer类，用于可视化聚类结果。
        :param kmeans: BKMeans实例
        """
        self.poi_gdf_label = poi_gdf_label
        self.statistics()

    def statistics(self):
        # 计算每个cluster的点数
        self.cluster_counts = self.poi_gdf_label['cluster'].value_counts().sort_index()
        print("每个cluster的点数:")
        print(self.cluster_counts)

        # 计算点数偏差（最大-最小）
        self.count_deviation = self.cluster_counts.max() - self.cluster_counts.min()
        print(f"点数偏差（最大-最小）: {self.count_deviation}")


    def showResults(self, nameDataSet):
        """
        绘制聚类结果和凸包，并保存为图片。
        :param nameDataSet: 数据集名称
        :param run: 运行编号
        :param timeInSec: 运行耗时
        """
        # 获取数据边界
        self.fig, self.ax = plt.subplots(figsize=(15, 10)) 
        x_min, x_max = self.poi_gdf_label.geometry.x.min(), self.poi_gdf_label.geometry.x.max()
        y_min, y_max = self.poi_gdf_label.geometry.y.min(), self.poi_gdf_label.geometry.y.max()
        x_length = x_max - x_min
        y_length = y_max - y_min
        if x_length > y_length:
            y_correction = (x_length - y_length) / 2.0
            y_min -= y_correction
            y_max += y_correction
        else:
            x_correction = (y_length - x_length) / 2.0
            x_min -= x_correction
            x_max += x_correction

        # 设置图形范围
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # 绘制数据点
        points_x = self.poi_gdf_label.geometry.x
        points_y = self.poi_gdf_label.geometry.y
        self.ax.scatter(points_x, points_y, c='black', s=10, alpha=0.6)

        # 计算每个簇的凸包
        cluster_labels = self.poi_gdf_label['cluster'].unique()
        for i in cluster_labels:
            cluster_points = self.poi_gdf_label[self.poi_gdf_label['cluster'] == i]
            if len(cluster_points) > 2:
                hull_points = self.computeConvexHull(cluster_points)
                hull_x = [p[0] for p in hull_points]
                hull_y = [p[1] for p in hull_points]
                # 闭合凸包
                hull_x.append(hull_x[0])
                hull_y.append(hull_y[0])
                self.ax.plot(hull_x, hull_y, 'k-', linewidth=1)

        # 绘制聚类中心（用每个簇的几何中心）
        for i in cluster_labels:
            cluster_points = self.poi_gdf_label[self.poi_gdf_label['cluster'] == i]
            x = cluster_points.geometry.x.mean()
            y = cluster_points.geometry.y.mean()
            clusterSize = len(cluster_points)
            meanPointsPerCluster = len(self.poi_gdf_label) / len(cluster_labels)
            if clusterSize < np.floor(meanPointsPerCluster):
                color = 'green'  # 点数过少
            elif clusterSize > np.ceil(meanPointsPerCluster):
                color = 'red'    # 点数过多
            else:
                color = 'blue'   # 点数适中
            self.ax.scatter(x, y, c=color, s=100, marker='o')

        # 设置标题
        title = f'Num_cluster = {len(cluster_labels)}, Total = {len(self.poi_gdf_label)}, Deviation = {self.count_deviation}'
        self.ax.set_title(title)

        # 移除坐标轴
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(False)

        # 保存图形
        plt.savefig(f'/Users/chendi/IdeaProjects/regionOpt/method/convexHulls/{nameDataSet}_ConvexHull.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def computeConvexHull(self, points_gdf):
        """
        计算点集的凸包，输入为geopandas的GeoSeries或GeoDataFrame。
        :param points_gdf: GeoSeries或GeoDataFrame，点类型
        :return: 凸包上的点坐标列表
        """
        # 如果是GeoDataFrame，取geometry列
        if hasattr(points_gdf, 'geometry'):
            points = points_gdf.geometry
        else:
            points = points_gdf
        
        # 至少3个点才能构成凸包
        if len(points) < 3:
            return list(points)
        
        # 使用shapely的MultiPoint和convex_hull
        from shapely.geometry import MultiPoint
        multipoint = MultiPoint([pt for pt in points])
        hull = multipoint.convex_hull
        # hull为Polygon，取其边界上的点
        if hull.geom_type == 'Polygon':
            hull_coords = list(hull.exterior.coords)
        else:
            hull_coords = list(hull.coords)
        return hull_coords
