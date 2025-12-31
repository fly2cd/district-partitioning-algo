
import geopandas as gpd
import matplotlib
# 设置兼容的后端，避免 PyCharm 后端兼容性问题
try:
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from shapely.geometry import LineString


def safe_show(plt, save_path=None):
    """
    安全地显示或保存 matplotlib 图表，避免 PyCharm 后端兼容性问题
    :param plt: matplotlib.pyplot 模块
    :param save_path: 可选，如果提供则保存图片而不是显示
    """
    try:
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'图片已保存到: {save_path}')
        else:
            plt.show()
    except (AttributeError, RuntimeError) as e:
        # 如果显示失败，尝试保存到临时文件
        import os
        import tempfile
        if save_path is None:
            save_path = os.path.join(tempfile.gettempdir(), 'matplotlib_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'显示失败，图片已保存到: {save_path}')
        print(f'错误信息: {e}')


def routes_plot(routes, ax):
    """
    绘制多条路线
    :param routes: 路线列表，每条路线是Point对象列表
    :param ax: matplotlib轴对象
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for idx, route in enumerate(routes):
        x = []
        y = []
        for point in route:
            x.append(point.x)
            y.append(point.y)
        ax.scatter(x, y)
        ax.plot(x, y, color=colors[idx % len(colors)])


def visual_tsp_result(block_geodata, routes, ax, label=None):
    """
    可视化TSP结果
    :param block_geodata: block地理数据（GeoDataFrame）
    :param routes: 路线列表
    :param ax: matplotlib轴对象
    :param label: 用于着色的列名
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    cmap = ListedColormap(colors)
    
    if block_geodata is not None:
        block_geodata.plot(ax=ax, column=label, cmap=cmap, edgecolor='gray', alpha=0.6)
    
    for idx, route in enumerate(routes):
        x = []
        y = []
        for point in route:
            x.append(point.x)
            y.append(point.y)
        
        ax.scatter(x, y, s=5)
        ax.plot(x, y, color=colors[idx % len(colors)])
    
    ax.set_axis_off()


def plot_single_solution(block_gdf, routes):
    """
    绘制单个解决方案
    :param block_gdf: block地理数据
    :param routes: 路线列表
    """
    fig, ax = plt.subplots(1, 1)
    visual_tsp_result(block_gdf, routes, ax)
    safe_show(plt)


def plot_depot_effect(routes_b, routes_c, block_gdf):
    """
    绘制仓库位置效果对比
    :param routes_b: 边界仓库的路线
    :param routes_c: 中心仓库的路线
    :param block_gdf: block地理数据
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    visual_tsp_result(block_gdf, routes_b, ax1)
    visual_tsp_result(block_gdf, routes_c, ax2)
    safe_show(plt)


def plot_district_partition(block_gdf, poi_gdf=None, label_col='label', title='District Partition Result'):
    """
    绘制分区结果
    :param block_gdf: block地理数据，必须包含label_col列
    :param poi_gdf: 可选的POI点数据
    :param label_col: 用于着色的列名
    :param title: 图表标题
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    cmap = ListedColormap(colors)
    
    # 绘制分区（block），按 label 着色
    block_gdf.plot(column=label_col, cmap=cmap, edgecolor='gray', alpha=0.6, ax=ax, label='')
    print(f'处理后block总数为{len(block_gdf)}')
    
    # 叠加 POI 点
    if poi_gdf is not None:
        poi_gdf.plot(ax=ax, color='black', markersize=10, alpha=0.7)
    
    ax.set_title(title)
    ax.set_axis_off()
    ax.legend()
    safe_show(plt)


def plot_boundary_blocks_and_lines(block_gdf, label_col='label'):
    """
    绘制每个cluster的边界block（高亮显示）和边界线
    :param block_gdf: block地理数据
    :param label_col: 标签列名
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    cmap = ListedColormap(colors)
    
    # 1. 绘制所有block，按label着色
    block_gdf.plot(column=label_col, cmap=cmap, edgecolor='gray', alpha=0.3, ax=ax)
    
    # 2. 找到所有cluster的边界block
    def is_boundary_block(idx, label):
        block_geom = block_gdf.at[idx, 'geometry']
        neighbors = block_gdf[block_gdf.index != idx]
        edge_neighbors = neighbors[neighbors.geometry.relate_pattern(block_geom, 'F***1****')]
        for nidx, nrow in edge_neighbors.iterrows():
            if nrow[label_col] != label:
                return True
        return False
    
    cluster_blocks = {label: set(block_gdf[block_gdf[label_col] == label].index)
                      for label in block_gdf[label_col].dropna().unique()}
    boundary_blocks = []
    for label, blocks in cluster_blocks.items():
        for idx in blocks:
            if is_boundary_block(idx, label):
                boundary_blocks.append(idx)
    
    # 3. 高亮边界block
    boundary_gdf = block_gdf.loc[boundary_blocks]
    boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, alpha=0.8, label='Boundary Block')
    
    # 4. 绘制每个cluster的边界线
    for label in cluster_blocks:
        cluster_gdf = block_gdf[block_gdf[label_col] == label]
        mp = cluster_gdf.geometry.unary_union
        if mp.geom_type == 'Polygon':
            lines = [LineString(mp.exterior.coords)]
        elif mp.geom_type == 'MultiPolygon':
            lines = [LineString(poly.exterior.coords) for poly in mp.geoms]
        else:
            continue
        for line in lines:
            ax.plot(*line.xy, color='k', linewidth=2, alpha=0.7)
    
    ax.set_title('Boundary Blocks and Cluster Boundaries')
    ax.set_axis_off()
    ax.legend()
    safe_show(plt)


def plot_swap_comparison(before_block_gdf, after_block_gdf, label_col='label'):
    """
    绘制swap_block_unit调整前后的分区对比（两个子图，左为调整前，右为调整后）
    :param before_block_gdf: 调整前的block数据
    :param after_block_gdf: 调整后的block数据
    :param label_col: 标签列名
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    cmap = ListedColormap(colors)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # --- 左图：调整前 ---
    if before_block_gdf is not None:
        before_block_gdf.plot(column=label_col, cmap=cmap, edgecolor='gray', alpha=0.3, ax=ax1)
        # 绘制边界线
        for label in before_block_gdf[label_col].dropna().unique():
            cluster_gdf = before_block_gdf[before_block_gdf[label_col] == label]
            mp = cluster_gdf.geometry.unary_union
            if mp.geom_type == 'Polygon':
                lines = [LineString(mp.exterior.coords)]
            elif mp.geom_type == 'MultiPolygon':
                lines = [LineString(poly.exterior.coords) for poly in mp.geoms]
            else:
                continue
            for line in lines:
                ax1.plot(*line.xy, color='k', linewidth=2, alpha=0.7)
        ax1.set_title('Before swap_block_unit')
        ax1.set_axis_off()
    
    # --- 右图：调整后 ---
    after_block_gdf.plot(column=label_col, cmap=cmap, edgecolor='gray', alpha=0.3, ax=ax2)
    for label in after_block_gdf[label_col].dropna().unique():
        cluster_gdf = after_block_gdf[after_block_gdf[label_col] == label]
        mp = cluster_gdf.geometry.unary_union
        if mp.geom_type == 'Polygon':
            lines = [LineString(mp.exterior.coords)]
        elif mp.geom_type == 'MultiPolygon':
            lines = [LineString(poly.exterior.coords) for poly in mp.geoms]
        else:
            continue
        for line in lines:
            ax2.plot(*line.xy, color='k', linewidth=2, alpha=0.7)
    ax2.set_title('After swap_block_unit')
    ax2.set_axis_off()
    plt.tight_layout()
    safe_show(plt)


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
        plt.savefig(f'../method/convexHulls/{nameDataSet}_ConvexHull.png',
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
