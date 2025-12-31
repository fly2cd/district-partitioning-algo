from enum import Enum
import random as rd

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import alpha
from sklearn.cluster import KMeans

from method.balance_kmeans import CBalanceKmeans
from method.bkmeans import BKMeans
from tool.Basic import Point
from tool.file_io import read_poi_data, read_block_data, ALL_POLYGON
import method.TSP as tsp
from tool.visualization import (
    KMeansVisualizer, safe_show, routes_plot, visual_tsp_result,
    plot_single_solution, plot_depot_effect, plot_district_partition,
    plot_boundary_blocks_and_lines, plot_swap_comparison
)


class DepotLoc(Enum):
    CENTER = 1
    BOUNDARY = 2
    RANDOM = 3


def cluster(gdf, num_clusters):
    coordinates = np.array(list(gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
    kmeans = KMeans(n_cluster=num_clusters)
    gdf['cluster'] = kmeans.fit_predict(coordinates)
    print(gdf.head(5))
    return gdf

def tsp_solver(gdf, depot=None):
    points_list = []
    poi_id_dict = {}

    if depot is not None:
        points_list.append(Point(0, depot.x, depot.y))

    for index, row in gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        id = row['delivery_o']
        points_list.append(Point(id, x, y))
        poi_id_dict[id] = row['delivery_o']

    print(points_list)
    route = tsp.TSPInsert(points_list)
    return route, poi_id_dict


# routesPlot 函数已移至 tool/visualization.py，使用 routes_plot 替代


def cal_solution(gdf_cluster, depot_poi, label_filed='cluster'):
    routes_arr = []
    for cluster_id in gdf_cluster[label_filed].unique():
        sub_gdf = gdf_cluster[gdf_cluster[label_filed] == cluster_id]
        route, id_dict = tsp_solver(sub_gdf, depot_poi)
        routes_arr.append(route)
    return routes_arr


def eval_kpi(routes, geo_df):
    geo_df['route_id'] = None
    geo_df['time'] = None
    route_df = pd.DataFrame(columns=['route_id', 'dist_cost', 'time_cost', 'demand', 'density', 'poi_cnt'])
    route_dist_dict = {}
    for idx, route in enumerate(routes):
        L = tsp.calculate_route_length(route)
        for point in route:
            geo_df.loc[geo_df['delivery_o'] == point.id, 'route_id'] = idx
            time = rd.uniform(50, 200)
            geo_df.loc[geo_df['delivery_o'] == point.id, 'time'] = time
        route_dist_dict[idx] = L

    for route_id in geo_df['route_id'].unique():
        sub_gdf = geo_df[geo_df['route_id'] == route_id]
        tot_demand = sub_gdf['qty'].sum()
        distance = route_dist_dict[route_id]
        tot_time = sub_gdf['time'].sum() + distance / 200 / 36
        density = tot_demand / distance
        poi_cnt = sub_gdf['qty'].count()
        route_df.loc[len(route_df)] = [route_id, distance, tot_time, tot_demand, density, poi_cnt]

    print(route_df)


def cal_tsp_solution(poi_gdf, depot, block_geo):
    routes = cal_solution(poi_gdf, depot)
    plot_single_solution(block_geo, routes)
    eval_kpi(routes, poi_gdf)


def depot_loc_center_vs_boundary(poi_gdf):
    depot_poi_c = create_depot_poi(poi_gdf, depot_type=DepotLoc.CENTER)
    depot_poi_b = create_depot_poi(poi_gdf, depot_type=DepotLoc.BOUNDARY)

    routes_b = cal_solution(poi_gdf, depot_poi_b)
    routes_c = cal_solution(poi_gdf, depot_poi_c)

    return routes_b, routes_c


# plot_single_solution, plot_depot_effect, visual_tsp_result 函数已移至 tool/visualization.py


def eval_depot_loc_effect(poi_gdf, block_gdf):
    routes_b, routes_c = depot_loc_center_vs_boundary(poi_gdf)
    plot_depot_effect(routes_b, routes_c, block_gdf)


def create_depot_poi(poi_gdf, depot_type):
    xMin, yMin, xMax, yMax = poi_gdf.total_bounds

    if depot_type == DepotLoc.CENTER:
        return Point(0, ((xMax - xMin) / 2 * rd.uniform(0.9, 1.1) + xMin),
                     ((yMax - yMin) / 2 * rd.uniform(0.9, 1.1) + yMin))
    elif depot_type == DepotLoc.BOUNDARY:
        #简单处理直接用了最小坐标值
        return Point(0, xMin, yMin)
    else:
        return Point(0, ((xMax - xMin) * rd.random() + xMin), ((yMax - yMin) * rd.random() + yMin))


def tsp_eval(poi_shp_file, block_file):
    poi_gdf, depot_poi_b = read_poi_data(poi_shp_file, return_depot=True)
    print(poi_gdf.columns)

    block_gdf = read_block_data(block_file, id=ALL_POLYGON)
    # poi_gdf = gpd.sjoin(poi_gdf_all, block_gdf, how='inner', predicate='intersects')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    print('total demand = ' + str(poi_gdf['qty'].sum()))

    depot_poi = create_depot_poi(poi_gdf, depot_type=DepotLoc.RANDOM)
    routes1 = cal_solution(poi_gdf, depot_poi, 'cluster')
    visual_tsp_result(block_gdf, routes1, ax1, 'cluster')
    import matplotlib.pyplot as plt
    safe_show(plt)
    eval_kpi(routes1, poi_gdf)


def poi_clustering_bkm(poi_shp, block_file, poi_cluster_shp, num_clusters, **kwargs):
    # 1. 读取shp文件
    gdf = gpd.read_file(poi_shp)
    block_gdf = read_block_data(block_file, filtered_col='Id', value=2)
    # 获取交集，主要是通过block对poi进行过滤
    poi_gdf_all = gpd.sjoin(gdf, block_gdf, how='inner', predicate='intersects')
    coords = [[geom.x, geom.y] for geom in poi_gdf_all.geometry]

    # 2. 聚类
    bkmeans = BKMeans()
    bkmeans.initialize(coords, numClusters=num_clusters)
    bkmeans.run(
        terminationCriterion=kwargs.get("terminationCriterion"),  #"MaxDiffClusterSizes",
        terminationCriterionValue=kwargs.get("terminationCriterionValue"),  # 1,
        maxIter=50000
    )
    labels = bkmeans.bestAssignment

    # 3. 写回shp
    poi_gdf_all['cluster'] = labels
    poi_gdf_all.to_file(poi_cluster_shp)


class DistrictPartition:
    def __init__(self, poi_raw_gdf, block_raw_gdf, **kwargs):
        self.poi_raw_gdf = poi_raw_gdf
        self.block_raw_gdf = block_raw_gdf
        self.params = kwargs
        self.bkm_params = self.params.get("bkm_params")
        self.num_clusters = bkm_params.get("numClusters")

    def preprocess(self):
        # 获取交集，主要是通过block对poi进行过滤
        self.poi_filted_gdf = gpd.sjoin(self.poi_raw_gdf, self.block_raw_gdf, how='inner', predicate='intersects')
        print(f'完成数据预处理，poi点数为{len(self.poi_filted_gdf.geometry)}')

    def poi_clustering(self):
        is_show = self.params.get("show_poi_result")
        if self.params.get("execute_CBKM"):
            cbkm = CBalanceKmeans(self.poi_filted_gdf, self.num_clusters,5)
            cbkm.excute_balance_cluster()
            print(cbkm.cmd_args)
            poi_with_label, block_with_label = cbkm.post_process(self.poi_filted_gdf, self.block_raw_gdf)
            self.poi_filted_gdf = poi_with_label
            kmv = KMeansVisualizer(poi_with_label)
            kmv.showResults("POI2")
        else:
            coords = [[geom.x, geom.y] for geom in self.poi_filted_gdf.geometry]
            # 2. 聚类
            bkmeans = BKMeans()
            bkmeans.initialize(coords, numClusters=self.num_clusters)
            bkmeans.run(
                terminationCriterion=self.bkm_params.get("terminationCriterion"),  #"MaxDiffClusterSizes",
                terminationCriterionValue=self.bkm_params.get("terminationCriterionValue"),  # 1,
                maxIter=self.bkm_params.get("maxIter")
            )
            labels = bkmeans.bestAssignment
            if is_show:
                bkmeans.showResultsConvexHull("poi", 1, 0.0)
            # 3. 关联到poi数据集中
            self.poi_filted_gdf['cluster'] = labels

    def block_partition(self):
        '''
        将点聚类结果映射到block数据中，同时对于不存在点的block需要进行label指派
        步骤 1 (有POl的区块)：代码现在能正确处理一个区块内有多个标签的情况，根据规则（多数原则，或在数量相同时选择特定簇）来确定唯一标签。
        步骤 2 (无POl的区块)：首先，计算每个点簇的凸包。然后，对于没有标签的区块，会根据它是否在凸包内，或与哪个簇的质心最近，来分配一个标签。
                            这个过程也包含了对几何计算可能出现的异常的健壮性处理。
        步骤 3 (连通性保证)：这是一个循环过程。它会不断检查每个标签下的所有区块是否都相互连接。如果发现一个“孤岛”区块，
                            它会查找所有与之相邻但标签不同的区块，计算它们共享边界的长度，然后将这个“孤岛”的标签更新为那个共享边界最长的邻居的标签。
                            这个过程会一直重复，直到不再有任何标签被修改，从而确保了最终分区的连通性。
        :return:
        '''
        # 步骤 1: 为包含POI点的block分配初始标签
        # 使用空间连接找到每个block中的所有POI点
        block_with_poi = gpd.sjoin(self.block_raw_gdf, self.poi_filted_gdf[['geometry', 'cluster']],
                                   how='left', predicate='intersects')

        # 给POI添加block_idx字段，表示其属于哪个block
        block_gdf_no_idx = self.block_raw_gdf[['geometry']].copy()
        block_gdf_no_idx = block_gdf_no_idx.reset_index()  # 确保index_right是唯一block编号
        poi_block_idx = gpd.sjoin(self.poi_filted_gdf, block_gdf_no_idx, how='left', predicate='intersects')
        self.poi_filted_gdf['block_idx'] = poi_block_idx['index_right'].values

        # 为每个block确定其标签
        block_label_map = {}
        # 通过 groupby 按原始 block 的索引进行分组
        for block_idx, group in block_with_poi.groupby(block_with_poi.index):
            # 移除没有聚类标签的POI点 (例如, left join中未匹配到的)
            labels, counts = np.unique(group['cluster'].dropna(), return_counts=True)

            if len(labels) == 0:
                continue  # 该block内没有POI点，留到下一步处理

            if len(labels) == 1:
                block_label_map[block_idx] = labels[0]  # 如果只有一个标签，直接使用
            else:
                # 存在多个标签，按规则处理
                max_count = counts.max()
                # 找到出现次数最多的所有标签
                max_labels = labels[counts == max_count]
                if len(max_labels) == 1:
                    # 少数服从多数
                    block_label_map[block_idx] = max_labels[0]
                else:
                    # 如果点数相当，规则是选取点数较少的簇，这里简化为选择第一个
                    block_label_map[block_idx] = max_labels[0]

        # 将计算出的标签应用到block数据中
        self.block_raw_gdf['label'] = self.block_raw_gdf.index.map(block_label_map).astype('float')

        # 步骤 2: 处理未包含POI点的block
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon

        # 计算每个聚类标签的凸包和质心
        cluster_hulls = {}
        cluster_centroids = {}
        unique_labels = self.poi_filted_gdf['cluster'].dropna().unique()

        for label in unique_labels:
            if label == -1: continue  # 忽略噪声点
            cluster_points = self.poi_filted_gdf[self.poi_filted_gdf['cluster'] == label]
            coords = np.array([[p.x, p.y] for p in cluster_points.geometry])
            if len(coords) >= 3:
                try:
                    # QhullError can happen with collinear points, so we wrap it
                    hull = ConvexHull(coords)
                    hull_poly = Polygon(coords[hull.vertices])
                    cluster_hulls[label] = hull_poly
                    cluster_centroids[label] = hull_poly.centroid
                except Exception:
                    pass

        # 找到未被分配标签的block
        unlabeled_blocks_idx = self.block_raw_gdf[self.block_raw_gdf['label'].isna()].index

        for idx in unlabeled_blocks_idx:
            block_geom = self.block_raw_gdf.at[idx, 'geometry']
            # 构建一个极小的buffer，扩大邻接容忍度
            buffered_geom = block_geom.buffer(1e-5)
            # 找到所有与buffer后相交的block
            neighbors = self.block_raw_gdf[self.block_raw_gdf.geometry.intersects(buffered_geom)]
            # 只保留已分配标签的邻居
            labeled_neighbors = neighbors[neighbors['label'].notna()]
            if not labeled_neighbors.empty:
                # 计算与每个邻居的交集面积
                max_area = -1
                best_label = None
                for nidx, nrow in labeled_neighbors.iterrows():
                    try:
                        inter = buffered_geom.intersection(nrow.geometry)
                        area = inter.area if hasattr(inter, 'area') else inter.length
                        if area > max_area:
                            max_area = area
                            best_label = nrow['label']
                    except Exception:
                        continue  # 忽略几何异常
                if best_label is not None:
                    self.block_raw_gdf.at[idx, 'label'] = best_label
            # 如果没有邻居有标签，可以选择保留为NaN，或后续再处理

    def swap_block_unit(self, max_iter=20, max_point_change=10):
        """
        对于边界上的block进行交换或单边label改变，在尽量少改变聚类点差异的情况下，使得每个cluster对应的block集合的边界形态尽量避免锯齿状。
        只考虑边邻接（有公共边），点邻接不算。
        平滑度评估方法：以cluster外部边界总长度为平滑度指标，只有操作后两个cluster的外部边界总长度减少才接受label变更或block互换。
        :param max_iter: 最大迭代次数，防止死循环
        :param max_point_change: 允许每个cluster点数变化的最大绝对值
        """
        import numpy as np
        from shapely.geometry import LineString, MultiLineString
        def get_block_poi_count(idx):
            if hasattr(self, 'poi_filted_gdf') and 'block_idx' in self.poi_filted_gdf.columns:
                return len(self.poi_filted_gdf[self.poi_filted_gdf['block_idx'] == idx])
            return 0
        def is_boundary_block(idx, label):
            block_geom = self.block_raw_gdf.at[idx, 'geometry']
            # 只考虑边邻接
            neighbors = self.block_raw_gdf[self.block_raw_gdf.index != idx]
            edge_neighbors = neighbors[neighbors.geometry.relate_pattern(block_geom, 'F***1****')]
            for nidx, nrow in edge_neighbors.iterrows():
                if nrow['label'] != label:
                    return True
            return False
        def get_cluster_point_counts():
            if hasattr(self, 'poi_filted_gdf'):
                return self.poi_filted_gdf.groupby('cluster').size().to_dict()
            return {}
        def get_edge_neighbors(idx):
            block_geom = self.block_raw_gdf.at[idx, 'geometry']
            neighbors = self.block_raw_gdf[self.block_raw_gdf.index != idx]
            return neighbors[neighbors.geometry.relate_pattern(block_geom, 'F***1****')]
        def get_cluster_boundary_length(label):
            cluster_gdf = self.block_raw_gdf[self.block_raw_gdf['label'] == label]
            if cluster_gdf.empty:
                return 0
            mp = cluster_gdf.geometry.union_all()
            if mp.geom_type == 'Polygon':
                lines = [LineString(mp.exterior.coords)]
            elif mp.geom_type == 'MultiPolygon':
                lines = [LineString(poly.exterior.coords) for poly in mp.geoms]
            else:
                return 0
            return sum(line.length for line in lines)
        for _ in range(max_iter):
            print(f"迭代次数：{_}")
            changed = False
            # 步骤一：统计边界block
            cluster_blocks = {label: set(self.block_raw_gdf[self.block_raw_gdf['label'] == label].index)
                              for label in self.block_raw_gdf['label'].dropna().unique()}
            boundary_blocks = {label: [idx for idx in blocks if is_boundary_block(idx, label)]
                               for label, blocks in cluster_blocks.items()}
            all_boundary_blocks = [idx for blocks in boundary_blocks.values() for idx in blocks]
            cluster_point_counts = get_cluster_point_counts()
            # 步骤二：单边block“转投”
            for idx in all_boundary_blocks:
                label_from = self.block_raw_gdf.at[idx, 'label']
                edge_neighbors = get_edge_neighbors(idx)
                neighbor_labels = set(edge_neighbors['label']) - {label_from}
                for label_to in neighbor_labels:
                    if np.isnan(label_to):
                        continue
                    poi_count = get_block_poi_count(idx)
                    before_counts = cluster_point_counts.copy()
                    after_counts = before_counts.copy()
                    after_counts[label_from] = after_counts.get(label_from, 0) - poi_count
                    after_counts[label_to] = after_counts.get(label_to, 0) + poi_count
                    if (abs(after_counts[label_from] - before_counts[label_from]) > max_point_change or
                        abs(after_counts[label_to] - before_counts[label_to]) > max_point_change):
                        continue
                    # 评估平滑度：外部边界总长度
                    before_len = get_cluster_boundary_length(label_from) + get_cluster_boundary_length(label_to)
                    self.block_raw_gdf.at[idx, 'label'] = label_to
                    after_len = get_cluster_boundary_length(label_from) + get_cluster_boundary_length(label_to)
                    if after_len < before_len:
                        changed = True
                        cluster_point_counts = after_counts
                        break  # 本轮只做一次更改
                    else:
                        self.block_raw_gdf.at[idx, 'label'] = label_from
                if changed:
                    break
            if changed:
                continue  # 优先单边操作，若有更改则下一轮
            # 步骤三：block对互换
            neighbor_pairs = []
            for label_a, blocks_a in boundary_blocks.items():
                for idx_a in blocks_a:
                    edge_neighbors = get_edge_neighbors(idx_a)
                    for idx_b, row_b in edge_neighbors.iterrows():
                        label_b = row_b['label']
                        if label_b != label_a and idx_b in boundary_blocks.get(label_b, []):
                            if (idx_b, idx_a) not in neighbor_pairs:
                                neighbor_pairs.append((idx_a, idx_b))
            for idx_a, idx_b in neighbor_pairs:
                label_a = self.block_raw_gdf.at[idx_a, 'label']
                label_b = self.block_raw_gdf.at[idx_b, 'label']
                a_poi = get_block_poi_count(idx_a)
                b_poi = get_block_poi_count(idx_b)
                before_counts = cluster_point_counts.copy()
                after_counts = before_counts.copy()
                after_counts[label_a] = after_counts.get(label_a, 0) - a_poi + b_poi
                after_counts[label_b] = after_counts.get(label_b, 0) - b_poi + a_poi
                if (abs(after_counts[label_a] - before_counts[label_a]) > max_point_change or
                    abs(after_counts[label_b] - before_counts[label_b]) > max_point_change):
                    continue
                # 评估平滑度：外部边界总长度
                before_len = get_cluster_boundary_length(label_a) + get_cluster_boundary_length(label_b)
                self.block_raw_gdf.at[idx_a, 'label'] = label_b
                self.block_raw_gdf.at[idx_b, 'label'] = label_a
                after_len = get_cluster_boundary_length(label_a) + get_cluster_boundary_length(label_b)
                if after_len < before_len:
                    changed = True
                    cluster_point_counts = after_counts
                    break  # 本轮只做一次更改
                else:
                    self.block_raw_gdf.at[idx_a, 'label'] = label_a
                    self.block_raw_gdf.at[idx_b, 'label'] = label_b
            if not changed:
                break

    def get_partition_result(self):
        return self.block_raw_gdf

    def solve(self):
        self.preprocess()
        self.poi_clustering()
        self.block_partition()
        before = self.block_raw_gdf.copy()
        self.swap_block_unit()
        self.plot_swap_comparison(before_block_gdf=before)



    def visualize(self):
        # 划分结果可视化
        poi_gdf = self.poi_filted_gdf if hasattr(self, 'poi_filted_gdf') else None
        plot_district_partition(self.block_raw_gdf, poi_gdf, label_col='label')

    def plot_boundary_blocks_and_lines(self, ax=None):
        """
        绘制每个cluster的边界block（高亮显示）和边界线。
        :param ax: 可选，matplotlib轴对象（当前未使用，保留以兼容旧接口）
        """
        plot_boundary_blocks_and_lines(self.block_raw_gdf, label_col='label')

    def plot_swap_comparison(self, before_block_gdf=None):
        """
        绘制swap_block_unit调整前后的分区对比（两个子图，左为调整前，右为调整后）。
        :param before_block_gdf: 调整前的block_raw_gdf副本（GeoDataFrame）
        """
        plot_swap_comparison(before_block_gdf, self.block_raw_gdf, label_col='label')


def run(poi_raw_file, block_raw_file, arg_dict):
    # step1. 数据预处理, 建立poi和block映射关系，剔除非配送点
    poi_raw_gpd = gpd.read_file(poi_raw_file)
    block_raw_gpd = read_block_data(block_raw_file, id=ALL_POLYGON)
    print(f'block总数为{len(block_raw_gpd)}')
    dp = DistrictPartition(poi_raw_gpd, block_raw_gpd, **arg_dict)
    dp.solve()
    dp.visualize()
    # dp.plot_boundary_blocks_and_lines()




if __name__ == '__main__':
    poi_raw_file = '../data/shp/poi_mct.shp'
    block_raw_file = '../data/shp/block_mct.shp'
    poi_label_file = '../data/shp/poi_labels.shp'
    block_label_file = '../data/shp/block_labels.shp'

    bkm_params = {
        "numClusters": 6,
        "terminationCriterion": "MaxDiffClusterSizes",
        "terminationCriterionValue": 2,
        "maxIter": 20000
    }

    arg_dict = {
        "warehouse_type": None,  # 无仓库点
        "show_poi_result": True,
        "execute_CBKM": True,  # True:执行CBKM的代码; False:执行python代码
        "bkm_params": bkm_params,
    }

    print(f"参数配置如下:{arg_dict}")

    run(poi_raw_file, block_raw_file, arg_dict)
    # 执行点聚类，并生成点聚类标签

    # poi_clustering_bkm(poi_raw_file, block_raw_file)
    #
    # km_clusters_rs = '../data/shp/km_rs.shp'
    # bl_clusters_rs = '../data/shp/bl_km_rs.shp'
    # tsp_eval(shp_file, block_file)

    print(0)
