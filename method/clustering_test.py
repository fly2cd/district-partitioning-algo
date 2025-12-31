
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

from method.balance_kmeans import CBalanceKmeans
from method.bkmeans import BKMeans
from tool.file_io import read_block_data
from tool.visualization import KMeansVisualizer
from sklearn.cluster import OPTICS
import matplotlib.cm as cm
from hdbscan import HDBSCAN
from sklearn.neighbors import KernelDensity

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

class DistrictPartition:
    def __init__(self, poi_raw_gdf, block_raw_gdf, **kwargs):
        self.poi_raw_gdf = poi_raw_gdf
        self.block_raw_gdf = block_raw_gdf
        self.params = kwargs
        self.bkm_params = self.params.get("bkm_params")
        self.num_clusters = self.bkm_params.get("numClusters")

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
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # 1. 绘制分区（block），按 label 着色
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        cmap = ListedColormap(colors)
        self.block_raw_gdf.plot(column='label', cmap=cmap, edgecolor='gray', alpha=0.6, ax=ax, label='')
        print(f'处理后block总数为{len(self.block_raw_gdf)}')
        # 2. 叠加 POI 点
        if hasattr(self, 'poi_filted_gdf'):
            self.poi_filted_gdf.plot(ax=ax, color='black', markersize=10, alpha=0.7)
        # # 3. 叠加仓库点
        # if self.warehouse_poi is not None:
        #     ax.scatter(self.warehouse_poi.x, self.warehouse_poi.y, c='yellow', s=100, marker='*', edgecolors='black', label='Warehouse')
        ax.set_title('District Partition Result')
        ax.set_axis_off()
        ax.legend()
        safe_show(plt)

    def plot_boundary_blocks_and_lines(self, ax=None):
        """
        绘制每个cluster的边界block（高亮显示）和边界线。
        :param ax: 可选，matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np
        from shapely.geometry import MultiPolygon, LineString

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        cmap = ListedColormap(colors)
        # 1. 绘制所有block，按label着色
        self.block_raw_gdf.plot(column='label', cmap=cmap, edgecolor='gray', alpha=0.3, ax=ax)

        # 2. 找到所有cluster的边界block
        def is_boundary_block(idx, label):
            block_geom = self.block_raw_gdf.at[idx, 'geometry']
            # 只考虑边邻接（有公共边的block）
            neighbors = self.block_raw_gdf[self.block_raw_gdf.index != idx]
            # 用relate_pattern确保是边邻接
            edge_neighbors = neighbors[neighbors.geometry.relate_pattern(block_geom, 'F***1****')]
            for nidx, nrow in edge_neighbors.iterrows():
                if nrow['label'] != label:
                    return True
            return False

        cluster_blocks = {label: set(self.block_raw_gdf[self.block_raw_gdf['label'] == label].index)
                          for label in self.block_raw_gdf['label'].dropna().unique()}
        boundary_blocks = []
        for label, blocks in cluster_blocks.items():
            for idx in blocks:
                if is_boundary_block(idx, label):
                    boundary_blocks.append(idx)
        # 3. 高亮边界block
        boundary_gdf = self.block_raw_gdf.loc[boundary_blocks]
        boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, alpha=0.8, label='Boundary Block')

        # 4. 绘制每个cluster的边界线
        for label in cluster_blocks:
            cluster_gdf = self.block_raw_gdf[self.block_raw_gdf['label'] == label]
            # 合并为MultiPolygon
            mp = cluster_gdf.geometry.union_all()
            # 取外部边界
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

    def plot_swap_comparison(self, before_block_gdf=None):
        """
        绘制swap_block_unit调整前后的分区对比（两个子图，左为调整前，右为调整后）。
        :param before_block_gdf: 调整前的block_raw_gdf副本（GeoDataFrame）
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from shapely.geometry import LineString
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        cmap = ListedColormap(colors)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        # --- 左图：调整前 ---
        if before_block_gdf is not None:
            before_block_gdf.plot(column='label', cmap=cmap, edgecolor='gray', alpha=0.3, ax=ax1)
            # 绘制边界线
            for label in before_block_gdf['label'].dropna().unique():
                cluster_gdf = before_block_gdf[before_block_gdf['label'] == label]
                mp = cluster_gdf.geometry.union_all()
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
        self.block_raw_gdf.plot(column='label', cmap=cmap, edgecolor='gray', alpha=0.3, ax=ax2)
        for label in self.block_raw_gdf['label'].dropna().unique():
            cluster_gdf = self.block_raw_gdf[self.block_raw_gdf['label'] == label]
            mp = cluster_gdf.geometry.union_all()
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

class OpticsClutering:
    def __init__(self, poi_gdf, min_samples=5, xi=0.05, min_cluster_size=0.05):
        self.poi_gdf = poi_gdf.copy()
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.labels_ = None

    def clustering(self):
        """
        对poi_gdf进行OPTICS聚类分析，结果写入self.poi_gdf['optics_label']
        """
        coords = np.array([[geom.x, geom.y] for geom in self.poi_gdf.geometry])
        optics = OPTICS(min_samples=self.min_samples, xi=self.xi, min_cluster_size=self.min_cluster_size)
        optics.fit(coords)
        self.labels_ = optics.labels_
        self.poi_gdf['optics_label'] = self.labels_
        return self.poi_gdf

    def visual_result(self, ax=None):
        """
        可视化OPTICS聚类结果
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if 'optics_label' not in self.poi_gdf.columns:
            print('请先运行clustering方法')
            return
        labels = self.poi_gdf['optics_label']
        unique_labels = np.unique(labels)
        colors = cm.get_cmap('tab20', len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            ax.scatter(self.poi_gdf.geometry.x[mask], self.poi_gdf.geometry.y[mask],
                       s=10, color=colors(i), label=f'Cluster {label}' if label != -1 else 'Noise', alpha=0.7)
        ax.set_title('OPTICS Clustering Result')
        ax.set_axis_off()
        ax.legend()
        safe_show(plt)

class HdbscanClustering:
    def __init__(self, poi_gdf, min_samples=5, min_cluster_size=5):
        self.poi_gdf = poi_gdf.copy()
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.labels_ = None

    def clustering(self):
        """
        对poi_gdf进行HDBSCAN聚类分析，结果写入self.poi_gdf['hdbscan_label']
        """
        coords = np.array([[geom.x, geom.y] for geom in self.poi_gdf.geometry])
        hdb = HDBSCAN(min_samples=self.min_samples, min_cluster_size=self.min_cluster_size)
        hdb.fit(coords)
        self.labels_ = hdb.labels_
        self.poi_gdf['hdbscan_label'] = self.labels_
        return self.poi_gdf

    def visual_result(self, ax=None):
        """
        可视化HDBSCAN聚类结果
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if 'hdbscan_label' not in self.poi_gdf.columns:
            print('请先运行clustering方法')
            return
        labels = self.poi_gdf['hdbscan_label']
        unique_labels = np.unique(labels)
        colors = cm.get_cmap('tab20', len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            ax.scatter(self.poi_gdf.geometry.x[mask], self.poi_gdf.geometry.y[mask],
                       s=10, color=colors(i), label=f'Cluster {label}' if label != -1 else 'Noise', alpha=0.7)
        ax.set_title('HDBSCAN Clustering Result')
        ax.set_axis_off()
        ax.legend()
        safe_show(plt)

class KernelAnalysis:
    def __init__(self, poi_gdf, bandwidth=0.01, kernel='gaussian'):
        self.poi_gdf = poi_gdf.copy()
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = None
        self.density_ = None

    def fit(self):
        """
        对poi_gdf进行核密度估计，结果写入self.density_（每个点的密度值）
        """
        coords = np.array([[geom.x, geom.y] for geom in self.poi_gdf.geometry])
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(coords)
        log_density = self.kde.score_samples(coords)
        self.density_ = np.exp(log_density)
        self.poi_gdf['kde_density'] = self.density_
        return self.poi_gdf

    def extract_hotspots(self, quantile=0.95):
        """
        提取热点区域（密度高于指定分位数的点），并生成热点编号
        :param quantile: 分位数阈值（如0.95表示前5%高密度为热点）
        """
        if 'kde_density' not in self.poi_gdf.columns:
            raise ValueError("请先运行fit方法")
        threshold = np.quantile(self.poi_gdf['kde_density'], quantile)
        self.poi_gdf['is_hotspot'] = self.poi_gdf['kde_density'] >= threshold
        # 给热点点分组编号（可用DBSCAN等空间聚类，也可直接编号）
        from sklearn.cluster import DBSCAN
        coords = np.array([[geom.x, geom.y] for geom in self.poi_gdf[self.poi_gdf['is_hotspot']].geometry])
        if len(coords) > 0:
            db = DBSCAN(eps=self.bandwidth*2, min_samples=3).fit(coords)
            self.poi_gdf.loc[self.poi_gdf['is_hotspot'], 'hotspot_id'] = db.labels_
        else:
            self.poi_gdf['hotspot_id'] = -1
        return self.poi_gdf

    def visual_result(self, ax=None, cmap='hot', show_hotspot=True):
        """
        可视化核密度分析结果（点密度热力图）
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if 'kde_density' not in self.poi_gdf.columns:
            print('请先运行fit方法')
            return
        sc = ax.scatter(self.poi_gdf.geometry.x, self.poi_gdf.geometry.y,
                        c=self.poi_gdf['kde_density'], cmap=cmap, s=15, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='Density')
        if show_hotspot and 'is_hotspot' in self.poi_gdf.columns:
            # 高亮热点
            hs = self.poi_gdf[self.poi_gdf['is_hotspot']]
            ax.scatter(hs.geometry.x, hs.geometry.y, c='cyan', s=30, edgecolor='k', label='Hotspot')
            # 标注热点编号
            if 'hotspot_id' in hs.columns:
                for hid in hs['hotspot_id'].unique():
                    if hid == -1: continue
                    sub = hs[hs['hotspot_id'] == hid]
                    # 标注热点中心
                    x, y = sub.geometry.x.mean(), sub.geometry.y.mean()
                    ax.text(x, y, f'Hotspot {int(hid)}', color='blue', fontsize=12, weight='bold')
        ax.set_title('Kernel Density Estimation (KDE) with Hotspots')
        ax.set_axis_off()
        ax.legend()
        safe_show(plt)

def execute_kernel(poi_raw_file, arg_dict):
    poi_raw_gpd = gpd.read_file(poi_raw_file)
    ka = KernelAnalysis(poi_raw_gpd, **arg_dict)
    ka.fit()
    ka.extract_hotspots(quantile=0.8)
    ka.visual_result()

def execute(poi_raw_file, block_raw_file, arg_dict):
    # step1. 数据预处理, 建立poi和block映射关系，剔除非配送点
    poi_raw_gpd = gpd.read_file(poi_raw_file)
    block_raw_gpd = read_block_data(block_raw_file, -1)
    print(f'block总数为{len(block_raw_gpd)}')
    dp = DistrictPartition(poi_raw_gpd, block_raw_gpd, **arg_dict)
    dp.solve()
    dp.visualize()

def execute_optics(poi_raw_file, arg_dict):
    poi_raw_gpd = gpd.read_file(poi_raw_file)
    oc = OpticsClutering(poi_raw_gpd, **arg_dict)
    oc.clustering()
    oc.visual_result()

def execute_hdbscan(poi_raw_file, arg_dict):
    poi_raw_gpd = gpd.read_file(poi_raw_file)
    hc = HdbscanClustering(poi_raw_gpd, **arg_dict)
    hc.clustering()
    hc.visual_result()





if __name__ == '__main__':
    poi_raw_file = '../data/shp/poi_mct.shp'
    # block_raw_file = 'data/block_raw.shp'
    optics_arg_dict = {
        'min_samples': 5,
        'xi': 0.05,
        'min_cluster_size': 0.1
    }

    hdbscan_arg_dict = {
        'min_samples': 20,
        'min_cluster_size': 5
    }

    kernel_arg_dict = {
        'bandwidth': 200,
        'kernel': 'gaussian'
    }


    # execute_optics(poi_raw_file, optics_arg_dict)
    execute_hdbscan(poi_raw_file, hdbscan_arg_dict)
    # execute_kernel(poi_raw_file, kernel_arg_dict)