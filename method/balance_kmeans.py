import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd

from tool.file_io import read_block_data
from tool.file_io import read_poi_data
from tool.visualization import KMeansVisualizer

'''
# !!!!!非常重要!!!!! 不看无法执行
1.该脚本代码并不是本身实现了平衡聚类的算法逻辑，而是调用了外部写好的算法库，具体地址可参考：
https://github.com/uef-machine-learning/Balanced_k-Means_Revisited.git
2.执行该脚本前，需要在本地对源文件进行编译，cd到SoftBKmeans目录进行make
3.按要求准备参数，启动可执行文件，得到最终结果
'''
RELATIVE_PATH = r'D:\IdeaProjects\district-partitioning-algo\data'

# 以下为执行路径、过程数据和最终数据文件
excuteable_path = r'D:\Projects\Balanced_k-Means_Revisited-main\x64\Debug\SoftBKmeans.exe'
_data_path = RELATIVE_PATH+'/coordinates.txt'
_label_path = RELATIVE_PATH+'/labels.pa'
_centroid_path = RELATIVE_PATH+'/centroids.txt'

class CBalanceKmeans:
    def __init__(self, poi_gdf, num_cluster=2, max_diff_ratio=5):
        """
        调用C++版本的BalanceKMeans算法
        :param poi_gdf: poi点
        :param num_cluster: 聚类数
        :param max_diff_ratio: 类簇之间的差异比例，默认5pct
        """
        # 要传递给unixec的命令行参数列表['--potion1', 'value1', '--option2', 'value']
        # -o labels.pa -c centroid.txt --switch 30 --maxdiff 50

        coordinates = np.array(list(poi_gdf.geometry.apply(lambda geom:(geom.x, geom.y))))
        np.savetxt(_data_path, coordinates, fmt='%f', delimiter=' ', comments='')
        self.maxdiff = len(coordinates)*max_diff_ratio/(num_cluster*100.0)

        self.cmd_args=[]
        self.cmd_args.extend(['-k', str(num_cluster)])
        self.cmd_args.extend(['--seed', '2344357'])  # 2344357、 363538
        self.cmd_args.extend(['-i', _data_path])
        self.cmd_args.extend(['-o', _label_path])
        self.cmd_args.extend(['-c', _centroid_path])
        self.cmd_args.extend(['--switch', '30'])
        self.cmd_args.extend(['--maxdiff', str(self.maxdiff)])


    def excute_balance_cluster(self):
        '''
        调用执行程序，具体用法可参考代码库的README.rd可用一下命令行进行测试：
            ./bkmeans -k 15 --seed 7059488 -i /Users/chendi/IdeaProjects/regionOpt/data/coordinates.txt -o labels.pa -c centroids.txt --switch 30 --maxdiff 50
        :param cmd_agrgs: 执行参数
        :return:
        '''
        result = subprocess.run([excuteable_path] + self.cmd_args, capture_output=True, text=True)

        print('stdout:', result.stdout)
        print('stderr:', result.stderr)

        if result.returncode == 0:
            print('Execution succeeded')
        else:
            print('Executing failed with return code', result.returncode)


    def post_process(self, poi_gdf, block_gdf):
        '''
        将聚类结果关联到poi和block数据集上
        :param poi_gdf: 原始的poi数据集
        :param block_gdf: 原始的block数据集
        :return:
        '''
        # 平衡聚类的结果会存到一个label.o的文件中，该文件其实是csv格式，需要关联到poi的属性表中
        labels_series = pd.read_csv(_label_path, header=None)

        if len(labels_series) == len(poi_gdf):
            poi_gdf['cluster'] = labels_series.values
        else:
            print("Error: the number of rows in labels.txt dose not match the number of rows in poi_gdf.")

        # 为block进行类型标记，如果一个block包含多个cluster的poi则取qty总量最大的cluster作为block的聚类标签
        sum_qty = poi_gdf.groupby(['index_right','cluster'])['qty'].sum().reset_index()
        max_qty_indices = sum_qty.groupby('index_right')['qty'].idxmax()
        max_cluster = sum_qty.loc[max_qty_indices, ['index_right', 'cluster', 'qty']]
        #将标签关联到block数据：
        block_label = block_gdf.merge(max_cluster, left_index=True, right_on='index_right', how='left')

        # 计算每个cluster的点数
        cluster_counts = poi_gdf['cluster'].value_counts().sort_index()
        print("每个cluster的点数:")
        print(cluster_counts)

        # 计算点数偏差（最大-最小）
        count_deviation = cluster_counts.max() - cluster_counts.min()
        print(f"点数偏差（最大-最小）: {count_deviation}")

        return poi_gdf, block_label


if __name__ == '__main__':
    shp_file = RELATIVE_PATH+'/shp/poi_mct.shp'
    block_file = RELATIVE_PATH+'/shp/block_mct.shp'
    poi_result_file = RELATIVE_PATH+'/shp/poi_labels.shp'
    block_result_file = RELATIVE_PATH+'/shp/block_labels.shp'


    poi_gdf = read_poi_data(shp_file)
    block_gdf = read_block_data(block_file, 'Id', 2)

    # 获取交集，主要是通过block对poi进行过滤
    poi_gdf_all = gpd.sjoin(poi_gdf, block_gdf, how='inner', predicate='intersects')
    # 初始化将gdf数据转成算法包需要的输入数据和参数
    '''
    cmd_args.extend(['-k', str(num_cluster)])
    cmd_args.extend(['--seed', '7059488'])
    cmd_args.extend(['-i', _data_path])
    cmd_args.extend(['-o', _label_path])
    cmd_args.extend(['-c', _centroid_path])
    cmd_args.extend(['--switch', '30'])
    cmd_args.extend(['--maxdiff', str(maxdiff)])
    bkmeans.initialize(data, numClusters=3)
    '''
    cbkm = CBalanceKmeans(poi_gdf_all, 6,5)
    cbkm.excute_balance_cluster()
    print(cbkm.cmd_args)
    poi_with_label, block_with_label = cbkm.post_process(poi_gdf_all, block_gdf)
    kmv = KMeansVisualizer(poi_with_label)
    kmv.showResults("POI2")
    # 保存poi数据和block数据
    poi_with_label.to_file(poi_result_file, driver="ESRI Shapefile", encoding="utf-8")
    block_with_label.to_file(block_result_file, driver="ESRI Shapefile", encoding="utf-8")

    print(f"clustering result saved to '{poi_result_file}'")



