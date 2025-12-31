"""
Created on 2025.12.31
@author: chendi
该类主要用于对分割结果评估，实现了指标计算逻辑,主要指标如下：

输入：
    1、配送点位数据（geopanda），属性信息[gid, qty（配送量）, arrival_time（送达时间）, departure_time（出发时间）,
                                    transfer_time（交接时长=出发时间-送达时间）, block_id(对应block数据中的gid), poi_order(点位配送顺序), region_id（所属区域ID）]
    2、block数据（geopanda）, 属性信息[gid, qty(block内的poi配送量之和), arrival_time（进入block的最早时间）, departure_time（离开block的最晚时间）, block_order(配送顺序) region_id（所属区域ID）]

输出：各项指标值
        1.Number(区域划分数量)：
        2.Total_Distance(总距离):
        3.OTD_Rate(按时履约率)：每条线路3小时内送达件数/总件数
        4.Spatial_Compactness(空间紧凑性) ：每个区域内的总面积/轮廓线长度
        5.Workload_SD(各区域配送量标准差)
        6.Distance_SD(各区域配送距离标准差)
        7.Time_SD(各区域配送时间标准差)
        8.Delivery_Difficulty_SD(各区域件均距离标准差) 各区域的"总距离/总件数"的标准差
        # 9.Boundary_Fluctuation(边界波动率)
        # 10.Core_Retention_Rate(核心区域稳定率)
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from tool.Dis import calculateEuclideanDistance


class Metric:
    """
    分割结果评估指标计算类
    """
    
    def __init__(self, poi_gdf: gpd.GeoDataFrame, block_gdf: gpd.GeoDataFrame):
        """
        初始化评估类
        :param poi_gdf: 配送点位数据，必须包含列：gid, qty, arrival_time, departure_time, 
                        transfer_time, block_id, poi_order, region_id, geometry
        :param block_gdf: block数据，必须包含列：gid, qty, arrival_time, departure_time, 
                         block_order, region_id, geometry
        """
        self.poi_gdf = poi_gdf.copy()
        self.block_gdf = block_gdf.copy()
        
        # 验证必要的列是否存在
        required_poi_cols = ['gid', 'qty', 'arrival_time', 'departure_time', 
                            'transfer_time', 'block_id', 'poi_order', 'region_id']
        required_block_cols = ['gid', 'qty', 'arrival_time', 'departure_time', 
                              'block_order', 'region_id']
        
        missing_poi = [col for col in required_poi_cols if col not in poi_gdf.columns]
        missing_block = [col for col in required_block_cols if col not in block_gdf.columns]
        
        if missing_poi:
            raise ValueError(f"POI数据缺少必要的列: {missing_poi}")
        if missing_block:
            raise ValueError(f"Block数据缺少必要的列: {missing_block}")
    
    def calculate_number(self) -> int:
        """
        计算区域划分数量
        :return: 区域数量
        """
        unique_regions = self.poi_gdf['region_id'].dropna().unique()
        return len(unique_regions)
    
    def calculate_total_distance(self) -> float:
        """
        计算总距离：根据poi_order顺序计算每个区域内的路径总距离
        :return: 总距离（单位：与坐标系统一致）
        """
        total_distance = 0.0
        
        # 按区域分组计算
        for region_id in self.poi_gdf['region_id'].dropna().unique():
            region_poi = self.poi_gdf[self.poi_gdf['region_id'] == region_id].copy()
            
            # 按poi_order排序
            region_poi = region_poi.sort_values('poi_order')
            
            # 计算相邻点之间的距离
            for i in range(len(region_poi) - 1):
                point1 = region_poi.iloc[i].geometry
                point2 = region_poi.iloc[i + 1].geometry
                
                # 使用欧几里得距离
                if hasattr(point1, 'x') and hasattr(point1, 'y'):
                    dist = calculateEuclideanDistance(point1.x, point1.y, point2.x, point2.y)
                else:
                    # 如果geometry不是Point类型，尝试获取坐标
                    coords1 = list(point1.coords)[0]
                    coords2 = list(point2.coords)[0]
                    dist = calculateEuclideanDistance(coords1[0], coords1[1], coords2[0], coords2[1])
                
                total_distance += dist
        
        return total_distance
    
    def calculate_otd_rate(self, time_limit_hours: float = 3.0) -> float:
        """
        计算按时履约率：每条线路3小时内送达件数/总件数
        :param time_limit_hours: 时间限制（小时），默认3小时
        :return: 按时履约率（0-1之间）
        """
        # 计算每个区域的配送时间（从第一个点到最后一个点的总时间）
        on_time_count = 0
        total_count = 0
        
        for region_id in self.poi_gdf['region_id'].dropna().unique():
            region_poi = self.poi_gdf[self.poi_gdf['region_id'] == region_id].copy()
            
            if len(region_poi) == 0:
                continue
            
            # 按poi_order排序
            region_poi = region_poi.sort_values('poi_order')
            
            # 计算该区域的配送时间（从第一个点的出发时间到最后一个点的到达时间）
            first_departure = region_poi.iloc[0]['departure_time']
            last_arrival = region_poi.iloc[-1]['arrival_time']
            
            # 如果时间单位是小时，直接比较；如果是其他单位，需要转换
            # 假设时间单位是小时
            delivery_time = last_arrival - first_departure
            
            # 统计该区域内按时送达的件数
            region_on_time = region_poi[region_poi['arrival_time'] <= first_departure + time_limit_hours]
            
            on_time_count += len(region_on_time)
            total_count += len(region_poi)
        
        if total_count == 0:
            return 0.0
        
        return on_time_count / total_count
    
    def calculate_spatial_compactness(self) -> float:
        """
        计算空间紧凑性：每个区域内的总面积/轮廓线长度
        :return: 平均空间紧凑性
        """
        compactness_values = []
        
        for region_id in self.block_gdf['region_id'].dropna().unique():
            region_blocks = self.block_gdf[self.block_gdf['region_id'] == region_id]
            
            if len(region_blocks) == 0:
                continue
            
            # 合并该区域的所有block
            try:
                union_geom = region_blocks.geometry.unary_union
                
                # 计算总面积
                total_area = union_geom.area
                
                # 计算轮廓线长度（边界长度）
                if hasattr(union_geom, 'exterior'):
                    # Polygon类型
                    boundary_length = union_geom.exterior.length
                elif hasattr(union_geom, 'boundary'):
                    # 其他几何类型
                    boundary_length = union_geom.boundary.length
                else:
                    # 如果无法获取边界，跳过
                    continue
                
                if boundary_length > 0:
                    compactness = total_area / boundary_length
                    compactness_values.append(compactness)
            except Exception as e:
                # 如果几何操作失败，跳过该区域
                print(f"警告：区域 {region_id} 的几何计算失败: {e}")
                continue
        
        if len(compactness_values) == 0:
            return 0.0
        
        return np.mean(compactness_values)
    
    def calculate_workload_sd(self) -> float:
        """
        计算各区域配送量标准差
        :return: 配送量标准差
        """
        region_workloads = []
        
        for region_id in self.poi_gdf['region_id'].dropna().unique():
            region_poi = self.poi_gdf[self.poi_gdf['region_id'] == region_id]
            total_qty = region_poi['qty'].sum()
            region_workloads.append(total_qty)
        
        if len(region_workloads) == 0:
            return 0.0
        
        return np.std(region_workloads)
    
    def calculate_distance_sd(self) -> float:
        """
        计算各区域配送距离标准差
        :return: 配送距离标准差
        """
        region_distances = []
        
        for region_id in self.poi_gdf['region_id'].dropna().unique():
            region_poi = self.poi_gdf[self.poi_gdf['region_id'] == region_id].copy()
            region_poi = region_poi.sort_values('poi_order')
            
            region_distance = 0.0
            for i in range(len(region_poi) - 1):
                point1 = region_poi.iloc[i].geometry
                point2 = region_poi.iloc[i + 1].geometry
                
                if hasattr(point1, 'x') and hasattr(point1, 'y'):
                    dist = calculateEuclideanDistance(point1.x, point1.y, point2.x, point2.y)
                else:
                    coords1 = list(point1.coords)[0]
                    coords2 = list(point2.coords)[0]
                    dist = calculateEuclideanDistance(coords1[0], coords1[1], coords2[0], coords2[1])
                
                region_distance += dist
            
            region_distances.append(region_distance)
        
        if len(region_distances) == 0:
            return 0.0
        
        return np.std(region_distances)
    
    def calculate_time_sd(self) -> float:
        """
        计算各区域配送时间标准差
        :return: 配送时间标准差
        """
        region_times = []
        
        for region_id in self.poi_gdf['region_id'].dropna().unique():
            region_poi = self.poi_gdf[self.poi_gdf['region_id'] == region_id].copy()
            region_poi = region_poi.sort_values('poi_order')
            
            if len(region_poi) == 0:
                continue
            
            # 计算该区域的配送时间
            first_departure = region_poi.iloc[0]['departure_time']
            last_arrival = region_poi.iloc[-1]['arrival_time']
            delivery_time = last_arrival - first_departure
            
            region_times.append(delivery_time)
        
        if len(region_times) == 0:
            return 0.0
        
        return np.std(region_times)
    
    def calculate_delivery_difficulty_sd(self) -> float:
        """
        计算各区域件均距离标准差：各区域的"总距离/总件数"的标准差
        :return: 件均距离标准差
        """
        difficulty_values = []
        
        for region_id in self.poi_gdf['region_id'].dropna().unique():
            region_poi = self.poi_gdf[self.poi_gdf['region_id'] == region_id].copy()
            region_poi = region_poi.sort_values('poi_order')
            
            # 计算该区域的总距离
            region_distance = 0.0
            for i in range(len(region_poi) - 1):
                point1 = region_poi.iloc[i].geometry
                point2 = region_poi.iloc[i + 1].geometry
                
                if hasattr(point1, 'x') and hasattr(point1, 'y'):
                    dist = calculateEuclideanDistance(point1.x, point1.y, point2.x, point2.y)
                else:
                    coords1 = list(point1.coords)[0]
                    coords2 = list(point2.coords)[0]
                    dist = calculateEuclideanDistance(coords1[0], coords1[1], coords2[0], coords2[1])
                
                region_distance += dist
            
            # 计算总件数
            total_qty = region_poi['qty'].sum()
            
            # 计算件均距离
            if total_qty > 0:
                difficulty = region_distance / total_qty
                difficulty_values.append(difficulty)
        
        if len(difficulty_values) == 0:
            return 0.0
        
        return np.std(difficulty_values)
    
    def calculate_all_metrics(self) -> dict:
        """
        计算所有指标
        :return: 包含所有指标值的字典
        """
        metrics = {
            'Number': self.calculate_number(),
            'Total_Distance': self.calculate_total_distance(),
            'OTD_Rate': self.calculate_otd_rate(),
            'Spatial_Compactness': self.calculate_spatial_compactness(),
            'Workload_SD': self.calculate_workload_sd(),
            'Distance_SD': self.calculate_distance_sd(),
            'Time_SD': self.calculate_time_sd(),
            'Delivery_Difficulty_SD': self.calculate_delivery_difficulty_sd()
        }
        
        return metrics
    
    def print_metrics(self):
        """
        打印所有指标
        """
        metrics = self.calculate_all_metrics()
        print("=" * 50)
        print("分割结果评估指标")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        print("=" * 50)
        
        return metrics
