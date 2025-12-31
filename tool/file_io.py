import geopandas as gpd
from tool.Basic import Point

# 常量定义
ALL_POLYGON = -1
NORMAL_POLYGON = 1
UNNORMAL_POLYGON = 2


def read_poi_data(shp_file, return_depot=False):
    """
    读取POI数据
    :param shp_file: Shapefile文件路径
    :param return_depot: 是否返回仓库点，如果True则返回(poi_gdf, depot_poi)，否则只返回poi_gdf
    :return: poi_gdf 或 (poi_gdf, depot_poi)
    """
    gdf_ori = gpd.read_file(shp_file)
    
    if return_depot:
        # 提取仓库点（delivery_o == 111）
        depot_gdf = gdf_ori[gdf_ori['delivery_o'] == 111]
        depot_poi = None
        if not depot_gdf.empty:
            depot_poi = Point(0, depot_gdf.iloc[0].geometry.x, depot_gdf.iloc[0].geometry.y)
        
        # 提取POI点（delivery_o != 111）
        poi_gdf = gdf_ori[gdf_ori['delivery_o'] != 111]
        return poi_gdf, depot_poi
    else:
        # 只返回POI点
        poi_gdf = gdf_ori[gdf_ori['delivery_o'] != 111]
        return poi_gdf


def read_block_data(shp_file, id=None, filtered_col=None, value=None):
    """
    读取block数据
    :param shp_file: Shapefile文件路径
    :param id: block的ID值，如果提供则只返回该ID的block（兼容旧接口）
    :param filtered_col: 过滤列名（兼容旧接口）
    :param value: 过滤值（兼容旧接口）
    :return: block_gdf
    """
    gdf_ori = gpd.read_file(shp_file)
    
    # 优先使用新接口（id参数）
    if id is not None:
        if id == ALL_POLYGON:
            return gdf_ori
        else:
            # 尝试使用'Id'列
            if 'Id' in gdf_ori.columns:
                block_gdf = gdf_ori[gdf_ori['Id'] == id]
                return block_gdf
            else:
                # 如果没有'Id'列，返回全部
                return gdf_ori
    
    # 兼容旧接口（filtered_col和value参数）
    if filtered_col is not None and value is not None:
        block_gdf = gdf_ori[gdf_ori[filtered_col] != value]
        return block_gdf
    
    # 如果都没有提供，返回全部数据
    return gdf_ori

