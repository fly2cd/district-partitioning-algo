import geopandas as gpd


def read_poi_data(shp_file):
    gdf_ori = gpd.read_file(shp_file)
    poi_gdf = gdf_ori[gdf_ori['delivery_o']!=111]
    return poi_gdf

def read_block_data(shp_file, filtered_col, value):
    gdf_ori = gpd.read_file(shp_file)
    block_gdf = gdf_ori[gdf_ori[filtered_col]!=value]
    return block_gdf

