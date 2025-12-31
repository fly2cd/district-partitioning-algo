import geopandas as gdp
import pandas as pd
from shapely import Point

ENCODING_MAP = {
    '0':'A',
    '1':'B',
    '2':'C',
    '3':'D',
    '4':'H',
    '5':'I',
    '6':'J',
    '7':'K',
    '8':'L',
    '9':'M',
    '.':'X',
    '-':'Y',
    '+':'Z',
    'e':'E',
}

DECODING_MAP = {v:k for k, v in ENCODING_MAP.items()}

def number_encoder(value):
    ...


def number_decoder(txt):
    decoded = []
    for word in txt.split('_'):
        decoded.append(DECODING_MAP.get(word, word))
    return ''.join(decoded)


def decoder(txt_file, out_file, numeric_cols, sep='|'):
    df=pd.read_csv(txt_file, sep=sep)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(number_decoder)
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError as e:
                print(e)
        else:
            print(f"{col}列名不在数据中")

    df.to_csv(out_file)
    print(f'解码完成输出数据：{out_file}')

def write_shp(csv_file, shp_file):
    # 1. 读取CSV文件
    df = pd.read_csv(csv_file)

    # 2. 创建几何列（WGS84坐标）
    geometry = [Point(lon, lat) for lon, lat in zip(df["lng"], df["lat"])]

    # 3. 转换为GeoDataFrame并指定坐标系
    gdf = gdp.GeoDataFrame(
        df[["id", "qty", "departure", "confirm"]],  # 保留属性字段
        geometry=geometry,
        crs="EPSG:4326"  # WGS84坐标系
    )

    # 4. 输出为Shapefile
    gdf.to_file(shp_file, encoding="utf-8")

if __name__ == "__main__":
    input_file = '/Users/chendi/IdeaProjects/regionOpt/data/locations.csv'
    output_file = '/Users/chendi/IdeaProjects/regionOpt/data/locations.csv'
    shp_file = '/Users/chendi/IdeaProjects/regionOpt/data/shp/poi.shp'
    number_cols = ['lat', 'lng', 'qty']
    # decoder(input_file, output_file, number_cols)

    write_shp(output_file, shp_file)
