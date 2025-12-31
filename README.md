# 区域划分算法 (District Partitioning Algorithm)

一个用于配送区域划分的Python算法库，支持POI点聚类、Block分区、TSP路径规划和结果评估。

## 📋 目录

- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [核心模块](#核心模块)
- [评估指标](#评估指标)
- [依赖说明](#依赖说明)
- [许可证](#许可证)

## ✨ 功能特性

- **多种聚类算法**：支持平衡K-means、标准K-means、HDBSCAN等聚类算法
- **智能分区**：基于POI点聚类结果进行Block分区，支持边界优化
- **路径规划**：集成TSP算法进行配送路径优化
- **结果评估**：提供8项评估指标，全面评估分区质量
- **可视化**：支持聚类结果、分区结果、路径规划的可视化展示
- **数据兼容**：支持Shapefile格式的地理数据输入输出

## 📁 项目结构

```
district-partitioning-algo/
├── data/                    # 数据目录
│   ├── shp/                # Shapefile数据文件
│   └── *.txt, *.csv        # 其他数据文件
├── entity/                  # 实体类
│   └── cluster.py          # Cluster, Point, Coordinate等实体类
├── method/                  # 算法实现
│   ├── bkmeans.py          # 平衡K-means算法（Python实现）
│   ├── balance_kmeans.py   # 平衡K-means算法（C++调用）
│   ├── TSP.py              # TSP路径规划算法
│   ├── ConvexHullFind.py   # 凸包计算
│   └── ...
├── tool/                    # 工具类
│   ├── file_io.py          # 文件读写工具
│   ├── visualization.py    # 可视化工具
│   ├── Metric.py           # 评估指标计算
│   ├── Basic.py            # 基础类（Point, Poi）
│   ├── Dis.py              # 距离计算工具
│   └── ...
├── example/                 # 示例代码
│   ├── partitioning_algo.py    # 分区算法主程序
│   └── ...
├── requirements.txt        # Python依赖
└── README.md              # 项目说明文档
```

## 🚀 安装

### 环境要求

- Python >= 3.8
- Windows / Linux / macOS

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd district-partitioning-algo
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **（可选）安装C++版本的平衡K-means算法**

如果需要使用C++版本的平衡K-means算法，需要：
- 从 [Balanced_k-Means_Revisited](https://github.com/uef-machine-learning/Balanced_k-Means_Revisited.git) 下载源码
- 编译生成可执行文件
- 在 `method/balance_kmeans.py` 中配置可执行文件路径

## 📖 快速开始

### 基本使用

```python
from example.partitioning_algo import run
from tool.file_io import read_poi_data, read_block_data, ALL_POLYGON

# 配置参数
bkm_params = {
    "numClusters": 6,
    "terminationCriterion": "MaxDiffClusterSizes",
    "terminationCriterionValue": 2,
    "maxIter": 20000
}

arg_dict = {
    "warehouse_type": None,
    "show_poi_result": True,
    "execute_CBKM": False,  # False: 使用Python版本; True: 使用C++版本
    "bkm_params": bkm_params,
}

# 运行分区算法
poi_file = 'data/shp/poi_mct.shp'
block_file = 'data/shp/block_mct.shp'
run(poi_file, block_file, arg_dict)
```

## 💡 使用示例

### 1. POI点聚类

```python
from method.bkmeans import BKMeans
from tool.file_io import read_poi_data
import geopandas as gpd

# 读取POI数据
poi_gdf = read_poi_data('data/shp/poi_mct.shp')

# 提取坐标
coords = [[geom.x, geom.y] for geom in poi_gdf.geometry]

# 初始化并运行平衡K-means
bkmeans = BKMeans()
bkmeans.initialize(coords, numClusters=6)
bkmeans.run(
    terminationCriterion="MaxDiffClusterSizes",
    terminationCriterionValue=2,
    maxIter=20000
)

# 获取聚类结果
labels = bkmeans.bestAssignment
poi_gdf['cluster'] = labels
```

### 2. Block分区

```python
from example.partitioning_algo import DistrictPartition
from tool.file_io import read_poi_data, read_block_data, ALL_POLYGON

# 读取数据
poi_gdf = read_poi_data('data/shp/poi_mct.shp')
block_gdf = read_block_data('data/shp/block_mct.shp', id=ALL_POLYGON)

# 创建分区对象
dp = DistrictPartition(poi_gdf, block_gdf, **arg_dict)

# 执行分区
dp.solve()

# 可视化结果
dp.visualize()

# 获取分区结果
result = dp.get_partition_result()
```

### 3. TSP路径规划

```python
from example.partitioning_algo import tsp_solver, cal_solution
from tool.Basic import Point

# 为每个区域计算TSP路径
routes = cal_solution(poi_gdf, depot_poi, label_filed='cluster')

# 可视化路径
from tool.visualization import plot_single_solution
plot_single_solution(block_gdf, routes)
```

### 4. 结果评估

```python
from tool.Metric import Metric

# 初始化评估器（需要包含完整的时间、顺序等信息）
metric = Metric(poi_gdf, block_gdf)

# 计算所有指标
metrics = metric.calculate_all_metrics()
print(metrics)

# 或打印所有指标
metric.print_metrics()
```

## 🔧 核心模块

### 1. 聚类算法 (`method/`)

- **BKMeans**: Python实现的平衡K-means算法
- **CBalanceKmeans**: C++版本的平衡K-means算法接口
- **TSP**: TSP路径规划算法（插入法）

### 2. 工具模块 (`tool/`)

- **file_io**: 文件读写工具，支持Shapefile格式
- **visualization**: 可视化工具，支持多种图表展示
- **Metric**: 评估指标计算，包含8项核心指标
- **Basic**: 基础数据结构（Point, Poi）
- **Dis**: 距离计算工具（欧几里得距离、球面距离）

### 3. 实体类 (`entity/`)

- **Coordinate**: 坐标类
- **Cluster**: 聚类簇类
- **Point**: 点类

## 📊 评估指标

`Metric` 类提供以下8项评估指标：

1. **Number**: 区域划分数量
2. **Total_Distance**: 总配送距离
3. **OTD_Rate**: 按时履约率（3小时内送达件数/总件数）
4. **Spatial_Compactness**: 空间紧凑性（总面积/轮廓线长度）
5. **Workload_SD**: 各区域配送量标准差
6. **Distance_SD**: 各区域配送距离标准差
7. **Time_SD**: 各区域配送时间标准差
8. **Delivery_Difficulty_SD**: 各区域件均距离标准差

### 使用评估指标

```python
from tool.Metric import Metric

# 确保POI和Block数据包含必要的字段
# POI数据需要: gid, qty, arrival_time, departure_time, transfer_time, 
#              block_id, poi_order, region_id
# Block数据需要: gid, qty, arrival_time, departure_time, block_order, region_id

metric = Metric(poi_gdf, block_gdf)
results = metric.calculate_all_metrics()

# 访问单个指标
number = metric.calculate_number()
total_distance = metric.calculate_total_distance()
otd_rate = metric.calculate_otd_rate(time_limit_hours=3.0)
```

## 📦 依赖说明

主要依赖包：

- **geopandas** >= 0.14.0: 地理数据处理
- **matplotlib** >= 3.7.0: 数据可视化
- **numpy** >= 1.24.0: 数值计算
- **pandas** >= 2.0.0: 数据处理
- **scipy** >= 1.10.0: 科学计算
- **shapely** >= 2.0.0: 几何操作
- **scikit-learn** >= 1.3.0: 机器学习算法
- **hdbscan** >= 0.8.0: HDBSCAN聚类算法

完整依赖列表请查看 `requirements.txt`。

## 🎯 主要功能说明

### DistrictPartition 类

主要的区域划分类，提供完整的划分流程：

```python
dp = DistrictPartition(poi_gdf, block_gdf, **kwargs)

# 主要方法：
dp.preprocess()          # 数据预处理
dp.poi_clustering()      # POI点聚类
dp.block_partition()     # Block分区
dp.swap_block_unit()     # 边界优化
dp.visualize()           # 可视化结果
dp.get_partition_result() # 获取分区结果
```

### 参数配置

```python
arg_dict = {
    "warehouse_type": None,           # 仓库类型
    "show_poi_result": True,          # 是否显示POI聚类结果
    "execute_CBKM": False,            # 是否使用C++版本
    "bkm_params": {                   # 平衡K-means参数
        "numClusters": 6,
        "terminationCriterion": "MaxDiffClusterSizes",
        "terminationCriterionValue": 2,
        "maxIter": 20000
    }
}
```

## 🔍 数据格式要求

### POI数据格式

Shapefile格式，必须包含以下字段：
- `gid`: 唯一标识
- `qty`: 配送量
- `delivery_o`: 配送订单号（111表示仓库点）
- `geometry`: 点几何（Point类型）

### Block数据格式

Shapefile格式，必须包含以下字段：
- `gid`: 唯一标识
- `Id`: Block ID
- `geometry`: 面几何（Polygon类型）

## 📝 注意事项

1. **C++版本平衡K-means**: 如需使用C++版本，需要先编译外部库并配置路径
2. **数据坐标系**: 确保POI和Block数据使用相同的坐标系
3. **内存占用**: 处理大规模数据时注意内存占用
4. **可视化后端**: 在PyCharm等IDE中可能需要配置matplotlib后端

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

详见 [LICENSE](LICENSE) 文件。

## 👤 作者

chendi

## 📧 联系方式

如有问题或建议，请提交Issue。

---

**最后更新**: 2025年

