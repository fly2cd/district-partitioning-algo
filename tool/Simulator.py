"""
Created on 2025.12.31
@author: chendi
该类是一个模拟器类，当区域结果划分完成后，增加一些路程扰动因素，模拟生成各配送点的送达时间、交接时间和出发时间，以及在途时间和距离。
输入：
    1、配送点位数据（geopanda），属性信息[gid, qty（配送量）, arrival_time（送达时间）, departure_time（出发时间）,
                                    transfer_time（交接时长=出发时间-送达时间）, block_id(对应block数据中的gid), poi_order(点位配送顺序), region_id（所属区域ID）]
    2、block数据（geopanda）, 属性信息[gid, qty(block内的poi配送量之和), arrival_time（进入block的最早时间）, departure_time（离开block的最晚时间）, block_order(配送顺序) region_id（所属区域ID）]

"""


