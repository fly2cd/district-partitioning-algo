from typing import List, Union
import math

class Coordinate:
    def __init__(self, values: Union[List[float], int], dimension: int = None):
        """
        初始化坐标类
        :param values: 可以是坐标值列表或维度大小
        :param dimension: 维度大小（当values为维度大小时使用）
        """
        if isinstance(values, int):
            self.dimension = values
            self.values = [0.0] * values
        else:
            self.dimension = len(values)
            self.values = values.copy() if isinstance(values, list) else values

    def getSqrDistance(self, coord: 'Coordinate') -> float:
        """
        计算与另一个坐标之间的平方距离
        :param coord: 另一个坐标
        :return: 平方距离
        """
        sum_sq = 0.0
        for i in range(self.dimension):
            diff = self.values[i] - coord.values[i]
            sum_sq += diff * diff
        return math.sqrt(sum_sq) ** 2

    def getDimension(self) -> int:
        """
        获取坐标维度
        :return: 维度大小
        """
        return self.dimension

    def addCoord(self, coord: 'Coordinate'):
        """
        将另一个坐标加到当前坐标上
        :param coord: 要添加的坐标
        """
        for i in range(self.dimension):
            self.values[i] += coord.values[i]

    def subCoord(self, coord: 'Coordinate'):
        """
        从当前坐标中减去另一个坐标
        :param coord: 要减去的坐标
        """
        for i in range(self.dimension):
            self.values[i] -= coord.values[i]

    def newCentroid(self, numPoints: int, centroid: 'Coordinate'):
        """
        计算新的质心坐标
        :param numPoints: 点的数量
        :param centroid: 用于存储新质心的坐标对象
        """
        for i in range(self.dimension):
            centroid.values[i] = self.values[i] / float(numPoints)

    def copy(self) -> 'Coordinate':
        """
        创建当前坐标的副本
        :return: 新的坐标对象
        """
        return Coordinate(self.values.copy(), self.dimension)

    def getValueInDim(self, dim: int) -> float:
        """
        获取指定维度的值
        :param dim: 维度索引
        :return: 该维度的值
        """
        return self.values[dim]

    def __str__(self) -> str:
        """
        返回坐标的字符串表示
        :return: 坐标的字符串表示
        """
        return f"Coordinate({self.values})"

    def __repr__(self) -> str:
        """
        返回坐标的详细字符串表示
        :return: 坐标的详细字符串表示
        """
        return f"Coordinate(values={self.values}, dimension={self.dimension})"


class Cluster:
    def __init__(self, centroid: Coordinate):
        """
        初始化簇类
        :param centroid: 簇的中心点坐标
        """
        self.centroid = centroid.copy()
        self.numPointsSeq = 0
        self.sumCoords = Coordinate(centroid.getDimension())

    def addPointSeq(self):
        """
        增加簇中的点数
        """
        self.numPointsSeq += 1

    def removePointSeq(self):
        """
        减少簇中的点数
        """
        self.numPointsSeq -= 1

    def addCoordSeq(self, coord: Coordinate):
        """
        添加坐标到簇中
        :param coord: 要添加的坐标
        """
        self.sumCoords.addCoord(coord)

    def removeCoordSeq(self, oldCoord: Coordinate):
        """
        从簇中移除坐标
        :param oldCoord: 要移除的坐标
        """
        self.sumCoords.subCoord(oldCoord)

    def getSqrDistance(self, coord: Coordinate) -> float:
        """
        计算给定坐标到簇中心的平方距离
        :param coord: 要计算距离的坐标
        :return: 平方距离
        """
        return self.centroid.getSqrDistance(coord)

    def setCentroidSeq(self):
        """
        根据当前点的总和更新簇中心
        """
        self.sumCoords.newCentroid(self.numPointsSeq, self.centroid)

    def getNumPointsSeq(self) -> int:
        """
        获取簇中的点数
        :return: 点数
        """
        return self.numPointsSeq

    def getCentroid(self) -> Coordinate:
        """
        获取簇的中心点
        :return: 中心点坐标
        """
        return self.centroid

    def __str__(self) -> str:
        """
        返回簇的字符串表示
        """
        return f"Cluster(centroid={self.centroid}, numPoints={self.numPointsSeq})"

    def __repr__(self) -> str:
        """
        返回簇的详细字符串表示
        """
        return f"Cluster(centroid={self.centroid}, numPoints={self.numPointsSeq}, sumCoords={self.sumCoords})"


class Point:
    def __init__(self, coord: Coordinate):
        """
        初始化点类
        :param coord: 坐标对象
        """
        self.coord = coord
        self.clusterId = -1

    def getClusterId(self) -> int:
        """
        获取点所属的簇ID
        :return: 簇ID
        """
        return self.clusterId

    def getCoord(self) -> Coordinate:
        """
        获取点的坐标
        :return: 坐标对象
        """
        return self.coord

    def assignToCluster(self, clusters: List[Cluster], numClusters: int, numIter: int,
                        penaltyNow: float, penaltyNext: float,
                        partlyRemainingFraction: float):
        """
        将点分配到簇
        :param clusters: 簇列表
        :param numClusters: 簇的数量
        :param numIter: 当前迭代次数
        :param penaltyNow: 当前惩罚值
        :param penaltyNext: 下一个惩罚值
        :param partlyRemainingFraction: 部分剩余分数
        """
        if numIter == 0:
            self.assignmentStandard(clusters, numClusters)
        else:
            oldClusterId = self.clusterId
            # 从旧簇中移除点
            if clusters[oldClusterId].getNumPointsSeq() == 1:
                # 簇只包含一个点，不做任何操作以避免除以零
                return

            clusters[oldClusterId].removePointSeq()
            clusters[oldClusterId].removeCoordSeq(self.coord)
            clusters[oldClusterId].setCentroidSeq()

            self.assignmentWithPenalty(clusters, numClusters, penaltyNow,
                                       penaltyNext, partlyRemainingFraction, oldClusterId)

        clusters[self.clusterId].addPointSeq()
        clusters[self.clusterId].addCoordSeq(self.coord)
        if numIter != 0:
            clusters[self.clusterId].setCentroidSeq()

    def assignmentWithPenalty(self, clusters: List[Cluster], numClusters: int,
                              penaltyNow: float, penaltyNext: float,
                              partlyRemainingFraction: float, oldClusterId: int):
        """
        使用惩罚项进行簇分配
        """
        cost = float('inf')
        sqrDistOldCluster = clusters[oldClusterId].getSqrDistance(self.coord)
        numPointsOldCluster = clusters[oldClusterId].getNumPointsSeq() + partlyRemainingFraction

        for j in range(numClusters):
            sqrDistJ = clusters[j].getSqrDistance(self.coord)
            numPointsJ = clusters[j].getNumPointsSeq()
            denom = numPointsOldCluster - numPointsJ
            if denom == 0:
                # 跳过分母为0的情况，防止ZeroDivisionError
                continue
            penaltyNeeded = (sqrDistJ - sqrDistOldCluster) / denom

            if numPointsOldCluster > numPointsJ:
                if penaltyNow < penaltyNeeded:
                    if penaltyNeeded < penaltyNext:
                        penaltyNext = penaltyNeeded
                else:
                    if sqrDistJ + penaltyNow * numPointsJ < cost and j != oldClusterId:
                        cost = sqrDistJ + penaltyNow * numPointsJ
                        self.clusterId = j
            else:
                if penaltyNow < penaltyNeeded and sqrDistJ + penaltyNow * numPointsJ < cost:
                    cost = sqrDistJ + penaltyNow * numPointsJ
                    self.clusterId = j

    def assignmentStandard(self, clusters: List[Cluster], numClusters: int):
        """
        标准簇分配（无惩罚项）
        """
        cost = float('inf')
        for j in range(numClusters):
            sqrDist = clusters[j].getSqrDistance(self.coord)
            if sqrDist < cost:
                cost = sqrDist
                self.clusterId = j

    def computeSqrDist(self, cluster: Cluster) -> float:
        """
        计算点到簇中心的平方距离
        """
        return self.coord.getSqrDistance(cluster.getCentroid())

    def computeAngle(self, end: 'Point') -> float:
        """
        计算与另一个点之间的角度
        """
        if end is None:
            return 0.0

        xStart = self.coord.getValueInDim(0)
        xEnd = end.getCoord().getValueInDim(0)
        yStart = self.coord.getValueInDim(1)
        yEnd = end.getCoord().getValueInDim(1)

        GK = yEnd - yStart
        H = math.sqrt(self.coord.getSqrDistance(end.getCoord()))
        angle = math.asin(GK / H)

        if xEnd < xStart:
            angle = math.pi - angle

        return angle

    def orderAngle(self, points: List['Point'], size: int):
        """
        根据角度对点进行排序
        """
        self.mergeSortAngle(points, 1, size - 1)

    def mergeSortAngle(self, points: List['Point'], start: int, end: int):
        """
        归并排序（按角度）
        """
        if start < end:
            middle = (start + end) // 2
            self.mergeSortAngle(points, start, middle)
            self.mergeSortAngle(points, middle + 1, end)
            self.mergeAngle(points, start, end, middle)

    def mergeAngle(self, points: List['Point'], start: int, end: int, middle: int):
        """
        归并排序的合并步骤
        """
        left = start
        right = middle + 1
        pointsWork = [None] * (end - start + 1)
        counter = 0

        while left <= middle or right <= end:
            if left > middle:
                pointsWork[counter] = points[right]
                right += 1
            elif right > end:
                pointsWork[counter] = points[left]
                left += 1
            elif self.computeAngle(points[left]) < self.computeAngle(points[right]):
                pointsWork[counter] = points[left]
                left += 1
            else:
                if self.computeAngle(points[left]) > self.computeAngle(points[right]):
                    pointsWork[counter] = points[right]
                    right += 1
                else:
                    # 如果角度相等，则删除x坐标较小的点
                    xValueLeft = points[left].getCoord().getValueInDim(0)
                    xValueRight = points[right].getCoord().getValueInDim(0)
                    if xValueLeft < xValueRight:
                        points[left] = None
                        pointsWork[counter] = points[left]
                        left += 1
                    else:
                        points[right] = None
                        pointsWork[counter] = points[right]
                        right += 1
            counter += 1

        for i in range(end - start + 1):
            points[start + i] = pointsWork[i]

    def isLeft(self, start: 'Point', end: 'Point') -> bool:
        """
        判断点是否在向量start->end的左侧
        """
        xStart = start.coord.getValueInDim(0)
        yStart = start.coord.getValueInDim(1)
        xEnd = end.coord.getValueInDim(0)
        yEnd = end.coord.getValueInDim(1)
        xThis = self.coord.getValueInDim(0)
        yThis = self.coord.getValueInDim(1)

        leftFrom = (xEnd - xStart) * (yThis - yStart) - (xThis - xStart) * (yEnd - yStart)
        return leftFrom > 0.0

    def __str__(self) -> str:
        """
        返回点的字符串表示
        """
        return f"Point(coord={self.coord}, clusterId={self.clusterId})"

    def __repr__(self) -> str:
        """
        返回点的详细字符串表示
        """
        return f"Point(coord={self.coord}, clusterId={self.clusterId})"

# 使用示例
if __name__ == "__main__":
    # 创建坐标和簇
    centroid = Coordinate([1.0, 2.0])
    cluster = Cluster(centroid)

    # 添加点
    point1 = Coordinate([1.5, 2.5])
    point2 = Coordinate([0.5, 1.5])

    cluster.addPointSeq()
    cluster.addCoordSeq(point1)
    cluster.addPointSeq()
    cluster.addCoordSeq(point2)

    # 更新中心点
    cluster.setCentroidSeq()

    # 打印信息
    print(f"Cluster center: {cluster.getCentroid()}")
    print(f"Number of points: {cluster.getNumPointsSeq()}")

    # 计算距离
    test_point = Coordinate([2.0, 3.0])
    distance = cluster.getSqrDistance(test_point)
    print(f"Distance to test point: {distance}")

    # 移除点
    cluster.removePointSeq()
    cluster.removeCoordSeq(point1)
    cluster.setCentroidSeq()
    print(f"After removing a point - center: {cluster.getCentroid()}")