import numpy as np
import math
import random
from typing import List, Set
import matplotlib.pyplot as plt
from entity.cluster import Cluster, Coordinate, Point
import geopandas as gpd

class Gnuplot:
    def __init__(self):
        """
        Gnuplot类用于兼容C++代码中的gnuplot命令收集，实际在Python中不做绘图，仅做接口保留。
        """
        self.commands = []

    def __call__(self, command: str):
        """
        收集gnuplot命令（无实际作用，仅做兼容）
        :param command: gnuplot命令字符串
        """
        self.commands.append(command)

    def execute(self):
        """
        执行gnuplot命令（在Python中无实际作用）
        """
        pass

class BKMeans:
    def __init__(self):
        """
        BKMeans类，平衡K均值聚类算法的主类。
        """
        self.dimension = 0
        self.size = 0
        self.numClusters = 0
        self.points = []
        self.clusters = []
        self.bestAssignment = []

    def initialize(self, vec: List[List[float]], numClusters: int):
        """
        初始化聚类算法，包括数据点和聚类中心。
        :param vec: 输入数据，二维数组，每行为一个样本
        :param numClusters: 聚类数目
        """
        self.size = len(vec)
        self.dimension = len(vec[0])
        self.numClusters = numClusters
        self.points = []
        self.bestAssignment = [0] * self.size

        # 初始化点
        for i in range(self.size):
            coord = Coordinate(vec[i], self.dimension)
            self.points.append(Point(coord))

        if numClusters > self.size:
            print("More clusters than points!")
            return

        self.initializeCenters()

    def initializeCenters(self):
        """
        随机初始化聚类中心。
        """
        self.clusters = []
        used_points = set()

        while len(self.clusters) < self.numClusters:
            if self.size < 1000000:  # 模拟 RAND_MAX
                random_int = random.randint(0, self.size - 1)
            else:
                random_int = int(random.random() * self.size)

            if random_int not in used_points:
                cluster = Cluster(self.points[random_int].getCoord())
                self.clusters.append(cluster)
                used_points.add(random_int)

    def run(self, terminationCriterion: str, terminationCriterionValue: float,
            stopWhenBalanced: bool = False, partlyRemainingFraction: float = 0.0,
            increasingPenaltyFactor: float = 1.0, useFunctionIter: bool = False,
            switchPostp: int = 0, maxIter: int = 100):
        """
        运行平衡K均值聚类主循环。
        :param terminationCriterion: 终止条件类型（如MaxDiffClusterSizes等）
        :param terminationCriterionValue: 终止条件阈值
        :param stopWhenBalanced: 达到平衡时是否立即终止
        :param partlyRemainingFraction: 部分剩余分数（用于惩罚项）
        :param increasingPenaltyFactor: 惩罚项增长因子
        :param useFunctionIter: 是否使用自定义迭代惩罚函数
        :param switchPostp: 后处理交换次数
        :param maxIter: 最大迭代次数
        """
        penaltyNow = 0.0
        penaltyNext = float('inf')
        balanceReq = False
        balanced = False
        terminate = False
        keepPenalty = False
        bestMSE = float('inf')
        numIter = 0

        while not terminate:
            # 分配点到簇
            for point in self.points:
                point.assignToCluster(self.clusters, self.numClusters, numIter,
                                      penaltyNow, penaltyNext, partlyRemainingFraction)

            if maxIter == 0:
                self.saveAssignments()
                return

            # 检查平衡要求
            if terminationCriterion == "MaxDiffClusterSizes":
                balanceReq = self.checkMaxDiffClusterSizes(int(terminationCriterionValue))
            elif terminationCriterion == "MaxSDCS":
                balanceReq = self.checkMaxSDCS(terminationCriterionValue)
            elif terminationCriterion == "MinNormEntro":
                balanceReq = self.checkMinNormEntro(terminationCriterionValue)
            elif terminationCriterion == "MinClusterSize":
                balanceReq = self.checkMinClusterSize(int(terminationCriterionValue))

            balanced = self.checkMaxDiffClusterSizes(1)

            if balanceReq:
                currentMSE = self.meanSquaredError()
                if currentMSE < bestMSE:
                    bestMSE = currentMSE
                    self.saveAssignments()
                    keepPenalty = True
                elif stopWhenBalanced or balanced:
                    terminate = True
                    keepPenalty = True

            if numIter != 0 and not keepPenalty:
                if useFunctionIter:
                    penaltyNow = self.functionIter(numIter) * penaltyNext
                else:
                    penaltyNow = increasingPenaltyFactor * penaltyNext
                penaltyNext = float('inf')

            keepPenalty = False
            numIter += 1

            if numIter > maxIter:
                break

        if not (numIter >= maxIter):
            self.restoreBestResults()

        if switchPostp > 0:
            self.performSwitchOptimization(switchPostp)

        print(f"iterations={numIter}")
        self.saveAssignments()

    def functionIter(self, numIter: int) -> float:
        """
        自定义迭代惩罚函数。
        :param numIter: 当前迭代次数
        :return: 惩罚系数
        """
        if numIter > 100:
            return 1.01
        else:
            return 1.1009 - 0.0009 * numIter

    def saveAssignments(self):
        """
        保存当前最优聚类分配结果。
        """
        for i in range(self.size):
            self.bestAssignment[i] = self.points[i].getClusterId()

    def meanSquaredError(self) -> float:
        """
        计算均方误差（MSE）。
        :return: MSE值
        """
        return self.sumOfSquaredErrors() / self.size

    def sumOfSquaredErrors(self) -> float:
        """
        计算所有点到其簇中心的平方误差和。
        :return: SSE值
        """
        sse = 0.0
        for point in self.points:
            clusterId = point.getClusterId()
            sqrError = point.computeSqrDist(self.clusters[clusterId])
            sse += sqrError
        return sse

    def checkMaxDiffClusterSizes(self, maxDiffClusterSizes: int) -> bool:
        """
        检查最大簇大小差是否满足阈值。
        :param maxDiffClusterSizes: 最大允许的簇大小差
        :return: 是否满足
        """
        diffClusterSizes = self.computeDiffClusterSizes()
        return diffClusterSizes <= maxDiffClusterSizes

    def computeDiffClusterSizes(self) -> int:
        """
        计算最大簇大小差。
        :return: 最大簇大小差
        """
        maxClusterSize = 0
        minClusterSize = self.size
        for cluster in self.clusters:
            clusterSize = cluster.getNumPointsSeq()
            maxClusterSize = max(maxClusterSize, clusterSize)
            minClusterSize = min(minClusterSize, clusterSize)
        return maxClusterSize - minClusterSize

    def checkMinClusterSize(self, minClusterSize: int) -> bool:
        """
        检查所有簇的最小大小是否满足阈值。
        :param minClusterSize: 最小允许的簇大小
        :return: 是否满足
        """
        for cluster in self.clusters:
            if cluster.getNumPointsSeq() < minClusterSize:
                return False
        return True

    def checkMaxSDCS(self, maxSDCS: float) -> bool:
        """
        检查标准差是否满足阈值。
        :param maxSDCS: 最大允许的标准差
        :return: 是否满足
        """
        sdcs = self.computeSDCS()
        return sdcs <= maxSDCS

    def computeSDCS(self) -> float:
        """
        计算簇大小的标准差。
        :return: 标准差
        """
        avgSize = self.size / self.numClusters
        sum_sq = 0.0
        for cluster in self.clusters:
            clusterSize = cluster.getNumPointsSeq()
            sum_sq += (clusterSize - avgSize) ** 2
        return math.sqrt(sum_sq / (self.numClusters - 1))

    def checkMinNormEntro(self, minNormEntro: float) -> bool:
        """
        检查归一化熵是否满足阈值。
        :param minNormEntro: 最小允许的归一化熵
        :return: 是否满足
        """
        normEntro = self.computeNormEntro()
        return normEntro >= minNormEntro

    def computeNormEntro(self) -> float:
        """
        计算归一化熵。
        :return: 归一化熵
        """
        sum_entropy = 0.0
        for cluster in self.clusters:
            clusterSize = cluster.getNumPointsSeq()
            p = clusterSize / self.size
            sum_entropy += p * math.log2(p)
        return -sum_entropy / math.log2(self.numClusters)

    def restoreBestResults(self):
        """
        恢复最优聚类分配结果。
        """
        for i in range(self.size):
            oldClusterId = self.points[i].getClusterId()
            newClusterId = self.bestAssignment[i]
            if newClusterId != oldClusterId:
                coord = self.points[i].getCoord()
                self.clusters[oldClusterId].removePointSeq()
                self.clusters[oldClusterId].removeCoordSeq(coord)
                self.clusters[newClusterId].addPointSeq()
                self.clusters[newClusterId].addCoordSeq(coord)
                self.points[i].setClusterId(newClusterId)

        for cluster in self.clusters:
            cluster.setCentroidSeq()

    def performSwitchOptimization(self, maxIterations: int):
        """
        后处理交换优化，进一步平衡簇。
        :param maxIterations: 最大交换迭代次数
        """
        cluvec = [set() for _ in range(self.numClusters)]
        for i in range(self.size):
            cluvec[self.points[i].clusterId].add(i)

        iter = 1
        while iter <= maxIterations:
            switchCount = 0
            for cluAid in range(self.numClusters - 1):
                for cluBid in range(cluAid + 1, self.numClusters):
                    switchCount += self.switchOpt(cluAid, cluBid, cluvec[cluAid], cluvec[cluBid])
                    self.clusters[cluAid].setCentroidSeq()
                    self.clusters[cluBid].setCentroidSeq()

            if switchCount == 0:
                break
            iter += 1

        self.saveAssignments()

    def switchOpt(self, cluAid: int, cluBid: int, cluA: Set[int], cluB: Set[int]) -> int:
        """
        两个簇之间的点交换优化。
        :param cluAid: 簇A编号
        :param cluBid: 簇B编号
        :param cluA: 簇A的点索引集合
        :param cluB: 簇B的点索引集合
        :return: 交换次数
        """
        cluAdist = []
        cluBdist = []

        for x in cluA:
            distB = self.clusters[cluBid].getSqrDistance(self.points[x].coord)
            distA = self.clusters[cluAid].getSqrDistance(self.points[x].coord)
            delta = distB - distA
            cluAdist.append((x, delta))

        cluAdist.sort(key=lambda x: x[1])

        for x in cluB:
            distB = self.clusters[cluBid].getSqrDistance(self.points[x].coord)
            distA = self.clusters[cluAid].getSqrDistance(self.points[x].coord)
            delta = distA - distB
            cluBdist.append((x, delta))

        cluBdist.sort(key=lambda x: x[1])

        minSize = min(len(cluBdist), len(cluAdist))
        switchCount = 0

        for i in range(minSize):
            switchDelta = cluAdist[i][1] + cluBdist[i][1]
            id1 = cluAdist[i][0]
            id2 = cluBdist[i][0]

            if switchDelta < 0.0:
                cluA.remove(id1)
                cluB.remove(id2)
                cluA.add(id2)
                cluB.add(id1)

                self.clusters[cluAid].removeCoordSeq(self.points[id1].coord)
                self.clusters[cluAid].addCoordSeq(self.points[id2].coord)
                self.clusters[cluBid].removeCoordSeq(self.points[id2].coord)
                self.clusters[cluBid].addCoordSeq(self.points[id1].coord)

                self.points[id1].clusterId, self.points[id2].clusterId = self.points[id2].clusterId, self.points[id1].clusterId
                switchCount += 1
            else:
                break

        return switchCount

    def showResultsConvexHull(self, nameDataSet: str, run: int, timeInSec: float):
        """
        可视化聚类结果及凸包。
        :param nameDataSet: 数据集名称
        :param run: 运行编号
        :param timeInSec: 运行耗时
        """
        visualizer = KMeansVisualizer(self)
        visualizer.showResults(nameDataSet, run, timeInSec)

    def getBounds(self) -> List[List[float]]:
        """
        获取所有点的x、y边界范围。
        :return: [[xMin, xMax], [yMin, yMax]]
        """
        xMin = float('inf')
        xMax = float('-inf')
        yMin = float('inf')
        yMax = float('-inf')

        for point in self.points:
            x = point.coord.getValueInDim(0)
            y = point.coord.getValueInDim(1)
            xMin = min(xMin, x)
            xMax = max(xMax, x)
            yMin = min(yMin, y)
            yMax = max(yMax, y)

        return [[xMin, xMax], [yMin, yMax]]


class KMeansVisualizer:
    def __init__(self, kmeans: BKMeans):
        """
        KMeansVisualizer类，用于可视化聚类结果。
        :param kmeans: BKMeans实例
        """
        self.kmeans = kmeans
        self.fig = None
        self.ax = None

    def showResults(self, nameDataSet: str, run: int, timeInSec: float):
        """
        绘制聚类结果和凸包，并保存为图片。
        :param nameDataSet: 数据集名称
        :param run: 运行编号
        :param timeInSec: 运行耗时
        """
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(15, 10))

        # 获取数据边界
        bounds = self.kmeans.getBounds()
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        # 调整边界使图形为正方形
        x_length = x_max - x_min
        y_length = y_max - y_min
        if x_length > y_length:
            y_correction = (x_length - y_length) / 2.0
            y_min -= y_correction
            y_max += y_correction
        else:
            x_correction = (y_length - x_length) / 2.0
            x_min -= x_correction
            x_max += x_correction

        # 设置图形范围
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # 绘制数据点
        points_x = [p.coord.getValueInDim(0) for p in self.kmeans.points]
        points_y = [p.coord.getValueInDim(1) for p in self.kmeans.points]
        self.ax.scatter(points_x, points_y, c='black', s=10, alpha=0.6)

        # 计算每个簇的凸包
        for i in range(self.kmeans.numClusters):
            cluster_points = [p for p in self.kmeans.points if p.clusterId == i]
            if len(cluster_points) > 2:
                hull_points = self.computeConvexHull(cluster_points)
                hull_x = [p.coord.getValueInDim(0) for p in hull_points]
                hull_y = [p.coord.getValueInDim(1) for p in hull_points]
                # 闭合凸包
                hull_x.append(hull_x[0])
                hull_y.append(hull_y[0])
                self.ax.plot(hull_x, hull_y, 'k-', linewidth=1)

        # 绘制聚类中心
        meanPointsPerCluster = self.kmeans.size / self.kmeans.numClusters
        for i, cluster in enumerate(self.kmeans.clusters):
            x = cluster.centroid.getValueInDim(0)
            y = cluster.centroid.getValueInDim(1)
            clusterSize = cluster.getNumPointsSeq()

            # 根据簇的大小选择颜色
            if clusterSize < math.floor(meanPointsPerCluster):
                color = 'green'  # 点数过少
            elif clusterSize > math.ceil(meanPointsPerCluster):
                color = 'red'    # 点数过多
            else:
                color = 'blue'   # 点数适中

            self.ax.scatter(x, y, c=color, s=100, marker='o')

        # 设置标题
        title = f'MSE = {self.kmeans.meanSquaredError():.4f}, '
        title += f'diffClusterSizes = {self.kmeans.computeDiffClusterSizes()}, '
        title += f'SDCS = {self.kmeans.computeSDCS():.4f}, '
        title += f'normEntro = {self.kmeans.computeNormEntro():.4f}, '
        title += f'timeInSec = {timeInSec:.2f}'
        self.ax.set_title(title)

        # 移除坐标轴
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(False)

        # 保存图形
        plt.savefig(f'/Users/chendi/IdeaProjects/regionOpt/method/convexHulls/{nameDataSet}_{run}_ConvexHull.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def computeConvexHull(self, points: List[Point]) -> List[Point]:
        """
        计算点集的凸包（Graham扫描法）。
        :param points: 点对象列表
        :return: 凸包上的点列表
        """
        if len(points) <= 2:
            return points

        # 找到y坐标最小的点
        start_point = min(points, key=lambda p: (p.coord.getValueInDim(1),
                                                 p.coord.getValueInDim(0)))

        # 计算其他点相对于起始点的角度
        angles = []
        for p in points:
            if p != start_point:
                dx = p.coord.getValueInDim(0) - start_point.coord.getValueInDim(0)
                dy = p.coord.getValueInDim(1) - start_point.coord.getValueInDim(1)
                angle = math.atan2(dy, dx)
                angles.append((p, angle))

        # 按角度排序
        angles.sort(key=lambda x: x[1])
        sorted_points = [start_point] + [p for p, _ in angles]

        # Graham扫描算法
        hull = [sorted_points[0], sorted_points[1]]
        for i in range(2, len(sorted_points)):
            while len(hull) > 1 and not sorted_points[i].isLeft(hull[-2], hull[-1]):
                hull.pop()
            hull.append(sorted_points[i])

        return hull


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    data = np.random.randn(100, 2) * 2
    data = data.tolist()

    # 创建KMeans实例
    bkmeans = BKMeans()
    bkmeans.initialize(data, numClusters=3)

    # 运行算法
    bkmeans.run(
        terminationCriterion="MaxDiffClusterSizes",
        terminationCriterionValue=1,
        maxIter=50000
    )

    # 可视化结果
    bkmeans.showResultsConvexHull("test", 2, 0.0)