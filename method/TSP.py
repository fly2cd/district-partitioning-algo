import math
import matplotlib.pyplot as plt

from method.ConvexHullFind import boundaryFind
from tool.Dis import calculateEuclideanDistance
import random as rd
from tool.Basic import Point


def pointsGenerate(xMin, yMin, xMax, yMax, nums):
    points = []
    for i in range(nums):
        points.append(Point(i, int((xMax-xMin) * rd.random() + xMin), int((yMax - yMin) * rd.random() + yMin)))
    return points

def pointsNormalize(points):
    points1 = [Point(p[0], p[1], p[2]) for p in points]
    return points1

def TSPInsert(points):
    if len(points) <= 1:
        return points

    Dis = {p.id:{p1.id:calculateEuclideanDistance(p.x, p.y, p1.x, p1.y) for p1 in points} for p in points}

    # 每个坐标对应的索引
    D = {}
    for p in points:
        D[p.id] = p

    # 凸包初始化
    boundary = boundaryFind([[p.x, p.y] for p in points])

    route = []
    for b in boundary[:-1]:
        for p in points:
            if p.x == b[0] and p.y == b[1]:
                route.append(p)
                break

    idsUsed = [p.id for p in route]
    idsLeft = [p.id for p in points if p.id not in idsUsed]

    while len(idsLeft) > 0:
        L = idsLeft
        dcosts = {}
        for idNew in L:
            dcosts[idNew] = {}
            for i in range(1, len(route)):
                dcosts[idNew][i] = Dis[route[i-1].id][idNew] + Dis[idNew][route[i].id] - Dis[route[i-1].id][route[i].id]
            dcosts[idNew][len(route)] = Dis[route[-1].id][idNew] + Dis[idNew][route[0].id] - Dis[route[-1].id][route[0].id]

        idNew0 = L[0]
        idex0 = len(route)
        dcost0 = Dis[route[-1].id][idNew0] + Dis[idNew0][route[0].id] - Dis[route[-1].id][route[0].id]
        for idNew in L:
            for i in range(1, len(route)+1):
                if dcost0 > dcosts[idNew][i]:
                    idNew0 = idNew
                    idex0 = i
                    dcost0 = dcosts[idNew][i]
        route.insert(idex0, D[idNew0])
        idsLeft.remove(idNew0)

    # 把七点拿到首位
    for i in range(len(route)):
        if route[i].id == points[0].id:
            route = route[i:] + route[:i]
            break

    #从后面可能更短
    if len(route) >=3 and calculateEuclideanDistance(route[0].x, route[0].y, route[1].x, route[1].y) > calculateEuclideanDistance(route[0].x, route[0].y, route[-1].x, route[-1].y):
        route = route[0:1] + route[1:][::-1]
    return route

def routePlot(route):
    x = []
    y = []
    for i in range(len(route)):
        x.append(route[i].x)
        y.append(route[i].y)

    plt.scatter(x, y)
    plt.plot(x, y, color='r')
    plt.show()

#定义一个函数来计算两点之间的欧几里得距离
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

#定义一个函数来计算路径的总长度
def calculate_route_length(route):
    total_length = 0
    for i in range(len(route) - 1):
        total_length += euclidean_distance(route[i], route[i+1])
    return total_length



if __name__ == '__main__':
    xMin, yMin, xMax, yMax, nums = 0, 0, 1000, 1000, 50
    points = pointsGenerate(xMin, yMin, xMax, yMax, nums)
    print(points)
    route = TSPInsert(points)
    print(route)
    routePlot(route)