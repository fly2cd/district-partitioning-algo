import functools
import math
import random as rd
import matplotlib.pyplot as plt


def pointsGenerate(xMin, yMin, xMax, yMax, nums):
    points = []
    for i in range(nums):
        points.append([int((xMax-xMin) * rd.random() + xMin),
                       int((yMax-yMin) * rd.random() + yMin)])

    return points

def cmp2d(x, y):
    if x[1] < y[1]:
        return -1
    elif x[1] > y[1]:
        return 1
    elif x[0] < y[0]:
        return -1
    elif x[0] > y[0]:
        return 1
    else:
        return 0

def boundaryFind(points):
    if len(points) == 0:
        return []

    points = sorted(points, key=functools.cmp_to_key(cmp2d))

    points1 = [points[0]]
    for i in range(1, len(points)):
        if points[i][0] != points1[-1][0] or points[i][1] != points1[-1][1]:
            points1.append(points[1])

    points = points1

    # 计算每两个点之间的角度
    LL_rad = []
    for i in range(len(points)):
        L_rad = []
        for j in range(len(points)):
            x1 = points[i][0]
            y1 = points[i][1]
            x2 = points[j][0]
            y2 = points[j][1]
            if x1 == x2 and y1 == y2:
                L_rad.append(0)
                continue
            if x2 > x1:
                if y2 >= y1:
                    L_rad.append(math.atan((y2 - y1) / (x2 - x1)))
                elif y2 < y1:
                    L_rad.append(2*math.pi + math.atan((y2 - y1) / (x2-x1)))
            elif x2 < x1:
                if y2 >= y1:
                    L_rad.append(math.pi +math.atan((y2 - y1)/(x2 - x1)))
                elif y2 < y1:
                    L_rad.append(math.pi + math.atan((y2 - y1)/(x2 -x1)))
            else:
                if y2 > y1:
                    L_rad.append(math.pi/2)
                else:
                    L_rad.append(3*math.pi/2)

        LL_rad.append(L_rad)

    LL_rad = [[round(1000*LL_rad[i][j])/1000 for j in range(len(LL_rad[i]))] for i in range(len(LL_rad))]

    # 先找到纵坐标最小的点
    points_used = [0 for i in range(len(points))]
    p0 = points[0]
    idex0 = 0
    rad0 = 0
    for i in range(len(points)):
        if p0[1] > points[i][1] or (p0[1] == points[i][1] and p0[0] > points[i][0]):
            p0 = points[i]
            idex0 = i

    boundary = []
    boundary.append(p0)
    points_used[idex0] = 1

    loop = 0
    while loop < len(points):
        loop += 1
        idexs = [i for i in range(len(points)) if LL_rad[idex0][i] >= rad0 and i != idex0]
        if len(idexs) == 0:
            break
        idex1 = idexs[0]
        for i in idexs:
            if LL_rad[idex0][idex1] > LL_rad[idex0][i]:
                idex1 = i
        boundary.append(points[idex1])
        if points_used[idex1] == 1:
            break
        points_used[idex1] = 1
        rad0 = LL_rad[idex0][idex1]
        idex0 = idex1

    return boundary

def clustersPlot(points, boundary):
    x = []
    y = []
    x1 = []
    y1 = []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    for i in range(len(boundary)):
        x1.append(boundary[i][0])
        y1.append(boundary[i][1])

    plt.scatter(x, y)
    plt.plot(x1, y1, color='r')

    plt.show()

if __name__ == '__main__':
    # points = pointsGenerate(0, 0, 1000, 1000, 30)
    points = [[1,1], [2,2], [1,1], [3,3], [2,2]]
    print(points)
    boundary = boundaryFind(points)
    print(boundary)
    clustersPlot(points, boundary)


