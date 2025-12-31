class Point:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return str([self.id, self.x, self.y])

# 计算点信息，在基础点上添加了货量，停留时间、到达时间和最晚到达时间
class Poi(Point):
    def __init__(self, id, x, y, demand):
        Point.__init__(self, id, x, y)
        self.demand = demand
        self.timeStay = None
        self.timeArrive = None
        self.deadline = None
        self.demandsMin = None
        self.demandsMax = None

    def __repr__(self):
        return str([self.id, self.x, self.y, self.demand])


# class Region:
