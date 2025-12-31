import math

def calculateSphericalDistance(lng1, lat1, lng2, lat2, cof=1.25):
    lat1 = lat1* math.pi/180
    lat2 = lat2* math.pi/180
    lng1 = lng1* math.pi/180
    lng2 = lng2* math.pi/180
    return cof * 6371.393 * math.acos(max(min(math.cos(lat1) * math.cos(lat2) * math.cos(lng1-lng2) + math.sin(lat1) * math.sin(lat2), 1), -1))

def calculateEuclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))