import math

def CalVector(p0,p1):
    x0,y0,z0 = p0[0],p0[1],p0[2]
    x1,y1,z1 = p1[0],p1[1],p1[2]
    dx = (x1-x0) * 100
    dy = (y1-y0) * 100
    dz = (z1-z0) * 100
    dist = math.sqrt((dx**2)+(dy**2)+(dz**2))
    if dist!=0:
        Zx = math.degrees(math.acos(dx/dist))
        Zy = math.degrees(math.acos(dy/dist))
        Zz = math.degrees(math.acos(dz/dist))
    else:
        Zx = math.nan
        Zy = math.nan
        Zz = math.nan
    return [dist/100, Zx ,Zy, Zz]

def create_3d_cube(x,y,z):
    top = [
        (x-1,y+1,z+1),  (x,y+1,z+1),    (x+1,y+1,z+1),
        (x-1,y,z+1),    (x,y,z+1),      (x+1,y,z+1),
        (x-1,y-1,z+1),  (x,y-1,z+1),    (x+1,y-1,z+1)
    ]
    mid = [
        (x-1,y+1,z),  (x,y+1,z),        (x+1,y+1,z),
        (x-1,y,z),    (x,y,z),          (x+1,y,z),
        (x-1,y-1,z),  (x,y-1,z),        (x+1,y-1,z)
    ]
    bot = [
        (x-1,y+1,z-1),  (x,y+1,z-1),    (x+1,y+1,z-1),
        (x-1,y,z-1),    (x,y,z-1),      (x+1,y,z-1),
        (x-1,y-1,z-1),  (x,y-1,z-1),    (x+1,y-1,z-1)
    ]

    array = top + mid + bot

    return array

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)