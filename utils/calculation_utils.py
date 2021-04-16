def interval_map_function(a,b,c, d):
    def map(x):
        return c + (d - c)/ (b-a) * (x-a)
    return map


