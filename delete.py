from multiprocessing import Pool

def f(x):
     return x*x

p = Pool(3)
q = Pool(3)
a = p.map(f, [1,2,3])
b = q.map(f,[10,11,12])
print(a,b)