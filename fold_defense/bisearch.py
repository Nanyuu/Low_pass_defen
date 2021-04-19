import numpy as np

my_array = np.array([0,1,2,2,2,3])

start = 0
end = my_array.size
target = 3

while(start < end):
    mid  = start + int((end-start)/2)
    if my_array[mid] < target:
        start = mid + 1
    else:
        end = mid

print("start = {}".format(start))
