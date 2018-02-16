import timeit
import numpy as np 

longlist = list(range(int(1e6)))

#idx = np.random.randint(total_transition_count, size = 16)

time_indv = timeit.timeit('[longlist[i] for i in np.random.randint(1e6, size = 16)]', number = 100000, globals=globals())
time_batch = timeit.timeit('[longlist[i:i+16] for i in [np.random.randint(1e6)]]', number = 100000, globals=globals())

print(time_indv, time_batch)
