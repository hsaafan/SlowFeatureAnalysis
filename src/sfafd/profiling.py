import timeit
import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt
import gc
from .examples import examples as main

def save_profile():
    pr = cProfile.Profile()
    pr.enable()

    main.run_rsfa(dynamic_copies=5)    

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()

    with open('incsfa_profile.txt', 'w+') as f:
        f.write(s.getvalue())
    return

def profile_algos(d=2):
    start_rsfa = timeit.default_timer()
    main.run_rsfa(dynamic_copies=d)
    rsfa_time = timeit.default_timer() - start_rsfa

    start_incsfa = timeit.default_timer()
    main.run_incsfa(dynamic_copies=d, use_SVD=False)
    incsfa_time = timeit.default_timer() - start_incsfa

    start_incsfa_svd = timeit.default_timer()
    main.run_incsfa(dynamic_copies=d, use_SVD=True)
    incsfa_time_svd = timeit.default_timer() - start_incsfa_svd

    return(incsfa_time, rsfa_time, incsfa_time_svd)

def plot_profile(d, time_array):
    plt.rcParams.update({'font.size': 16})
    plt.subplots_adjust(0.17, 0.05, 0.95, 0.95, 0, 0.05)
    for name, time in time_array:
        plt.scatter(d, time, label=name)
    plt.title("Training Time")
    plt.xlabel("Sample Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    d = np.arange(start=2, stop=17, step=1, dtype=int)
    rsfa_times = np.zeros_like(d, dtype=float)
    incsfa_times = np.zeros_like(d, dtype=float)
    incsfa_times_svd = np.zeros_like(d, dtype=float)
    for ind, val in enumerate(d):
        val = int(val)
        print(f"Profiling for {val} lagged copies")
        incsfa_times[ind], rsfa_times[ind], incsfa_times_svd[ind] = profile_algos(val)
        gc.collect()
    plt.figure()
    d = d*33 + 33
    np.save('rsfa_times.npy', rsfa_times)
    np.save('incsfa_times.npy', incsfa_times)
    np.save('incsfa_svd_times.npy', incsfa_times_svd)
