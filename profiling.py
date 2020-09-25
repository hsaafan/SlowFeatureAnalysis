import timeit
import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt
import gc
import main

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
    main.run_incsfa(dynamic_copies=d)
    incsfa_time = timeit.default_timer() - start_incsfa

    return(incsfa_time, rsfa_time)

if __name__ == "__main__":
    d = np.arange(start=2, stop=11, step=1, dtype=int)
    rsfa_times = np.zeros_like(d, dtype=float)
    incsfa_times = np.zeros_like(d, dtype=float)
    for ind, val in enumerate(d):
        val = int(val)
        print(f"Profiling for {val} lagged copies")
        incsfa_times[ind], rsfa_times[ind] = profile_algos(val)
        gc.collect()
    plt.figure()
    d = d*33 + 33
    plt.rcParams.update({'font.size': 16})
    plt.subplots_adjust(0.17, 0.05, 0.95, 0.95, 0, 0.05)
    plt.scatter(d, rsfa_times, label="RSFA")
    plt.scatter(d, incsfa_times, label="IncSFA")
    plt.title("Training Time")
    plt.xlabel("Sample Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()
