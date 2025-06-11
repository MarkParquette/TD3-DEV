from plot_results import sliding_average
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import torch as T

def read_results(policy, env, seed, discounted=False, path='./baseline_results'):
    file_name = f"{path}/{policy}_{env}_{seed}{'_disc' if discounted else ''}.npy"

    try:
        data = np.load(file_name)
    except:
        data = None

    return data

def gen_detailed_report(policies, envs, seeds, path):
    print("Seed,                       Env,     Policy,     Mean,      Max,      Ave,      Max,   (Mean),    (Max),    (Ave),    (Max),  Count")

    for e in envs:
        for s in seeds:
            for p in policies:
                data = read_results(p, e, s, discounted=False, path=path)
                disc_data = read_results(p, e, s, discounted=True, path=path)

                if data is None or disc_data is None:
                    continue

                ud_mean = np.mean(data)
                ud_max = np.max(data)

                ud_sa = sliding_average(data, 10)
                ave_mean = np.mean(ud_sa)
                ave_max = np.max(ud_sa)

                disc_mean = np.mean(disc_data)
                disc_max = np.max(disc_data)

                d_sa = sliding_average(disc_data, 10)
                disc_ave_mean = np.mean(d_sa)
                disc_ave_max = np.max(d_sa)
                
                print(f"{s:>4}, {e:>25}, {p:>10}, {ud_mean:>8.2f}, {ud_max:>8.2f}, {ave_mean:>8.2f}, {ave_max:>8.2f}, {disc_mean:>8.2f}, {disc_max:>8.2f}, {disc_ave_mean:>8.2f}, {disc_ave_max:>8.2f}, {len(data):>6}")

def gen_summary_report(policies, envs, path):
    print("Seed,                       Env,     Policy,     Mean,      Max,      Ave,      Max,   (Mean),    (Max),    (Ave),    (Max),  Count")

    for e in envs:
            for p in policies:
                data = read_results(p, e, "Ave", discounted=False, path=path)
                disc_data = read_results(p, e, "Ave", discounted=True, path=path)

                if data is None or disc_data is None:
                    continue

                ud_mean = np.mean(data)
                ud_max = np.max(data)

                ud_sa = sliding_average(data, 10)
                ave_mean = np.mean(ud_sa)
                ave_max = np.max(ud_sa)

                disc_mean = np.mean(disc_data)
                disc_max = np.max(disc_data)

                d_sa = sliding_average(disc_data, 10)
                disc_ave_mean = np.mean(d_sa)
                disc_ave_max = np.max(d_sa)
                
                print(f" Ave, {e:>25}, {p:>10}, {ud_mean:>8.2f}, {ud_max:>8.2f}, {ave_mean:>8.2f}, {ave_max:>8.2f}, {disc_mean:>8.2f}, {disc_max:>8.2f}, {disc_ave_mean:>8.2f}, {disc_ave_max:>8.2f}, {len(data):>6}")

    pass

def gen_summary_files(policies, envs, seeds, path):
    print("Seed,                       Env,     Policy,     Mean,      Max,      Ave,      Max,   (Mean),    (Max),    (Ave),    (Max),  Count")

    for e in envs:
        for p in policies:
            summary = []
            disc_summary = []

            for s in seeds:
                data = read_results(p, e, s, discounted=False, path=path)
                disc_data = read_results(p, e, s, discounted=True, path=path)

                if data is None or disc_data is None:
                    continue

                summary.append(data)
                disc_summary.append(disc_data)

            summary = T.FloatTensor(np.array(summary))
            print(f"summary {e}:{p} {summary} {summary.dim()}")
            summary_mean = np.array(T.mean(summary, dim=0, keepdim=True)[0])
            print(summary_mean)

            disc_summary = T.FloatTensor(np.array(disc_summary))
            print(f"disc_summary {e}:{p} {disc_summary} {disc_summary.dim()}")
            disc_summary_mean = np.array(T.mean(disc_summary, dim=0, keepdim=True)[0])
            print(disc_summary_mean)

            summary_fname = f"{path}/{p}_{e}_Ave.npy"
            print(summary_fname)
            np.save(summary_fname, summary_mean)

            disc_summary_fname = f"{path}/{p}_{e}_Ave_disc.npy"
            print(disc_summary_fname)
            np.save(disc_summary_fname, disc_summary_mean)

def export_results(results_path, gamma=0.99, steps_per_result=5000):
    files = [f for f in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, f))]
    try:
        files.remove(".DS_Store") 
    except:
        pass

    print("Seed,                       Env,     Policy,   Discount,    Step,    Rewards")

    for file in files:
        #print(file)
        policy, env, seed, disc = re.split("[_.]", file)[0:4]
        disc = disc == "disc"
        discount = 1.0 if not disc else gamma
        try:
            data = np.load(os.path.join(results_path, file))
        except:
            print("*** FAILED TO READ FILE {file}")
            exit(0)
        
        x = 0
        for y in data:
            print(f"{seed:>4}, {env:>25}, {policy:>10}, {discount:>10.2f}, {int(x):>7d}, {y:10.2f}")
            x += steps_per_result

