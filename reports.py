from plot_results import sliding_average
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_results(policy, env, seed, discounted=False, path='./baseline_results'):
    file_name = f"{path}/{policy}_{env}_{seed}{'_disc' if discounted else ''}.npy"

    try:
        data = np.load(file_name)
    except:
        data = None

    return data

def gen_detailed_report(
    policies=["TD3", "TD3-DEV", "DDQN", "DDQN-DEV"],
    envs=["LunarLander-v3", "LunarLanderContinuous-v3", "BipedalWalker-v3", "Hopper-v5", "Ant-v5"],
    seed=range(10),
    path='./baseline_results'
):
    print("Seed,                       Env,     Policy,     Ave,     Max,    (Ave),    (Max),  Count")

    for e in envs:
        for s in seed:
            for p in policies:
                data = read_results(p, e, s, discounted=False, path=path)
                disc_data = read_results(p, e, s, discounted=True, path=path)

                if data is None or disc_data is None:
                    continue

                ave = sliding_average(data, 10)
                ave_mean = np.mean(ave)
                ave_max = np.max(data)

                disc_ave = sliding_average(disc_data, 10)
                disc_ave_mean = np.mean(disc_ave)
                disc_ave_max = np.max(disc_data)
                
                print(f"{s:>4}, {e:>25}, {p:>10}, {ave_mean:>7.2f}, {ave_max:>7.2f}, {disc_ave_mean:>8.2f}, {disc_ave_max:>8.2f}, {len(data):>6}")
