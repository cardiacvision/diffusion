# %%
import numpy as np
import matplotlib.pyplot as plt
from array2gif import write_gif
# %%

trajs = np.load("./diffusion_exp_diffusion_evolved_trajs.npy", allow_pickle=True)

# %%
simul_lens = np.load("./simul_exp_diffusion_evolved_lens.npy", allow_pickle=True)
plt.subplot(121)
plt.hist(simul_lens[simul_lens < 1500], density=True)
plt.title("PDE Initial Conditions")

diffusion_lens = np.load("./diffusion_exp_diffusion_evolved_lens.npy", allow_pickle=True)
plt.subplot(122)
plt.hist(diffusion_lens[diffusion_lens < 1500], density=True)
plt.title("Diffusion Initial Conditions")

plt.gcf().set_size_inches(10, 5)
# %%
trajs[0][0].shape
# %%
batch = trajs[3]
traj1 = []
for i in batch:
    traj1.append(i[20])

traj1 = np.concatenate(traj1)
traj1.shape

# %%
traj1 = traj1.reshape(-1, 64, 64, 1).repeat(3, -1).transpose(0, 3, 1, 2)
write_gif(traj1 * 255, "z_simul_test_traj.gif")
# plt.plot(range(500), (traj1 < 0.1).mean(-1).mean(-1).mean(-1)[:500])

# %%
plt.plot(range(150), (traj1 < 0.005).mean(-1).mean(-1).mean(-1)[:150])
# %%
(traj1 < 0.15).mean(-1).mean(-1).mean(-1)
# %%
new = traj1[40].copy()
new[new < 0.05] = 0
plt.imshow(new.transpose(1, 2, 0))

# %%
temp = np.array([0, 1, 2, 3])
idx, = np.where(temp > 2)
idx[0]
# %%

decay_times = []
for batch in trajs:
    for j in range(len(batch[0])):
        traj = []
        for i in batch:
            traj.append(i[j])
        traj = np.concatenate(traj)
        traj = traj.reshape(2500, 64, 64)
        percent_under_thresh = (traj < 0.1).mean(-1).mean(-1)
        decay_time = np.where(percent_under_thresh < 0.05)[0][0]
        decay_times.append(decay_time)

plt.hist(decay_times, bins=15)
# %%
len(decay_times)
# %%
batch[0].shape
# %%
traj.shape
# %%
