import numpy as np
from pathlib import Path
root = "/mnt/data1/shared/data/tanish-sph-v2/AP/train"
root = Path(root)

save_path = Path("./readable/AP/data_viz")

i = 0
for j in root.glob("*.npz"):
    x = np.load(j)
    x = np.concatenate([x["XYZ"], x["Voltage"]], axis=-1)
    np.random.shuffle(x)
    np.savez_compressed(save_path / j.name, XYZ=x[:16384, :3], Voltage=x[:16384, 3:4])
    # np.random.shuffle(x)
    # np.savez_compressed(save_path / "32k" / j.name, XYZ=x[:16384*2, :3], Voltage=x[:16384*2, 3:4])
    i += 1
    if i > 20:
        break


    
