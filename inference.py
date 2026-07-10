import os

import matplotlib.pyplot as plt
import torch

from dataset import generate_random_seed
from network import Generator

save_dir = "./imgs"
os.makedirs(save_dir, exist_ok=True)

G = Generator()
PATH_G = "trained_G.pth"
G.load_state_dict(torch.load(PATH_G, map_location=torch.device("cpu")))

with torch.no_grad():
    for i in range(2):
        output = G(generate_random_seed(100))

        img = output.detach().permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

        img = img.clip(0, 1)

        save_path = os.path.join(save_dir, f'generated_{i+1}.png')
        plt.imsave(save_path, img)

print("Done.")
