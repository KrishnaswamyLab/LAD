from atongtf import dataset
from atongtf import image_transforms
import numpy as np
import matplotlib.pyplot as plt

d = dataset.Cifar_Anomaly_Dataset(5, 0.02, verbose=True)
imgs = (d.get_train()[10:12] + 1) / 2
simgs = image_transforms.shuffle_patches(imgs, 4)
fig, axes = plt.subplots(2,1)
axes[0].imshow(imgs[0].squeeze())
axes[1].imshow(simgs[0].squeeze())
plt.show()


