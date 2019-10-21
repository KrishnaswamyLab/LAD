import numpy as np

def shuffle_patches(images, patch_size):
    dims = images.shape[1:] # (X, 28, 28, 1) for mnist
    patches = []
    for i in range(0, dims[0] // patch_size):
        for j in range(0, dims[1] // patch_size):
            big_i = i * patch_size
            big_j = j * patch_size
            patches.append(images[:, big_i:big_i+patch_size, big_j:big_j+patch_size])
    # (patchs, num_images, height, width, depth)
    patches = np.array(patches)
    np.random.shuffle(patches)

    imgs = np.zeros_like(images)
    for i in range(0, dims[0] // patch_size):
        for j in range(0, dims[1] // patch_size):
            big_i = i * patch_size
            big_j = j * patch_size
            imgs[:, big_i:big_i+patch_size, big_j:big_j+patch_size] = patches[i * dims[0] // patch_size + j]
    return imgs
