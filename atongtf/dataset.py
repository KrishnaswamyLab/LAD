import tensorflow.keras.datasets as kd
import pandas as pd
import matplotlib.pyplot as plt
from . import util
import numpy as np
import graphtools
import time
import pygsp
import scipy
from sklearn.decomposition import PCA
import phate

def plot_pca(data, labels=None):
    pca = PCA(2)
    embed = pca.fit_transform(data)
    plt.scatter(embed[:,0], embed[:,1], c=labels)
    plt.show()

def keras_image_format_to_std(data):
    """ converts 0-255 int range data to -1 to 1 float data """
    data = (data.astype(np.float32) - 127.5) / 127.5
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=3)
    return data

def rotate_to_higher_dimensions(data, dim):
    """ Rotates a dataset into higher dimensions.
    Assumes that data is a (examples, features) 2d numpy array.
    Args:
    """
    assert len(data.shape) == 2
    n, m = data.shape
    # Select m random orthoganol vectors in new space
    random_basis = scipy.stats.special_ortho_group.rvs(dim)[:m,:]
    # Rotate data into new space (examples, new_features)
    return data @ random_basis

class Dataset(object):
    def __init__(self):
        self.embed_dir = './embeddings'
    def get_train(self): raise NotImplementedError
    def get_test(self): raise NotImplementedError
    def get_shape(self): raise NotImplementedError 
    def is_image_type(self): return False
    def get_embedding(self): raise NotImplementedError
    def factory(name):
        if name == 'mnist': return Keras_Dataset(kd.mnist, name)
        if name == 'cifar10': return Keras_Dataset(kd.cifar10, name)
        if name == 'cifar_ship_deer': return Keras_Dataset(kd.cifar10, name, label_subset = [4,8])
        if name == 'cifar_dog_cat': return Keras_Anomaly(kd.cifar10, name)
        if name == 'fashion_mnist': return Keras_Dataset(kd.fashion_mnist, name)
        if name == 'fashion_mnist_shirt_boot': return Keras_Dataset(kd.fashion_mnist, label_subset = [6,9], name=name)
        if name == 'cifar100': return Keras_Dataset(kd.cifar100, name)
        if name == 'mnist5': return Mnist_Fives_Dataset()
        if name == 'mnist7': return Mnist_Digit_Dataset(7, 'sevens')
        if name == 'mnist5_small7': return Mnist_Fives_Small_Sevens_Dataset()
        if name == 'mnist05': return Mnist_Digit_Dataset([0,5], 'zeros_and_fives')
        if name == 'mnist45': return Mnist_Digit_Dataset([4,5], 'fours_and_fives')
        raise AssertionError('Bad Dataset Creation: %s' % name)
    factory = staticmethod(factory)


class Generated_Dataset(Dataset):
    def __init__(self, random_state = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.data, self.labels = self.generate()
    def generate(self): raise NotImplementedError
    def plot(self): raise NotImplementedError
    def get_shape(self): return self.data.shape[1:]
    def get_train(self): return self.data
    def get_test(self): return self.data
    def get_train_labels(self): return self.labels

class Branch_Dataset(Generated_Dataset):
    def __init__(self, random_state = 42, noise=1.0):
        self.noise = noise
        super().__init__()

    def generate(self):
        n = 1000
        d = 2
        s = self.noise
        d1 = np.random.randn(n,d)*s
        d2 = np.random.randn(n,d)*s + [2,0]
        d3 = np.concatenate([np.random.randn(n//2,2)*s + [4,2],
                             np.random.randn(n//2,2)*s + [4,-2]])
        d1 = np.random.randn(n,d)
        d2 = np.random.randn(n,d) + [2,0]
        d3 = np.concatenate([np.random.randn(n//2,2) + [4,2],
                         np.random.randn(n//2,2) + [4,-2]])
        data = np.concatenate([d1,d2,d3])
        data = data / 2
        labels = np.repeat(np.arange(3), n)
        return data, labels

    def plot(self):
        plt.scatter(self.data[:,0], self.data[:,1], c=self.labels)
        plt.show()


class Hub_Dataset(Generated_Dataset):
    """ Generate a hub of 5 genes """
    def __init__(self, modules, random_state = 42):
        assert modules <= 8
        self.modules = modules
        super().__init__(random_state)
        print(self.data.shape, self.labels.shape)

    def generate(self):
        y = np.arange(2 ** self.modules, dtype=np.uint8).reshape(-1, 1)
        x = np.unpackbits(y, axis=1).astype(np.float32)
        x = np.repeat(x, 50, axis=1)
        x = np.repeat(x, 1000, axis=0)
        x = x + np.random.normal(0, 0.1, size=x.shape)
        x *=10
        y = np.repeat(y.flatten(), 1000)
        return x, y

    def plot(self):
        plot_pca(self.data, self.labels)
    
    def plot_heatmap(self):
        import seaborn as sns
        sns.heatmap(self.data[::1000],
                xticklabels = False,
                yticklabels = False,
                cbar = False,
                #cmap = "RdBu_r",
                #center = 0,

        )
        #plt.title('%d Modules' % self.modules)
        plt.savefig('%d_modules.png' % self.modules)
        plt.close()
 
class Gaussian_Dataset(Generated_Dataset):
    """ Parameterized gaussian blobs
    """
    def __init__(self, n = 100, densities = None, dimensions=2):
        self.n = n
        self.densities = densities
        if densities is None:
            self.densities = [1,3,10]
        self.dimensions = dimensions
        self.data, self.labels = self.generate()
    def generate(self):
        data = []
        labels = []
        for i, density in enumerate(self.densities):
            data.append(np.random.multivariate_normal([5*i,5*i],np.eye(2), size=density*self.n))
            labels.append(i*np.ones(density*self.n))
        data = np.concatenate(data)
        data = data - np.mean(data, axis=0)

        return data, np.concatenate(labels)
        #y = np.sort(np.random.uniform(0, self.scale, self.n))
        #self.random_dir = scipy.stats.special_ortho_group.rvs(self.dims)[:,0]
        #x = np.random.normal(np.outer(np.ones(self.n) * y, self.random_dir), scale= self.noise / (self.dims ** 2))
    def plot(self):
        plt.scatter(self.data[:,0], self.data[:,1], s=1, c=self.labels)
        plt.show()

class Gaussian_Mixture_Dataset(Generated_Dataset):
    """ Parameterized series of smaller and smaller gaussians
    """
    def __init__(self, n = 100, densities = None, dimensions=2):
        self.n = n
        self.densities = densities
        if densities is None:
            self.densities = [1,3,10]
        self.dimensions = dimensions
        self.data, self.labels = self.generate()
    def generate(self):
        data = []
        labels = []
        for i, density in enumerate(self.densities):
            data.append(np.random.multivariate_normal([5*i, 5*i],np.eye(2), size=density*self.n))
            labels.append(i*np.ones(density*self.n))
        data = np.concatenate(data)
        data = data - np.mean(data, axis=0)
        data = data / np.var(data, axis=0)

        return data, np.concatenate(labels)
        #y = np.sort(np.random.uniform(0, self.scale, self.n))
        #self.random_dir = scipy.stats.special_ortho_group.rvs(self.dims)[:,0]
        #x = np.random.normal(np.outer(np.ones(self.n) * y, self.random_dir), scale= self.noise / (self.dims ** 2))
    def plot(self):
        plt.scatter(self.data[:,0], self.data[:,1], s=1, c=self.labels)
        plt.show()

class Circle_Dataset(Generated_Dataset):
    """ A Circle where at each point the conditional distribution is 50% in each direction
    """
    def __init__(self, n=100):
        self.n = n
        super().__init__()

    def transform(self, arr, radius = 1):
        """ takes 1-d array and transforms to circular coordinates
        """
        while sum(arr < 0) > 0:
            arr[arr < 0] += 1
        while sum(arr > 1) > 0:
            arr[arr > 1] -= 1
        assert max(arr) <= 1
        assert min(arr) >= 0
        arr = 2 * np.pi * arr
        y = np.sin(arr) * radius
        x = np.cos(arr) * radius
        return np.concatenate([x,y], axis=1)

    def generate(self):
        x = np.random.rand(self.n, 1)
        direction = np.random.choice([-0.04, 0.04], size=(self.n, 1))
        noise = np.random.randn(self.n, 1) * 0.02
        y = x + direction + noise
        x = self.transform(x, 1)
        y = self.transform(y, 1)
        return x,y

    def plot(self):
        plt.scatter(self.data[:,0], self.data[:,1], s=1)
        plt.show()
    
    def plot_arrows(self):
        n = 100
        y, x = self.data[:n], self.labels[:n]
        plt.quiver(x[:,0], x[:,1], (y-x)[:,0], (y-x)[:,1], angles='xy', scale_units='xy', scale=1)
        plt.show()


class Circle_Transition_Dataset(Circle_Dataset):
    def generate(self):
        step = 0.04
        noise = 0.02
        x = np.random.rand(self.n, 1)
        y = x + step + np.random.randn(self.n,1) * noise
        x = self.transform(x)
        y = self.transform(y)
        # n must be even
        assert (self.n // 2) * 2 == self.n
        labels = np.concatenate([np.zeros(self.n//2), np.ones(self.n//2)]).astype(np.int32)
        return np.concatenate([x,y], axis=1), labels


class Keras_Dataset(Dataset):
    def __init__(self, data_loader, name, label_subset=None, verbose=False):
        super().__init__()
        self.data = data_loader.load_data()
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.data
        self.embed_name = ('%s_phate' % name)
        self.name = name
        self.label_subset = np.unique(self.train_labels)
        self.verbose = verbose
        if label_subset is not None:
            self.subset_labels(label_subset)

    def flatten(self):
        n = np.prod(dataset.get_shape())
        self.train_images = self.train_images.reshape(-1, n)
        self.test_images = self.test_images.reshape(-1, n)
        return self

    def scramble_features(self):
        n = np.prod(self.get_shape())
        permutation = np.random.permutation(n)
        def permute_images(imgs):
            perm_matrix = np.zeros((n,n))
            for i in range(n):
                perm_matrix[i, permutation[i]] = 1
            return imgs @ perm_matrix
        self.train_images = permute_images(self.train_images.reshape(-1, n))
        self.test_images = permute_images(self.test_images.reshape(-1, n))
        return self

    def subset_labels(self, label_subset): 
        if not (isinstance(label_subset, list) or isinstance(label_subset, tuple)):
            label_subset = (label_subset,)
        def filter_labels(labels, cl):
            mask = np.zeros_like(labels, dtype=np.bool)
            for c in cl:
                mask |= labels == c
            return mask.flatten()
        train_idx = filter_labels(self.train_labels, label_subset)
        test_idx  = filter_labels(self.test_labels, label_subset)
        self.train_images = self.train_images[train_idx]
        self.train_labels = self.train_labels[train_idx]
        self.test_images = self.test_images[test_idx]
        self.test_labels = self.test_labels[test_idx]
        self.label_subset = label_subset
        if self.verbose:
            print('%s dataset has %d training images and %d test images' %
                  (self.name, len(self.train_labels), len(self.test_labels)))

    def get_train(self):
        return keras_image_format_to_std(self.train_images)
    def get_test(self):
        return keras_image_format_to_std(self.test_images)
    def get_shape(self):
        return keras_image_format_to_std(self.test_images).shape[1:]
    def is_image_type(self):
        return True
    def get_train_labels(self):
        return self.train_labels
    def get_test_labels(self):
        return self.test_labels
    def get_embedding(self):
        def do_phate():
            import phate
            phate_op = phate.PHATE(k=5, a=None, t=52, n_jobs=-2)
            #phate_op = phate.PHATE(k=5, a=None, t=105, n_jobs=-2)
            return phate_op.fit_transform(np.reshape(self.train_images, (-1, 28*28)))
        return util.npdo(do_phate, self.embed_dir + '%s/%s.npy' %(self.embed_dir, self.embed_name))
    def plot_embed(self, labels = None, title=None, fig = None, ax = None, set_legend=True, vmin=None, vmax=None):
        if labels is None: labels = self.train_labels.astype(np.int32)
        sc = util.scatter_plot2d(self.get_embedding(), 
                                 labels, 'PHATE k=5 a=None, t=52' if title is None else title, 
                                 set_legend=set_legend, fig=fig, ax = ax, vmin=vmin, vmax=vmax)
        if fig is None: util.save_show(self.embed_dir, self.embed_name if title is None else title)
        return sc
    def subset(self, num):
        self.train_images = self.train_images[:num]
        self.train_labels = self.train_labels[:num]
        self.test_images = self.test_images[:num]
        self.test_labels = self.test_labels[:num]

class Keras_Anomaly_old(Keras_Dataset):
    def __init__(self, data_loader, name, norm_class=5, anomaly_class=3, num_normal=5000, num_anomaly=50):
        super().__init__(data_loader, label_subset=[norm_class, anomaly_class], name='dog_small_cat')
        d_normal = Keras_Dataset(data_loader, 'normal', label_subset=[5])
        d_normal.subset(num_normal)
        d_anomaly = Keras_Dataset(data_loader, 'anomaly', label_subset=[3])
        d_anomaly.subset(num_anomaly)
        print(d_anomaly.get_train().shape)
        self.train_labels = np.concatenate([d_normal.train_labels, d_anomaly.train_labels])
        self.train_images = np.concatenate([d_normal.train_images, d_anomaly.train_images])
        self.test_labels = np.concatenate([d_normal.test_labels, d_anomaly.test_labels])
        self.test_images = np.concatenate([d_normal.test_images, d_anomaly.test_images])
        print('%s dataset has %d training images and %d test images' %
                (self.name, len(self.train_labels), len(self.test_labels)))

class Mnist_Dataset(Keras_Dataset):
    def __init__(self):
        super().__init__(kd.mnist, 'mnist')
    def build_graph(self):
        # TODO incomplete
        start = time.time()
        G = graphtools.Graph(np.reshape(self.train_images, (-1, 28*28))[:2000],
                             n_pca=100,
                             n_jobs = -2,
                             use_pygsp=True,
                            )
        # G.diff_op is sparse, make it dense before matrix power
        #print(np.linalg.matrix_power(G.diff_op.toarray(), 5))
        #print(pygsp.utils.resistance_distance(G))
        end = time.time()
        print('time in seconds: %0.4f' % (end - start))


class Flat_Mnist_Dataset(Mnist_Dataset):
    def __init__(self):
        n = 28*28
        super().__init__()
        self.train_images = self.train_images.reshape(-1, n)
        self.test_images = self.test_images.reshape(-1, n)

class Rand_Mnist_Dataset(Mnist_Dataset):
    def __init__(self):
        n = 28*28
        permutation = np.random.permutation(n)
        super().__init__()
        def permute_images(imgs):
            perm_matrix = np.zeros((n,n))
            for i in range(n):
                perm_matrix[i, permutation[i]] = 1
            return imgs @ perm_matrix
        self.train_images = permute_images(self.train_images.reshape(-1, n))
        self.test_images = permute_images(self.test_images.reshape(-1, n))
        #(self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.data

class Mnist_Digit_Dataset(Mnist_Dataset):
    def __init__(self, label_subset, subset_name = 'digit', verbose=False):
        """ Initializes Mnist with subset of digits.
        Takes a digit or list of digits to subset.
        """
        super().__init__()
        self.subset_name = subset_name
        if not (isinstance(label_subset, list) or isinstance(label_subset, tuple)):
            label_subset = (label_subset,)
        def filter_labels(labels, cl):
            mask = np.zeros_like(labels, dtype=np.bool)
            for c in cl:
                mask |= labels == c
            return mask

        train_idx = filter_labels(self.train_labels, label_subset)
        test_idx  = filter_labels(self.test_labels, label_subset)
        self.train_images = self.train_images[train_idx]
        self.train_labels = self.train_labels[train_idx]
        self.test_images = self.test_images[test_idx]
        self.test_labels = self.test_labels[test_idx]
        self.embed_name = 'mnist_%s_phate' % self.subset_name
        self.label_subset = label_subset
        if verbose:
            print('Mnist %s dataset has %d training images and %d test images' %
                    (self.subset_name, len(self.train_labels), len(self.test_labels)))

class Mnist_Fives_Dataset(Mnist_Digit_Dataset):
    def __init__(self):
        super().__init__(5, subset_name = 'fives')

class Mnist_Fives_Small_Sevens_Dataset(Mnist_Digit_Dataset):
    def __init__(self, num_fives=5000, num_sevens=100):
        super().__init__([5,7], subset_name='five_small_seven')
        self.num_fives = num_fives
        self.num_sevens = num_sevens
        d_fives = Mnist_Digit_Dataset(5)
        d_fives.subset(num_fives)
        d_sevens = Mnist_Digit_Dataset(7)
        d_sevens.subset(num_sevens)
        self.train_labels = np.concatenate([d_fives.train_labels, d_sevens.train_labels])
        self.train_images = np.concatenate([d_fives.train_images, d_sevens.train_images])
        self.test_labels = np.concatenate([d_fives.test_labels, d_sevens.test_labels])
        self.test_images = np.concatenate([d_fives.test_images, d_sevens.test_images])
        print('Mnist %s dataset has %d training images and %d test images' %
                (self.subset_name, len(self.train_labels), len(self.test_labels)))


class Keras_Anomaly_Dataset(Keras_Dataset):
    """ Anomaly dataset as defined in protocol 2 of Perera et al. 2019 (OCGAN). 
    
    Full train data, then a random corruption of other digits with a given percentage of corruption. Test set is the full test set.
    """
    def __init__(self, data_loader, name, norm_class, frac_out, verbose=False):
        assert frac_out >= 0 and frac_out < 1
        super().__init__(data_loader, name)
        d_full = Keras_Dataset(data_loader, 'all')
        d_in = Keras_Dataset(data_loader, 'norm', label_subset=norm_class)
        out = list(range(10))
        out.remove(norm_class)
        d_out = Keras_Dataset(data_loader, 'anomaly', label_subset=out)
        n_out = int(frac_out / (1 - frac_out) * d_in.get_train().shape[0])
        d_out.subset(n_out)
        self.train_labels = np.concatenate([d_in.train_labels, d_out.train_labels])
        self.train_images = np.concatenate([d_in.train_images, d_out.train_images])
        self.test_labels = d_full.test_labels
        self.test_images = d_full.test_images
        if verbose:
            print('%s Dataset has %d train with %d corruption and %d test' % (name, len(self.train_labels), n_out, len(self.test_labels)))

class Cifar_Anomaly_Dataset(Keras_Anomaly_Dataset):
    def __init__(self, digit, frac_out, verbose=False):
        super().__init__(kd.cifar10, 'cifar_anomaly', digit, frac_out, verbose)

class Mnist_Anomaly_Dataset(Keras_Anomaly_Dataset):
    def __init__(self, digit, frac_out, verbose=False):
        super().__init__(kd.mnist, 'mnist_anomaly', digit, frac_out, verbose)

class Mnist_Plus_Black_Dataset(Mnist_Dataset):
    """
    Standard mnist, however test dataset contains all true images and black image
    """
    def __init__(self):
        super().__init__()
        self.black_image = np.zeros((1,28,28))
        self.test_labels = np.concatenate([self.test_labels, np.array([-1])])
        self.test_images = np.concatenate([self.test_images, self.black_image])

class Wishbone_Dataset(Dataset):
    def __init__(self, even_branches=False):
        import fcsparser
        import pandas as pd
        super().__init__()
        self.data_dir = '/home/atong/data/wishbone_thymus_panel1_rep1.fcs'
        _, self.full_data = fcsparser.parse(self.data_dir)
        if even_branches:
            d = []
            for b,g in self.full_data.groupby('Branch'):
                d.append(g.sample(8000))
            self.full_data = pd.concat(d)
        self.data = self.filter_data(self.full_data)
        #self.test_split = (self.data.shape[0] * 4) // 5
        #self.train_data = self.data.iloc[self.test_split:]
        #self.test_data  = self.data.iloc[:self.test_split]
        self.branch_labels = self.full_data['Branch']
        self.trajectory_labels = self.full_data['Trajectory']
    def filter_data(self, data):
        # Filer for important cells
        data = data[np.concatenate([data.columns[:13], data.columns[14:21]])]
        data = np.arcsinh(data / 5)
        return data
    def get_train(self):
        return np.array(self.data)
    def get_train_labels(self):
        return self.branch_labels
    def get_shape(self):
        return [self.data.shape[1]]

class EB_Velocity_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_file = '/home/atong/data/eb_velocity.npz'
        self.data = np.load(self.data_file)
        self.color = self.data['color']
        self.delta = self.data['delta_embedding']
        self.emb = self.data['embedding']
        self.emb_min, self.emb_max = np.min(self.emb, axis=0), np.max(self.emb, axis=0)
        self.del_min, self.del_max = np.min(self.delta, axis=0), np.max(self.delta, axis=0)

        # Normalize (min, max) = (0,1)
        self.train_labels = (self.emb - self.emb_min) / (self.emb_max - self.emb_min)
        self.train_data = (self.delta - self.del_min) / (self.del_max - self.del_min)

    def get_train(self): return self.train_data
    def get_train_labels(self): return self.train_labels
    def get_shape(self): return [self.train_data.shape[1]]

class EB_Velocity_Transition_Dataset(EB_Velocity_Dataset):
    def __init__(self):
        super().__init__()
        self.ixs = self.data['ixs']
        self.tr = self.data['tr']
        self.train_data = self.train_data[self.ixs,:]
        self.train_labels = self.train_labels[self.ixs,:]
        self.train_transitions = self.tr
    def get_transitions(self): return self.train_transitions

    def get_train_batch(self, idxs):
        trans_probs = self.train_transitions[idxs]
        nexts = []
        for i, id in enumerate(idxs):
            nex = np.random.multinomial(1, trans_probs[i]).astype(np.bool)
            nexts.append(self.train_data[nex])
        return np.array(nexts).squeeze()

class VACS_Vector_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_file = '/home/atong/data/VACS_phate_vecs.pkl'
        self.data = np.array(pd.read_pickle(self.data_file))
        # Set NAN to zero
        self.data[np.isnan(self.data)] = 0
        # Standardize by dividing by variance
        self.var = np.var(self.data, axis=0)[:2]
        self.mean = np.mean(self.data, axis=0)[:2]
        self.split = self.data.shape[0] // 10 * 8 # 80% 20% train test split
        self.train_data = (self.data[:self.split, :2] - self.mean) / self.var
        self.train_labels = self.train_data + self.data[:self.split, 2:] / self.var
        self.test_data = (self.data[self.split:, :2] - self.mean) / self.var
        self.test_labels = self.test_data + self.data[self.split:, 2:] / self.var
        print(var, np.sum(np.isnan(self.data[:self.split,2:])))
        print(np.sum(np.isnan(self.train_data)), np.sum(np.isnan(self.train_labels)))
        
    def get_train(self): return self.train_data
    def get_train_labels(self): return self.train_labels
    def get_test(self): return self.test_data
    def get_test_labels(self): return self.test_labels
    def get_shape(self): return [self.train_data.shape[1]]
    def get_mean_var(self): return self.mean, self.var

class VACS_Dataset(Dataset):
    def __init__(self, frac_out=0.):
        super().__init__()
        self.data_file = '/home/atong/data/VACS_normalized.pkl'
        self.data = pd.read_pickle(self.data_file)
        self.split = self.data.shape[0] // 10 * 8 # 80% 20% train test split
        self.train_data = self.data[:self.split]
        self.test_data = self.data[self.split:]
        self.creat_split = self.data.CREAT.quantile(0.95)
        nominal_train = self.train_data[self.train_data.CREAT <= self.creat_split]
        abnormal_train = self.train_data[self.train_data.CREAT > self.creat_split]
        n_out = int(frac_out / (1 - frac_out) * nominal_train.shape[0])
        self.remixed_train = pd.concat([nominal_train, abnormal_train[:n_out]])
    def get_train(self):
        return np.array(self.remixed_train)
    def get_train_labels(self):
        return np.array(self.nominal_train.CREAT > self.creat_split)
    def get_test(self):
        return np.array(self.test_data)
    def get_test_labels(self):
        return np.array(self.test_data.CREAT > self.creat_split)
    def get_shape(self):
        return [self.data.shape[1]]


class Hub_Spoke(Dataset):
    """ Generate a hub of 5 genes """
    def __init__(self, n = 60000, dims = 15, noise = 1.0, random_state = 42, perturb=False):
        self.perturb=perturb
        self.dims=dims
        self.noise=noise
        self.random_state=random_state
        self.n = n
        super().__init__()
        self.data, self.labels = self.generate_full()
    def generate_full(self):
        np.random.seed(self.random_state)
        num_modules = 3
        sub_per_module = 2
        m = num_genes_per_module = self.dims // num_modules
        self.dims = num_modules * m
        pairwise_correlation = 0.6
        sigma = np.ones((m, m)) * pairwise_correlation
        for i in range(m):
            sigma[i,i] = 1
        sigma = scipy.linalg.block_diag(*([sigma] * num_modules))
        x = np.random.multivariate_normal(np.zeros(self.dims), self.noise * sigma, self.n)

        """
        y = np.random.binomial(1, 0.5, size=(self.n, num_modules))
        active = np.zeros((self.n, self.dims))
        for ind,i in enumerate(y):
            active[ind] = np.repeat(i, num_genes_per_module) * 10
        print(active[:10, :])
        """
        if self.perturb:
            y = np.random.choice(num_modules * sub_per_module, size=self.n)
            active = np.zeros((self.n, num_modules))
            for i,yi in enumerate(y):
                active[i, yi // sub_per_module] = 10
            active = np.repeat(active, m, axis=1)
            for i,yi in enumerate(y):
                yii = m*(yi // sub_per_module) + (m-1)
                if  yi % sub_per_module == 1:
                    active[i, yii] += 10
                if yi % sub_per_module == 2:
                    active[i, yii - 1] += 10
        else:
            y = np.random.choice(num_modules, size=self.n)
            active = np.zeros((self.n, num_modules))
            for i,yi in enumerate(y):
                active[i, yi] = 10
            active = np.repeat(active, m, axis=1)
        x = x + active
        print(np.unique(y))
        return x,y

    def predict(self, preds):
        savedir = self.flags.model_dir
        #util.plot_heatmap_d3d_cat(preds, self.y, savedir=savedir, title_suffix='_y')
        util.plot_heatmap_d3d_cat(preds, self.y, savedir=savedir, title_suffix='_y_abs', abs_val = True)
        self.plot_output(preds, savedir=savedir)
        self.plot_output_pca(preds, savedir=savedir)
        #util.plot_heatmap_d3d_cat(preds, self.y, savedir=savedir, title_suffix='y1')
        #util.plot_heatmap_d3d_cat(preds, self.y, savedir=savedir, title_suffix='y2')
        """
        util.plot_heatmap_d3d(preds, self.x[:,0], savedir=savedir)
        util.plot_heatmap_d3d(preds, self.x[:,5], savedir=savedir, title_suffix='y5')
        util.plot_heatmap_d3d(preds, self.x[:,10], savedir=savedir, title_suffix='y10')
        """
    def plot_pca(self):
        pca = PCA(2)
        self.pca = pca.fit_transform(self.data)
        print(np.unique(self.labels))
        cmap = plt.get_cmap('tab10')
        print(self.pca)
        for i in np.unique(self.labels):
            d = self.pca[self.labels == i]
            plt.scatter(d[:,0], d[:,1], c = cmap(i), label=i, s=1)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.xticks([])
            plt.yticks([])
        plt.show()
    def plot_output_pca(self, preds, savedir=None):
        y = self.y
        if len(y.shape) > 1:
            y = y[:,0]
        ins = preds['input']
        outs = preds['prediction']
        fig = plt.figure(figsize = (10,5))
        ax0 = fig.add_subplot(1, 2, 1)#, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2)#, projection='3d')
        print(ins.shape)
        pins = PCA(ins, n_components=2)
        pouts = PCA(outs, n_components=2)
        phate.plot.scatter(pins[:,0], pins[:,1],  \
                c=y, ax=ax0, xlabel = 'PCA 1', ylabel = 'PCA 2', legend=False)
        phate.plot.scatter(pouts[:,0], pouts[:,1],  \
                c=y, ax=ax1, xlabel = 'PCA 1', ylabel = 'PCA 2', legend=False)
        util.save_show(savedir, 'output_pca')

class Mouse(Dataset):
    """
    Some interesting Genes pax6, tubb3
    """
    def __init__(self):
        self.data = pd.read_pickle('/home/atong/data/genemania/noonan3.gzip')

        self.embedding = np.load('/home/atong/data/genemania/noonan_phate.npy')
        self.markers = self.get_markers()
        self.pca, self.means, self.pca_components, _ = self.get_pca(1000)

    def get_pca(self, n_components):
        pca = PCA(n_components, random_state = 42)
        embedding = pca.fit_transform(self.get_train())
        print(np.sum(pca.explained_variance_ratio_))
        return embedding, pca.mean_, pca.components_, pca.explained_variance_ratio_

    def get_markers(self):
        def load_markers(fname):
            return np.loadtxt(fname, skiprows=1, delimiter=',', dtype=np.unicode)
        bm = load_markers('/home/atong/datasets/noonan/brain_markers.csv')
        fm   = load_markers('/home/atong/datasets/noonan/filter_cells.csv')
        from collections import defaultdict
        d = defaultdict(list)
        for row in bm:
            d[row[2]].append(row[1])
        for row in fm:
            d[row[2]].append(row[0])
        return d

    def plot_phate(self):
        phate.plot.scatter2d(self.embedding, c = self.data['Pax6'])
        plt.show()
    
    def get_shape(self, use_pca=False): 
        if use_pca:
            return [self.pca.shape[1]]
        return (self.data.shape[1],)
    def get_train(self, use_pca=False): 
        if use_pca:
            return self.pca
        return np.array(self.data)
    def plot_pca(self):
        plot_pca(self.data, self.data['Pax6'])
    def get_time(self):
        pca = PCA(10)
        embed = pca.fit_transform(self.get_train())
        #print(pca.explained_variance_ratio_)
        return embed

class TFDS(Dataset):
    def __init__(self, name):
        import tensorflow_datasets as tfds
        data, info = tfds.load(name, with_info=True)
        #data = tfds.as_numpy(data)
        import time
        iterator = data['train'].batch(128).make_one_shot_iterator()
        start = time.time()
        next_element = iterator.get_next()
        end = time.time()
        print(next_element)
        print(end-start)

class Dsprites_Dataset(Dataset):
    def __init__(self):
        self.path = '/home/atong/data/dsprites.npz'
        self.data = np.load(self.path, encoding='bytes')
        self.images = self.data['imgs'].astype(np.float32)
        # add Channels dimension
        self.images = np.expand_dims(self.images, -1)
        self.meta = self.data['metadata'][()]
        self.labels = self.data['latents_values']
        #self.latents_names = self.meta['latents_names']

    def get_shape(self):
        return self.images.shape[1:]
    
    def get_train(self):
        return self.images

    def get_train_labels(self):
        return self.labels


if __name__ == '__main__':
    #d = Keras_Dataset(kd.cifar10, name='cifar10')
    #d = Mnist_Fives_Small_Sevens_Dataset()
    #d = Mnist_Dataset()
    #l = d.get_train_labels()
    #np.save('mnist_train_labels', l)
    #print(l.shape)
    d = Wishbone_Dataset()
