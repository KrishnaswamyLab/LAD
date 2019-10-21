import networkx as nx
import numpy as np
import tensorflow as tf
import math
from functools import reduce
import scipy.linalg
import scipy.io

def create_ising_graph(height, width, name, **kwargs):
    if name == 'grid':
        return create_ising_lattice(height, width, periodic=False)
    if name == 'torus':
        return create_ising_lattice(height, width, periodic=True)
    if name == 'ones':
        return np.ones(height, width)
    if name == 'rings':
        return wrap_nx(nx.cycle_graph, height, width)
    if name == 'lines':
        return wrap_nx(nx.path_graph, height, width)
    if name == 'vlines':
        return create_vlines(height, width)
    if name == 'wishbone':
        return create_ising_wishbone(height, width, **kwargs)
    if name == 'star':
        return wrap_nx(nx.star_graph, height, width - 1) # Star graphs produce a graph with n+1 nodes
    if name == 'kite':
        assert width == 10
        return wrap_nx(nx.krackhardt_kite_graph, height, None)
    if name == 'tree':
        assert not (width & width + 1) # width is 2^k - 1 for some k
        return wrap_nx(lambda h: nx.balanced_tree(2, h), height, int(math.log(width + 1, 2))-1)
    raise RuntimeError('ising graph type not known: %s' % name)

def normalize(g):
    return nx.normalized_laplacian_matrix(nx.from_numpy_matrix(g)).todense().astype(np.float32)

def create_laplacian(height, width, name, ltype, **kwargs):
    f = None
    if ltype == 'unnormalized': f = create_unnormalized_laplacian
    if ltype == 'sym': f = create_symmetric_laplacian
    if ltype == 'rw' : f = create_rw_laplacian
    if f is None:
        raise RuntimeError('Unknown laplacian type: %s' % ltype)
    return f(height, width, name, **kwargs)

def create_unnormalized_laplacian(height, width, name, **kwargs):
    g = create_ising_graph(height, width, name, **kwargs)
    return nx.laplacian_matrix(nx.from_numpy_matrix(g)).todense()

def create_symmetric_laplacian(height, width, name, **kwargs):
    g = create_ising_graph(height, width, name, **kwargs)
    return nx.normalized_laplacian_matrix(nx.from_numpy_matrix(g)).todense()

def create_rw_laplacian(height, width, name, **kwargs):
    g = create_ising_graph(height, width, name, **kwargs)
    A = nx.adjacency_matrix(nx.from_numpy_matrix(g)).todense()
    D = np.diagflat(A.sum(axis=1))
    with np.errstate(divide='ignore'):
        inv_D = 1.0 / D
    inv_D[np.isinf(inv_D)] = 0
    L = D - A
    return inv_D @ L

def wrap_nx1d(nxf, n): 
    return nx.to_numpy_matrix(nxf(n))

def wrap_nx(nxf, k, m): 
    graphs = [nxf(m) for i in range(k)]
    return nx.to_numpy_matrix(reduce(nx.disjoint_union, graphs))

def create_vlines(h,w):
    assert h == 2
    G = nx.empty_graph(h*w)
    G.add_edges_from((v, v+w) for v in range(w))
    return nx.to_numpy_matrix(G)

def create_ising_lattice(height, width, periodic=False):
    return nx.to_numpy_matrix(nx.grid_2d_graph(height, width, periodic = periodic))

def create_ising_wishbone(h, w, **kwargs):
    """ Create wishbone-like graph with first 1/2 of nodes connected """
    assert h == 2 # Only works for 2 branches
    G = nx.empty_graph(h * w)
    n = w
    G.add_edges_from([(v, v+1) for v in range(n-1)])
    G.add_edges_from([(v, v+1) for v in range(n,2*n-1)])
    G.add_edges_from([(v, v+n) for v in range(n // 2)]) # Connect first half of nodes
    return nx.to_numpy_matrix(G)

def ising_loss_laplacian(laplacian, activations, fixed_values = None, weight = 1, eps = 1e-12, normalize=False):
    #print('WARNING SETTING NORMALIZE TO FALSE')
    #normalize=True
    np.set_printoptions(precision=3)
    L = tf.expand_dims(tf.constant(laplacian, dtype=tf.float32), 0)
    ds = tf.shape(activations)
    if fixed_values is None:
        f = activations
    else:
        #print(activations.shape, fixed_values.shape)
        f = tf.concat([activations, tf.multiply(tf.cast(fixed_values, tf.float32), 10.0)], axis = 1)
    L = tf.tile(L, [ds[0], 1, 1])  # [Batch x Width x Width]
    a = tf.expand_dims(f, -1) # [Batch x Width x 1]
    at = tf.expand_dims(f, 1) # [Batch x 1 x Width]
    numerator = tf.reshape(tf.matmul(tf.matmul(at, L), a), [-1, 1]) # [Batch x 1]
    if not normalize:
        return weight * tf.reduce_mean(numerator)
    denominator = tf.reduce_sum(tf.multiply(f, f), axis=1, keepdims=True) + eps # [Batch x 1]
    return weight * tf.reduce_mean(tf.divide(numerator, denominator))


def ising_loss(G, activations, fixed_values = None, weight = 1, eps = 1e-12):
    """ DEPRECATED
    Imposes a loss on the activations of a layer of activations.

    Let a = activations, then computes scalar value 
    weight * (max_eig_val(G) - (a^T G a) / (a^t a)
    This is related to the reyleigh quotient which ranges over the eigenvalues of G.
    This loss is then non-negative and scale-free with regard to linear activation scaling.

    Args:
        G: numpy 2d adjacency matrix [layer_width] x [layer_width] 
        activations: [Batch] x [layer_width]

    Returns:
        A tensorflow scalar loss
    """
    maxeig = np.max(np.linalg.eigvalsh(G))
    G = tf.expand_dims(tf.constant(G, dtype=tf.float32), 0)
    ds = tf.shape(activations)
    if fixed_values is None:
        f = activations
    else: 
        f = tf.concat([activations, fixed_values], axis = 1)
    G = tf.tile(G, [ds[0], 1, 1])  # [Batch x Width x Width]
    a = tf.expand_dims(f, -1) # [Batch x Width x 1]
    at = tf.expand_dims(f, 1) # [Batch x 1 x Width]
    denominator = tf.reduce_sum(tf.multiply(f, f), axis=1, keepdims=True) + eps # [Batch x 1]
    numerator = tf.reshape(tf.matmul(tf.matmul(at, G), a), [-1, 1]) # [Batch x 1]
    return weight * (maxeig - tf.reduce_mean(tf.divide(numerator, denominator)))

def compare_losses():
    batch, height, width = (2,2,5)

    shape = 'lines'
    #Jfor shape in ['lines', 'rings', 'grid', 'torus', 'star']:
    Lrw = create_laplacian(height, width, 'lines', 'rw')
    Lsym = create_laplacian(height, width, 'lines', 'sym')
    L = create_laplacian(height, width, 'lines', 'unnormalized')
    print(sorted(np.linalg.eigvals(Lrw)))
    print(sorted(np.linalg.eigvalsh(Lsym)))
    print(sorted(np.linalg.eigvalsh(L)))
    G = create_ising_graph(height, width, 'lines')
    #G = np.ones((height*width, height*width), dtype = np.float32)
    #L = create_laplacian(height, width, shape)
    #L = nx.normalized_laplacian_matrix(nx.from_numpy_matrix(G)).todense()
    activations = np.random.normal(size=(batch, height*width)).astype(np.float32)
    #activations = np.reshape(np.arange(batch*height*width, dtype=np.float32), (batch,height*width))
    l_loss = ising_loss_laplacian(L, activations)
    lrw_loss = ising_loss_laplacian(Lrw, activations)
    lsym_loss = ising_loss_laplacian(Lsym, activations)
    g_loss = ising_loss(G, activations)
    sess = tf.Session()
    print(shape, sess.run((l_loss, lrw_loss, lsym_loss, g_loss)))

def compare_regularizations():
    batch, height, width = (2,1,10)

    from pprint import pprint
    shape = 'rings'
    #Jfor shape in ['lines', 'rings', 'grid', 'torus', 'star']:
    Lrw = create_laplacian(height, width, 'lines', 'rw')
    Lsym = create_laplacian(height, width, 'lines', 'sym')
    L = create_laplacian(height, width, 'lines', 'unnormalized')
    print(L)
    import matplotlib.pyplot as plt
    eigvals, eigvecs = np.linalg.eigh(L)
    print(eigvals)
    print(type(eigvals))
    eigvecs = np.array(eigvecs)
    for t in range(10):
        plt.plot(range(10), eigvecs[:,t])
    #plt.show()
    #pprint(np.linalg.eigh(L))
    #pprint(sorted(np.linalg.eigvals(Lrw)))
    #Jpprint(sorted(np.linalg.eigvalsh(Lsym)))
    #pprint(sorted(np.linalg.eigvalsh(L)))

def create_heat_kernel(height, width, name, power = 1, subsample = 4, **kwargs):
    g = create_ising_graph(height, width, name, **kwargs)
    g = np.array(g) + np.identity(height*width)
    g /= scipy.linalg.norm(g,ord=1,axis=1)
    g = np.transpose(g)
    gg = np.identity(height*width)
    for i in range(power):
        gg = np.dot(gg, g)
    gg = gg[::subsample,:]
    return gg

def get_heat_kernels(height, width, name, power=1, subsample=4, **kwargs):
    k = create_heat_kernel(height, width, name, power, subsample, **kwargs)
    kinv = np.linalg.pinv(k)
    return k, kinv

def create_weighted_eigenvector_kernel(height, width, name, **kwargs):
    L = create_symmetric_laplacian(height, width, name, **kwargs)
    eigvals, eigvecs = np.linalg.eigh(L)
    return np.diag(np.exp(-eigvals)) @ eigvecs

def heat_loss(kernel, activations, eps = 1e-12):
    k = tf.expand_dims(tf.constant(kernel, dtype=tf.float32), 0)
    ds = tf.shape(activations)
    print(kernel.shape)
    k = tf.tile(k, [ds[0], 1, 1])  # [Batch x Width x Width]
    #k = tf.Print(k, [k])
    print(k.shape)
    a = tf.expand_dims(activations, -1)
    t = tf.reduce_sum(tf.abs(tf.matmul(k,a)))
    #t = tf.Print(t, [t])
    return t

def heat_loss2(kernel, activations, eps = 1e-12):
    k = tf.expand_dims(tf.constant(kernel, dtype=tf.float32), 0)
    ds = tf.shape(activations)
    k = tf.tile(k, [ds[0], 1, 1])  # [Batch x Width x Width]
    a = tf.expand_dims(activations, 1)
    t = 1 - tf.reduce_sum(tf.abs(k - a))
    return t

def test_create_heat_kernel():
    g = create_heat_kernel(1,5,"star")
    assert sum(scipy.linalg.norm(np.dot(g,g), ord=1, axis=1)) == 5

def plot_kernel_shape():
    import matplotlib.pyplot as plt
    k = create_heat_kernel(1,20,"rings", power=1)
    print(k)
    plt.plot(range(20), k[3])
    k = create_heat_kernel(1,20,"rings", power=2)
    print(k)
    plt.plot(range(20), k[3])
    k = create_heat_kernel(1,20,"rings", power=5)
    plt.plot(range(20), k[3])
    k = create_heat_kernel(1,20,"rings", power=7)
    plt.plot(range(20), k[3])
    k = create_heat_kernel(1,20,"rings", power=10)
    plt.plot(range(20), k[3])
    plt.show()
    print(k)
    #test_create_heat_kernel()

def get_kernels():
    import matplotlib.pyplot as plt
    k = create_heat_kernel(1,n,"rings", power=power, subsample=subsample)
    #print(np.linalg.det(k))
    kinv = np.linalg.pinv(k)
    #Jprint(np.sum(kpinv, axis=1))
    #print(np.sum(kpinv, axis=0))
    for i in range(n // subsample):
        plt.plot(range(n), k[i])
    #print(kpinv)
    #plt.plot(range(n), np.sum(k, axis=0))
    #plt.show()
    print(k,kinv)

    #plt.savefig('kernel_inv')
    return k, kinv

def test_heat_loss():
    k = create_heat_kernel(1,20,"rings", power=10)
    print(np.sum(k, axis=1))
    print(np.sum(k, axis=0))
    x = np.ones((40,20), dtype=np.float32) / 20
    hl = heat_loss(k,x)
    sess = tf.Session()
    print(sess.run(hl))

def load_kernel_mat(path, name):
    U = scipy.io.loadmat(path)
    print(U.keys())
    #for k,u in U.items():
    #    print(k, u.shape)
#    print(U.keys())
#    print(type(U))
#    print(np.sum(U[name], axis=1))
#    print(np.sum(U[name], axis=0))
    return U[name].astype(np.float32)

def test_load_kernel_mat():
    ainv = load_kernel_mat('abspline3.mat', 'backward')
    a = load_kernel_mat('abspline3.mat', 'forward')
    print(a.shape, ainv.shape)
    #load_kernel_mat('wavelets_full.mat', 'hat1backward')

def test_create_weighted_eigenvector_kernel():
    k = create_weighted_eigenvector_kernel(1, 4, "rings")
    print(np.dot(k, [1,0,0,0]))


if __name__ == '__main__':
    pass
    #a,ainv = get_heat_kernels(1,20,"rings", power=5, subsample=2)
    #print(a, ainv)
    #test_heat_loss()
    #plot_kernel_shape2()
    #create_weighted_eigenvector_kernel(1, 20, "rings")
    #test_create_weighted_eigenvector_kernel()
    test_load_kernel_mat()
