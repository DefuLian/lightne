import numpy as np
from netmf import approximate_normalized_graph_laplacian, approximate_deepwalk_matrix, direct_compute_deepwalk_matrix, load_adjacency_matrix
from scipy.sparse.linalg import svds
import argparse
import predict
import logging

def netmf(network, window, neg, rank):
    if window > 3:
        A = network
        vol = float(A.sum())
        # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
        # keep top #rank eigenpairs
        evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=rank, which="LA")
        # approximate deepwalk matrix
        deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU, window=window, vol=vol, b=neg)
    else:
        deepwalk_matrix = direct_compute_deepwalk_matrix(network, window=window, b=neg)
    return deepwalk_matrix


def high_order_network(network, config):
    return netmf(network, config.window, config.neg, config.rank)

def proj_stiefel_manifold(A):
    """ Project matrix A on stiefel manifold, where A is stored in two-dimensional array
    :param A: a matrix to be projected
    :return: a matrix on stiefel manifold but with minimal distance from A
    """
    N, K = np.shape(A)
    assert N >= K
    U, _, V = np.linalg.svd(A, full_matrices=False)
    return np.matmul(U, V)

def proj_hamming_balance(V):
    """ Project matrix V on balanced hamming space
    :param V: a matrix to be projected
    :return: a matrix on balanced hamming space
    """
    N, K = np.shape(V)
    multiplier = np.median(V, axis=0)
    return 2 * (V - multiplier > 0).astype(np.float32) - 1

def network_hashing(network, config):
    """ learning binary representation for network
    :param network: adjacency matrix of high-order network
    :param config:
    :return:
    """
    net = network.tocsr()
    net_t = network.T.tocsr()
    prev_loss = np.inf
    n = network.shape[0]
    dim = config.dim
    u, s, _ = svds(net, dim, return_singular_vectors="u")
    B = proj_hamming_balance(u * s)
    mtb = net_t @ B

    def select(x, ratio):
        k = x.shape[0]
        d = np.floor(k * ratio)
        ind = np.argpartition(x, -d)[-d:]
        sel = np.zeros((k,1), np.bool)
        sel[ind] = True
        return sel

    def random_selection(x, ratio):
        k = x.shape[0]
        sel = np.zeros((k,1), np.bool)
        d = np.floor(k * ratio)
        ind = np.random.choice(k, d)
        sel[ind] = True
        return sel

    def compute_loss(network, P, Q):
        val = network.multiply(network).sum() - 2 * ((P.T @ network) * Q.T).sum() + ((Q.T @ Q) * (P.T @ P)).sum()
        return val / 2

    for iter in range(config.max_iter):
        Q = proj_stiefel_manifold(mtb)
        if config.gamma > 0:
            B_h = np.sqrt(n) * proj_stiefel_manifold(B)
        else:
            B_h = np.zeros_like(B)
        curr_loss = compute_loss(net, B, Q) + config.gamma / 2 * np.sum((B - B_h)*(B - B_h))
        logging.info('%3d iteration, loss %.3f', iter, curr_loss)
        if abs(prev_loss - curr_loss) < 1e-6:
            break
        prev_loss = curr_loss
        if config.ratio < 1:
            s = select(np.sum(Q * mtb, axis=0), config.ratio)
            b1 = proj_hamming_balance(net * Q[:, s] + config.gamma * B_h[:, s])
            mtb[:, s] = mtb[:, s] + net_t @ (b1 - B[:, s])
            B[:, s] = b1
        else:
            B = proj_hamming_balance(net * Q + config.gamma * B_h)
            mtb = net_t * B
    return B



def network_quantization(network, config):
    """ learning composite embedding for network
    :param network:
    :param config:
    :return:
    """



def main():
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    parser = argparse.ArgumentParser(description="Memory and search efficient network embedding")
    parser.add_argument('-i', "--input", type=str, required=True,
                        help=".mat input file path")
    parser.add_argument('-nn', '--network-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument('-o',"--output", type=str, help="embedding output file path")

    parser.add_argument("--rank", default=256, type=int,
                        help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
                        help="dimension of embedding")
    parser.add_argument('-T', "--window", default=10, type=int, help="context window size")
    parser.add_argument('-b', "--neg", default=1.0, type=float, help="negative sampling")
    parser.add_argument('-ln', "--label-name", type=str, default='group', help='variable name of node label inside a .mat file.')
    parser.add_argument('--ratio', type=float, default=1, help='each time only ratio of subspaces are updated')
    parser.add_argument('--max-iter', type=int, default=50, help='the maximum number of iterations')
    parser.add_argument('--gamma', type=float, default=0, help='regularization coefficient for decorrelation')
    args = parser.parse_args()

    label = predict.load_label(file=args.input, variable_name=args.label_name)
    network = load_adjacency_matrix(args.input, variable_name=args.network_name)
    embedding = network_hashing(high_order_network(network, args), args)
    for tr in range(9,10):
        predict.predict_cv(embedding, label, train_ratio=0.1*tr, n_splits=10, random_state=10, C=1.)

if __name__ == "__main__":
    import sys
    sys.exit(main())