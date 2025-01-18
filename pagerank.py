import sys
import numpy as np
import scipy as sp
import networkx as nx
from sklearn.preprocessing import normalize
from pagerank_utils import *

usage_str = """Usage: python3 pagerank.py filename tol
  filename:   (string with extension .adjlist) name of a file in adjacency-list format
              (see https://networkx.org/documentation/stable/reference/readwrite/adjlist.html)
       tol:   (positive integer) exponent for the error tolerance: 10 ** (-tol)
"""

print("Processing command line arguments...")
try:
    graph_file = sys.argv[1]
    file_name, ext = graph_file.rsplit(sep='.', maxsplit=1)
    assert(ext == "adjlist")
    mat_file = file_name + ".npz"
    rank_file = file_name + ".npy"

    tol = int(sys.argv[2])
    assert(tol > 0)
    error_tol = 10 ** (-1 * tol)
except:
    print("    | Bad command line arguments...")
    print(usage_str)
    exit(1)

print(f'Running PageRank on \'{graph_file}\'...')
print(f'    | error tolerance: {error_tol}')

mat_done = False
try:
    print(f'Loading \'{mat_file}\'...')
    a = sp.sparse.load_npz(mat_file)
    mat_done = True
except:
    print(f'Cannot load \'{mat_file}\'...')

if not mat_done:
    print(f'Creating networkx graph from \'{graph_file}\'...')
    try:
        g = nx.read_adjlist(graph_file, create_using=nx.DiGraph)
    except:
        print(f'Something went wrong reading \'{graph_file}\', check it is in your directory')
        exit(1)

    print(f'    | number of nodes: {nx.number_of_nodes(g)}')
    print(f'    | number of edges: {nx.number_of_edges(g)}')

    print("Creating SciPy sparse matrix from networkx graph...")
    a = nx.adjacency_matrix(g)
    # Taking its transpose since the NetworkX adjacency matrix representation is different
    a = sp.sparse.csc_matrix((a.data, a.indices, a.indptr), shape=a.shape)
    print(f'    | approximate size: {np.round((a.data.nbytes + a.indptr.nbytes + a.indices.nbytes) / 1000000)}mb')
    if a.shape[0] <= 20:
        print(a.toarray())

    print("Normalizing the matrix to make it (nearly) stochastic...")
    a = normalize(a, norm='l1', axis=0)
    if a.shape[0] <= 20:
        print(a.toarray())

    print(f'Writing matrix to \'{mat_file}\'...')
    sp.sparse.save_npz(f'{mat_file}', a)

print("Finding the columns which are all zeros...")
zero_cols = np.diff(a.indptr) == 0
print(f'Number all-zeros columns: {np.sum(zero_cols)}')

last = np.ones(a.shape[1]) / a.shape[1]
print(f'Loading ranking \'{rank_file}\'...')
try:
    last = np.load(open(rank_file, 'rb'))
except:
    print(f'Cannot load \'{rank_file}\'')

print("Running power iteration...")
page_rank = power_iter(a, last, zero_cols, tol=error_tol)

print(f'Writing ranking to \'{rank_file}\'...')
np.save(rank_file, page_rank)

print("Printing top min(25, # of pages) pages...")
top_pages = np.argsort(page_rank)[:-26:-1]
for page in top_pages:
    print(f'    | {page}')
