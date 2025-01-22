# PageRank Implementation

This project computes PageRank scores for nodes in a directed graph using the power iteration method.

## Overview
- Input: A graph in adjacency-list format (`.adjlist`).
- Process: The program normalizes the adjacency matrix, handles dangling nodes, and applies a damping factor to simulate random web surfing.
- Output: PageRank scores saved to a `.npy` file.

## Usage
Run the program with:  
python3 pagerank.py <filename> <tol>

- `<filename>`: The `.adjlist` file describing the graph.
- `<tol>`: Tolerance as `10^(-tol)` for convergence.

## Example
To run on a sample graph:  
python3 pagerank.py basic.adjlist 10

## Dependencies
Install required libraries using:  
pip install numpy scipy scikit-learn networkx

## Files
- `pagerank.py`: Main script for the algorithm.
- `pagerank_utils.py`: Helper functions for computation.
- `.adjlist` files: Input graphs.
- `.npz` files: Sparse adjacency matrices.
- `.npy` files: PageRank scores.

## Makefile Commands
- `basic`: Run on `basic.adjlist`.
- `stanford`, `berkstan`, `google`: Run on specific datasets.
- `destroy`: Remove output files.
- `fullrun`: Run all datasets.


