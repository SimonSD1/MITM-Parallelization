# Parallel Meet-in-the-Middle Attack

## Description

This project is focused on parallelizing the **Meet-in-the-Middle Attack** algorithm, which is designed to efficiently search for a "golden collision" in two functions `f(x)` and `g(y)`, such that:

- `f(x) = g(y)`
- A predicate `π(x, y)` holds, i.e., `π(x, y) = 1`

The traditional **Meet-in-the-Middle** attack can be executed sequentially, but this project focuses on optimizing the algorithm through parallelization techniques to handle large input sizes (with `n ≥ 40`) and improve both computation time and memory management.

### Objective

The goal of this project is to parallelize the **Meet-in-the-Middle Attack** using **MPI** (Message Passing Interface) and **OpenMP** to distribute the computation efficiently across multiple processes and cores. The project aims to scale the solution to handle larger input sizes, which is critical as the problem becomes more computationally intensive with increasing `n`.

## Approach

The attack works by:
1. **Dictionary Construction**: Creating a distributed dictionary that stores the pairs `f(x) -> x`.
2. **Parallel Search**: The reverse pass, where for each `y`, we check the corresponding values of `x` such that `g(y) -> x` exists in the dictionary.
3. **Predicate Check**: For each pair `(x, y)`, check if `π(x, y)` holds true. If so, return `(x, y)`.

### Parallelization

- **MPI**: Used to distribute data across multiple compute nodes, enabling efficient distributed memory handling.
- **Sharding**: A technique used to partition the dictionary across processes to minimize memory bottlenecks and optimize parallelization.
