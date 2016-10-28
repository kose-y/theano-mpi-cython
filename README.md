# theano-mpi-cython
Theano + Cuda-enabled MPI, without pycuda via Cython

A simple example code to use MPI on Theano. No need to use PyCUDA.

## Tested with
* Centos 6.7
* 8 Nvidia GTX 980 GPUs
* Python 3.3
* Cython 0.24.1
* Theano 0.9.0dev3.dev-138f48ae02611808091a5778ed1228a8b20c40a5
* OpenMPI 1.10, CUDA-enabled compilation
* mpi4py 2.0.0
* numpy 1.11.2 (only for testing)

## How to use
To build the shared library:
```
make build
```

Then to test (requires at least two GPUs): 
```
make test 
```

or: 
```
mpirun -np <number of threads> python test.py
```
number of threads can be any number of threads/GPUs you want to use. 
`test.py` is set up in a way that each thread uses one separate GPU.
