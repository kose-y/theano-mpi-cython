cdef extern from "mpi-compat.h": pass
cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport *
#import numpy as np
#cimport numpy as np
cdef extern from "stdio.h":
    int printf(char*, ...)

cdef c_sayhello(MPI_Comm comm, float * data, int count):
    cdef int size, rank, plen
    cdef char pname[MPI_MAX_PROCESSOR_NAME]
	
    MPI_Comm_size(comm, &size)
    MPI_Comm_rank(comm, &rank)
    MPI_Get_processor_name(pname, &plen)
    printf(b"Hello, World! I am process %d of %d on %s.\n",
           rank, size, pname)
    MPI_Bcast(<void *> data, count, MPI_FLOAT, 0, comm)
	

def sayhello(MPI.Comm comm not None , theano_shared):
    cdef MPI_Comm c_comm = comm.ob_mpi
    cdef long data = theano_shared.container.value.gpudata
    print(comm.rank, theano_shared.get_value()[99])
    c_sayhello(c_comm, <float *> data, theano_shared.container.value.shape[0])
    print(comm.rank, theano_shared.get_value()[99])
	
