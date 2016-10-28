from mpi4py import MPI
import helloworld as hw

import theano.gof.compilelock
theano.gof.compilelock.set_lock_status(False)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu{}'.format(rank))

import theano

import numpy as np
rng = np.random.RandomState(16962+rank)

vec = rng.rand(1000,1)

shared_vec = theano.shared(np.asarray(vec, theano.config.floatX))

hw.sayhello(comm, shared_vec)


