import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int64_t DTYPE_t
# @cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def cooc(np.ndarray[DTYPE_t, ndim=1] inds_seq, int width = 4, int vocab_size = 1000):
    cdef np.ndarray[DTYPE_t, ndim=2] cooc = np.zeros([vocab_size, vocab_size], dtype=np.int64)
    cdef int length = inds_seq.size
    print(f'processing ind seq of length {length}')
    # cdef DTYPE_t w
    cdef int i, w, ci, c
    for i in range(width, length - width):
        w = inds_seq[i]
        for ci in range(i-width, i):
            c = inds_seq[ci]
            cooc[w, c] += 1
        for ci in range(i+1, i+width+1):
            c = inds_seq[ci]
            cooc[w, c] += 1
    return cooc
