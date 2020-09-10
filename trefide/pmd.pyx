# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
import os
import sys
import multiprocessing

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from sklearn.utils.extmath import randomized_svd as svd
from functools import partial
import matplotlib.pyplot as plt
from libcpp cimport bool

FOV_BHEIGHT_WARNING = "Input FOV height must be an evenly divisible by block height."
FOV_BWIDTH_WARNING = "Input FOV width must be evenly divisible by block width."
DSUB_BHEIGHT_WARNING = "Block height must be evenly divisible by spatial downsampling factor."
DSUB_BWIDTH_WARNING = "Block width must be evenly divisible by spatial downsampling factor."
TSUB_FRAMES_WARNING = "Num Frames must be evenly divisible by temporal downsampling factor."


# -----------------------------------------------------------------------------#
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":

    cdef cppclass PMD_params:
        PMD_params(
            const int _bheight,
            const int _bwidth,
            int _d_sub,
            const int _t,
            int _t_sub,
            const double _spatial_thresh,
            const double _temporal_thresh,
            const size_t _max_components,
            const size_t _consec_failures,
            const size_t _max_iters_main,
            const size_t _max_iters_init,
            const double _tol,
            void *_FFT,
            bool _enable_temporal_denoiser,
            bool _enable_spatial_denoiser) nogil

    size_t pmd(
            double* R,
            double* R_ds,
            double* U,
            double* V,
            PMD_params *pars) nogil

    void batch_pmd(double** Up,
                   double** Vp,
                   size_t* K,
                   const int num_blocks,
                   PMD_params* pars,
                   double* movie,
                   int fov_height,
                   int fov_width,
                   size_t* indices) nogil

    void downsample_3d(const int d1,
                       const int d2,
                       const int d_sub,
                       const int t,
                       const int t_sub,
                       const double *Y,
                       double *Y_ds) nogil


# -----------------------------------------------------------------------------#
# -------------------------- Single-Block Wrapper -----------------------------#
# -----------------------------------------------------------------------------#


cpdef size_t decompose(const int d1, const int d2, const int t, double[::1] Y,
        double[::1] U, double[::1] V, const double spatial_thresh, const double
        temporal_thresh, const size_t max_components, const size_t
        consec_failures, const size_t max_iters_main, const size_t
        max_iters_init, const double tol, bool enable_temporal_denoiser = True,
        bool enable_spatial_denoiser = True) nogil:
    """Apply TF/TV Penalized Matrix Decomposition (PMD) to factor a
       column major formatted video into spatial and temporal components.

    Parameters
    ----------
    d1 :
        height of video
    d2 :
        width of video
    t :
        frames of video
    Y :
        video data of shape (d1 x d2) x t
    U :
        decomposed spatial component matrix
    V :
        decomposed temporal component matrix
    spatial_thresh :
        spatial threshold
    temporal_thresh :
        temporal threshold
    max_components :
        maximum number of components
    consec_failures :
        number of failures before stopping
    max_iters_main :
        maximum number of iterations refining a component
    max_iters_init :
        maximum number of iterations refining a component during decimated
        initialization
    tol : convergence tolerence
    enable_temporal_denoiser :
        whether enable temporal denoiser, True by default
    enable_spatial_denoiser :
        whether enable spatial denoiser, True by default

    Returns
    -------
    result :
        rank of the compressed video/patch
    """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        parms = new PMD_params(d1, d2, 1, t, 1,
                               spatial_thresh, temporal_thresh,
                               max_components, consec_failures,
                               max_iters_main, max_iters_init, tol,
                               NULL,
                               enable_temporal_denoiser,
                               enable_spatial_denoiser)
        result = pmd(&Y[0], NULL, &U[0], &V[0], parms)
        del parms
        return result


cpdef size_t decimated_decompose(const int d1,
                                 const int d2,
                                 int d_sub,
                                 const int t,
                                 int t_sub,
                                 double[::1] Y,
                                 double[::1] Y_ds,
                                 double[::1] U,
                                 double[::1] V,
                                 const double spatial_thresh,
                                 const double temporal_thresh,
                                 const size_t max_components,
                                 const size_t consec_failures,
                                 const int max_iters_main,
                                 const int max_iters_init,
                                 const double tol,
                                 bool enable_temporal_denoiser = True,
                                 bool enable_spatial_denoiser = True) nogil:
    """ Apply decimated TF/TV Penalized Matrix Decomposition (PMD) to factor a
    column major formatted video into spatial and temporal components.

    Parameters
    ----------
    d1 :
        height of video
    d2 :
        width of video
    d_sub :
        spatial downsampling factor
    t :
        frames of video
    t_sub :
        temporal downsampling factor
    Y :
        video data of shape (d1 x d2) x t
    Y_ds :
        downsampled video data
    U :
        decomposed spatial matrix
    V :
        decomposed temporal matrix
    spatial_thresh :
        spatial threshold,
    temporal_thresh :
        temporal threshold,
    max_components :
        maximum number of components,
    consec_failures :
        number of failures before stopping
    max_iters_main :
        maximum number of iterations refining a component
    max_iters_init :
        maximum number of iterations refining a component during decimated initialization
    tol :
        convergence tolerence
    enable_temporal_denoiser :
        whether enable temporal denoiser, True by default
    enable_spatial_denoiser :
        whether enable spatial denoiser, True by default

    Returns
    -------
    result :
        rank of the compressed video/patch
    """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        parms = new PMD_params(d1, d2, d_sub, t, t_sub,
                               spatial_thresh, temporal_thresh,
                               max_components, consec_failures,
                               max_iters_main, max_iters_init, tol,
                               NULL,
                               enable_temporal_denoiser,
                               enable_spatial_denoiser)
        result = pmd(&Y[0], &Y_ds[0], &U[0], &V[0], parms)
        del parms
        return result


# -----------------------------------------------------------------------------#
# --------------------------- Multi-Block Wrappers ----------------------------#
# -----------------------------------------------------------------------------#

cpdef batch_decompose(const int d1,
                      const int d2,
                      const int t,
                      double[:, :, ::1] Y,
                      const int bheight,
                      const int bwidth,
                      const double spatial_thresh,
                      const double temporal_thresh,
                      const size_t max_components,
                      const size_t consec_failures,
                      const size_t max_iters_main,
                      const size_t max_iters_init,
                      const double tol,
                      int d_sub = 1,
                      int t_sub = 1,
                      bool enable_temporal_denoiser = True,
                      bool enable_spatial_denoiser = True):
    """ Apply TF/TV Penalized Matrix Decomposition (PMD) in batch to factor a
    column major formatted video into spatial and temporal components.

    Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function
    factor_patch with OpenMP directives to parallelize batch processing.

    Parameters
    ----------
    d1 :
        height of video
    d2 :
        width of video
    t :
        frames of video
    Y :
        video data of shape (d1 x d2) x t
    bheight :
        height of video block
    bwidth :
        width of video block
    spatial_thresh :
        spatial threshold,
    temporal_thresh :
        temporal threshold,
    max_components :
        maximum number of components,
    consec_failures :
        number of failures before stopping
    max_iters_main :
        maximum number of iterations refining a component
    max_iters_init :
        maximum number of iterations refining a component during decimated
        initialization
    tol :
        convergence tolerence
    d_sub :
        spatial downsampling factor
    t_sub :
        temporal downsampling factor
    enable_temporal_denoiser :
        whether enable temporal denoiser, True by default
    enable_spatial_denoiser :
        whether enable spatial denoiser, True by default

    Returns
    -------
    U :
        spatial components matrix
    V :
        temporal components matrix
    K :
        rank of each patch
    indices :
        location/index inside of patch grid
    """

    # Assert Evenly Divisible FOV/Block Dimensions
    if d1 % bheight != 0:
        raise ValueError(FOV_BHEIGHT_WARNING+" d1: {} bheight: {}".format(d1, bheight))
    if d2 % bwidth != 0:
        raise ValueError(FOV_BWIDTH_WARNING+" d2: {} d_sub: {}".format(d2, bwidth))
    if bheight % d_sub != 0:
        raise ValueError(DSUB_BHEIGHT_WARNING+" bheight {}: d_sub: {}".format(bheight, d_sub))
    if bwidth % d_sub != 0:
        raise ValueError(DSUB_BWIDTH_WARNING+" bwidth {}: d_sub: {}".format(bwidth, d_sub))
    if t % t_sub != 0:
        raise ValueError(TSUB_FRAMES_WARNING+" t {}: t_sub: {}".format(t, t_sub))

    # Initialize Counters
    # cdef size_t iu, ku
    # cdef int h, w, k, b, bi, bj, frame
    cdef int nbi = d1 // bheight
    cdef int nbj = d2 // bwidth
    cdef int num_blocks = nbi * nbj
    # cdef int bheight_ds = bheight / d_sub
    # cdef int bwidth_ds = bwidth / d_sub
    # cdef int t_ds = t / t_sub

    # cdef int fov_height = d1
    # cdef int fov_width = d2

    # Compute block-start indices and spatial cutoff
    # TODO: add in function to generalize indices map
    cdef size_t[:, ::1] indices
    ind = np.array([np.tile(range(nbi), nbj), np.repeat(range(nbj), nbi)], dtype=np.uint64)
    ind = np.transpose(ind)
    indices = np.ascontiguousarray(ind) * np.array([bheight, bwidth], dtype=np.uint64)[None, :]

    strides = np.asarray(Y).strides
    itemsize = np.asarray(Y).itemsize
    cdef int height_stride = strides[0] / itemsize
    cdef int width_stride = strides[1] / itemsize

    # Preallocate Space For Outputs
    cdef double[:,::1] U = np.zeros((num_blocks, bheight * bwidth * max_components), dtype=np.float64)
    cdef double[:,::1] V = np.zeros((num_blocks, t * max_components), dtype=np.float64)
    cdef size_t[::1] K = np.empty((num_blocks,), dtype=np.uint64)

    # Allocate Input Pointers
    # cdef double** Rp = <double **> malloc(num_blocks * sizeof(double*))
    # cdef double** Rp_ds = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Vp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Up = <double **> malloc(num_blocks * sizeof(double*))

    # Release Gil Prior To Referencing Address & Calling Multithreaded Code
    with nogil:

        # Assign Pre-allocated Output Memory To Pointer Array & Allocate Residual Pointers
        for b in range(num_blocks):
            Up[b] = &U[b, 0]
            Vp[b] = &V[b, 0]

        params = new PMD_params(bheight, bwidth, d_sub, t, t_sub,
                                spatial_thresh, temporal_thresh, max_components,
                                consec_failures, max_iters_main, max_iters_init,
                                tol, NULL, enable_temporal_denoiser,
                                enable_spatial_denoiser)

        batch_pmd(Up, Vp, &K[0], num_blocks, params, &Y[0, 0, 0], height_stride, width_stride, &indices[0, 0])

        del params

    free(Up)
    free(Vp)

    # Format Components & Return To Numpy Array
    return (np.asarray(U).reshape((num_blocks, bheight, bwidth, max_components), order='F'),
            np.asarray(V).reshape((num_blocks, max_components, t), order='C'),
            np.asarray(K), np.asarray(ind))


cpdef double[:,:,::1] batch_recompose(double[:, :, :, :] U, double[:,:,::1] V,
        size_t[::1] K, size_t[:,:] indices):
    """Reconstruct A Denoised Movie from components returned by batch_decompose

    Parameters
    ----------
    U :
        spatial component matrix
    V :
        temporal component matrix
    K :
        rank of each patch
    indices :
        location/index inside of patch grid

    Returns
    -------
    Yd :
        denoised video data
    """

    # Get Block Size Info From Spatial
    # cdef size_t num_blocks = U.shape[0]
    cdef size_t bheight = U.shape[1]
    cdef size_t bwidth = U.shape[2]
    cdef size_t t = V.shape[2]

    # Get Mvie Size Infro From Indices
    cdef size_t nbi, nbj
    nbi = np.max(indices[:,0]) + 1
    nbj = np.max(indices[:,1]) + 1
    cdef size_t d1 = nbi * bheight
    cdef size_t d2 = nbj * bwidth

    # Allocate Space For reconstructed Movies
    Yd = np.zeros(d1*d2*t, dtype=np.float64).reshape((d1,d2,t))

    # Loop Over Blocks
    cdef size_t bdx, idx, jdx #, kdx
    for bdx in range(nbi*nbj):
        idx = indices[bdx,0] * bheight
        jdx = indices[bdx,1] * bwidth
        Yd[idx:idx+bheight, jdx:jdx+bwidth,:] += np.reshape(
                np.dot(U[bdx,:,:,:K[bdx]],
                       V[bdx,:K[bdx],:]),
                (bheight,bwidth,t),
                order='F')

    # Rank One updates
    return np.asarray(Yd)


# -----------------------------------------------------------------------------#
# --------------------------- Overlapping Wrappers ----------------------------#
# -----------------------------------------------------------------------------#


cpdef double[:,:,::1] weighted_recompose(double[:, :, :, :] U, double[:,:,:] V,
        size_t[:] K, size_t[:,:] indices, double[:,:] W):
    """Reconstruct A Denoised Movie from components returned by batch_decompose

    Parameters
    ----------
    U :
        spatial component matrix
    V :
        temporal component matrix
    K :
        rank of each patch
    indices :
        location/index inside of patch grid
    W :
        weighting component matrix

    Returns
    -------
    Yd :
        denoised video data
    """

    # Get Block Size Info From Spatial
    # cdef size_t num_blocks = U.shape[0]
    cdef size_t bheight = U.shape[1]
    cdef size_t bwidth = U.shape[2]
    cdef size_t t = V.shape[2]

    # Get Movie Size Info From Indices
    cdef size_t nbi, nbj
    nbi = len(np.unique(indices[:,0]))
    idx_offset = np.min(indices[:,0])
    nbj = len(np.unique(indices[:,1]))
    jdx_offset = np.min(indices[:,1])
    cdef size_t d1 = nbi * bheight
    cdef size_t d2 = nbj * bwidth

    # Allocate Space For reconstructed Movies
    Yd = np.zeros(d1*d2*t, dtype=np.float64).reshape((d1, d2, t))

    # Loop Over Blocks
    cdef size_t bdx, idx, jdx #, kdx
    for bdx in range(nbi*nbj):
        idx = (indices[bdx, 0] - idx_offset) * bheight
        jdx = (indices[bdx, 1] - jdx_offset) * bwidth
        Yd[idx:idx+bheight, jdx:jdx+bwidth,:] += np.reshape(
                np.dot(U[bdx,:,:,:K[bdx]], V[bdx,:K[bdx],:]),
                (bheight, bwidth, t),
                order='F') * np.asarray(W[:, :, None])
    return np.asarray(Yd)


cpdef overlapping_batch_decompose(const int d1,
                                  const int d2,
                                  const int t,
                                  double[:, :, ::1] Y,
                                  const int bheight,
                                  const int bwidth,
                                  const double spatial_thresh,
                                  const double temporal_thresh,
                                  const size_t max_components,
                                  const size_t consec_failures,
                                  const size_t max_iters_main,
                                  const size_t max_iters_init,
                                  const double tol,
                                  int d_sub=1,
                                  int t_sub=1,
                                  bool enable_temporal_denoiser = True,
                                  bool enable_spatial_denoiser = True):
    """ 4x batch denoiser. Apply TF/TV Penalized Matrix Decomposition (PMD) in
    batch to factor a column major formatted video into spatial and temporal
    components.

    Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function
    factor_patch with OpenMP directives to parallelize batch processing.

    Parameters
    ----------
    d1 :
        height of video
    d2 :
        width of video
    t :
        frames of video
    Y :
        video data of shape (d1 x d2) x t
    bheight :
        height of video block
    bwidth :
        width of video block
    spatial_thresh :
        spatial threshold,
    temporal_thresh :
        temporal threshold,
    max_components :
        maximum number of components,
    consec_failures :
        number of failures before stopping
    max_iters_main :
        maximum number of iterations refining a component
    max_iters_init :
        maximum number of iterations refining a component during decimated
        initialization
    tol :
        convergence tolerence
    d_sub :
        spatial downsampling factor
    t_sub :
        temporal downsampling factor
    enable_temporal_denoiser :
        whether enable temporal denoiser, True by default
    enable_spatial_denoiser :
        whether enable spatial denoiser, True by default

    Returns
    -------
    U :
        spatial components matrix
    V :
        temporal components matrix
    K :
        rank of each patch
    I :
        location/index inside of patch grid
    W :
        weighting components matrix
    """
    # Assert Even Blockdims
    if (bheight & 1):
        raise ValueError("Block height must be an even integer.")
    if (bwidth & 1):
        ValueError("Block width must be an even integer.")

    # Assert Even Blockdims
    if d1 % bheight != 0:
        raise ValueError("Input FOV height must be an evenly divisible by block height.")
    if d2 % bwidth != 0:
        raise ValueError("Input FOV width must be evenly divisible by block width.")

    # Declare internal vars
    cdef int i,j
    cdef int hbheight = bheight/2
    cdef int hbwidth = bwidth/2
    # cdef int nbrow = d1/bheight
    # cdef int nbcol = d2/bwidth

    # -------------------- Construct Combination Weights ----------------------#

    # Generate Single Quadrant Weighting matrix
    cdef double[:,:] ul_weights = np.empty((hbheight, hbwidth), dtype=np.float64)
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = min(i, j)

    # Compute Cumulative Overlapped Weights (Normalizing Factor)
    cdef double[:,:] cum_weights = np.asarray(ul_weights) +\
            np.fliplr(ul_weights) + np.flipud(ul_weights) +\
            np.fliplr(np.flipud(ul_weights))

    # Normalize By Cumulative Weights
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = ul_weights[i,j] / cum_weights[i,j]


    # Construct Full Weighting Matrix From Normalize Quadrant
    cdef double[:,:] W = np.hstack([np.vstack([ul_weights,
                                               np.flipud(ul_weights)]),
                                    np.vstack([np.fliplr(ul_weights),
                                               np.fliplr(np.flipud(ul_weights))])])

    # Initialize Outputs
    cdef dict U = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict V = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict K = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict I = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}

    # ---------------- Handle Blocks Overlays One At A Time --------------#

    # ----------- Original Overlay -----------
    # Only Need To Process Full-Size Blocks
    U['no_skew']['full'],\
    V['no_skew']['full'],\
    K['no_skew']['full'],\
    I['no_skew']['full'] = batch_decompose(d1, d2, t, Y, bheight, bwidth,
                                           spatial_thresh,temporal_thresh,
                                           max_components, consec_failures,
                                           max_iters_main, max_iters_init,
                                           tol, d_sub, t_sub,
                                           enable_temporal_denoiser,
                                           enable_spatial_denoiser)

    # ---------- Vertical Skew -----------
    # Full Blocks
    U['vert_skew']['full'],\
    V['vert_skew']['full'],\
    K['vert_skew']['full'],\
    I['vert_skew']['full'] = batch_decompose(d1 - bheight, d2, t,
                                             Y[hbheight:d1-hbheight,:,:],
                                             bheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init,
                                             tol, d_sub, t_sub,
                                             enable_temporal_denoiser,
                                             enable_spatial_denoiser)

    # wide half blocks
    U['vert_skew']['half'],\
    V['vert_skew']['half'],\
    K['vert_skew']['half'],\
    I['vert_skew']['half'] = batch_decompose(bheight, d2, t,
                                             np.vstack([Y[:hbheight,:,:],
                                                        Y[d1-hbheight:,:,:]]),
                                             hbheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init,
                                             tol, d_sub, t_sub,
                                             enable_temporal_denoiser,
                                             enable_spatial_denoiser)

    # --------------Horizontal Skew----------
    # Full Blocks
    U['horz_skew']['full'],\
    V['horz_skew']['full'],\
    K['horz_skew']['full'],\
    I['horz_skew']['full'] = batch_decompose(d1, d2 - bwidth, t,
                                             Y[:, hbwidth:d2-hbwidth,:],
                                             bheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init,
                                             tol, d_sub, t_sub,
                                             enable_temporal_denoiser,
                                             enable_spatial_denoiser)

    # tall half blocks
    U['horz_skew']['half'],\
    V['horz_skew']['half'],\
    K['horz_skew']['half'],\
    I['horz_skew']['half'] = batch_decompose(d1, bwidth, t,
                                             np.hstack([Y[:,:hbwidth,:],
                                                        Y[:,d2-hbwidth:,:]]),
                                             bheight, hbwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init,
                                             tol, d_sub, t_sub,
                                             enable_temporal_denoiser,
                                             enable_spatial_denoiser)

    # -------------Diagonal Skew----------
    # Full Blocks
    U['diag_skew']['full'],\
    V['diag_skew']['full'],\
    K['diag_skew']['full'],\
    I['diag_skew']['full'] = batch_decompose(d1 - bheight, d2 - bwidth, t,
                                             Y[hbheight:d1-hbheight,
                                               hbwidth:d2-hbwidth, :],
                                             bheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init,
                                             tol, d_sub, t_sub,
                                             enable_temporal_denoiser,
                                             enable_spatial_denoiser)

    # tall half blocks
    U['diag_skew']['thalf'],\
    V['diag_skew']['thalf'],\
    K['diag_skew']['thalf'],\
    I['diag_skew']['thalf'] = batch_decompose(d1 - bheight, bwidth, t,
                                             np.hstack([Y[hbheight:d1-hbheight,
                                                          :hbwidth, :],
                                                        Y[hbheight:d1-hbheight,
                                                          d2-hbwidth:, :]]),
                                             bheight, hbwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init,
                                             tol, d_sub, t_sub,
                                              enable_temporal_denoiser,
                                              enable_spatial_denoiser)

    # wide half blocks
    U['diag_skew']['whalf'],\
    V['diag_skew']['whalf'],\
    K['diag_skew']['whalf'],\
    I['diag_skew']['whalf'] = batch_decompose(bheight, d2 - bwidth, t,
                                             np.vstack([Y[:hbheight,
                                                          hbwidth:d2-hbwidth,
                                                          :],
                                                        Y[d1-hbheight:,
                                                          hbwidth:d2-hbwidth,
                                                          :]]),
                                             hbheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init,
                                             tol, d_sub, t_sub,
                                              enable_temporal_denoiser,
                                              enable_spatial_denoiser)

    # Corners
    U['diag_skew']['quarter'],\
    V['diag_skew']['quarter'],\
    K['diag_skew']['quarter'],\
    I['diag_skew']['quarter'] = batch_decompose(bheight, bwidth, t,
                                                np.hstack([
                                                    np.vstack([Y[:hbheight,
                                                                 :hbwidth,
                                                                 :],
                                                               Y[d1-hbheight:,
                                                                 :hbwidth,
                                                                 :]]),
                                                    np.vstack([Y[:hbheight,
                                                                 d2-hbwidth:,
                                                                 :],
                                                               Y[d1-hbheight:,
                                                                 d2-hbwidth:,
                                                                 :]])
                                                               ]),
                                                hbheight, hbwidth,
                                                spatial_thresh, temporal_thresh,
                                                max_components, consec_failures,
                                                max_iters_main, max_iters_init,
                                                tol, d_sub, t_sub,
                                                enable_temporal_denoiser,
                                                enable_spatial_denoiser)

    # Return Weighting Matrix For Reconstruction
    return U, V, K, I, W



cpdef overlapping_batch_recompose(const int d1, const int d2, const int t,
        const int bheight, const int bwidth, U, V, K, I, W):
    """4x batch denoiser.

    Reconstruct A Denoised Movie from components returned by batch_decompose.

    Parameters
    ----------
    d1 :
        height of video
    d2 :
        width of video
    t :
        time frames of video
    bheight :
        block height
    bwidth :
        block width
    U :
        spatial components matrix
    V :
        temporal components matrix
    K :
        rank of each patch
    I :
        location/index inside of patch grid
    W :
        weighting components matrix

    Returns
    -------
    Yd :
        recomposed video data matrix
    """
    # Assert Even Blockdims
    if (bheight & 1):
        raise ValueError("Block height must be an even integer.")
    if (bwidth & 1):
        raise ValueError("Block width must be an even integer.")

    # Assert Even Blockdims
    if d1 % bheight != 0:
        raise ValueError("Input FOV height must be evenly divisible by block height.")
    if d2 % bwidth != 0:
        raise ValueError("Input FOV width must be evenly divisible by block width.")

    # Declare internal vars
    # cdef int i,j
    cdef int hbheight = bheight/2
    cdef int hbwidth = bwidth/2
    cdef int nbrow = d1/bheight
    # cdef int nbcol = d2/bwidth

    # Allocate Space For reconstructed Movies
    Yd = np.zeros((d1, d2, t), dtype=np.float64)

    # ---------------- Handle Blocks Overlays One At A Time --------------#

    # ----------- Original Overlay --------------
    # Only Need To Process Full-Size Blocks
    Yd += weighted_recompose(U['no_skew']['full'],
                             V['no_skew']['full'],
                             K['no_skew']['full'],
                             I['no_skew']['full'],
                             W)

    # ---------- Vertical Skew --------------
    # Full Blocks
    Yd[hbheight:d1-hbheight,:,:] += weighted_recompose(U['vert_skew']['full'],
                                                       V['vert_skew']['full'],
                                                       K['vert_skew']['full'],
                                                       I['vert_skew']['full'],
                                                       W)
    # wide half blocks
    Yd[:hbheight,:,:] += weighted_recompose(U['vert_skew']['half'][::2],
                                            V['vert_skew']['half'][::2],
                                            K['vert_skew']['half'][::2],
                                            I['vert_skew']['half'][::2],
                                            W[hbheight:, :])
    Yd[d1-hbheight:,:,:] += weighted_recompose(U['vert_skew']['half'][1::2],
                                               V['vert_skew']['half'][1::2],
                                               K['vert_skew']['half'][1::2],
                                               I['vert_skew']['half'][1::2],
                                               W[:hbheight, :])

    # --------------Horizontal Skew--------------
    # Full Blocks
    Yd[:, hbwidth:d2-hbwidth,:] += weighted_recompose(U['horz_skew']['full'],
                                                      V['horz_skew']['full'],
                                                      K['horz_skew']['full'],
                                                      I['horz_skew']['full'],
                                                      W)
    # tall half blocks
    Yd[:,:hbwidth,:] += weighted_recompose(U['horz_skew']['half'][:nbrow],
                                           V['horz_skew']['half'][:nbrow],
                                           K['horz_skew']['half'][:nbrow],
                                           I['horz_skew']['half'][:nbrow],
                                           W[:, hbwidth:])
    Yd[:,d2-hbwidth:,:] += weighted_recompose(U['horz_skew']['half'][nbrow:],
                                              V['horz_skew']['half'][nbrow:],
                                              K['horz_skew']['half'][nbrow:],
                                              I['horz_skew']['half'][nbrow:],
                                              W[:, :hbwidth])

    # -------------Diagonal Skew--------------
    # Full Blocks
    Yd[hbheight:d1-hbheight, hbwidth:d2-hbwidth, :] += weighted_recompose(U['diag_skew']['full'],
                                                                          V['diag_skew']['full'],
                                                                          K['diag_skew']['full'],
                                                                          I['diag_skew']['full'],
                                                                          W)
    # tall half blocks
    Yd[hbheight:d1-hbheight,:hbwidth,:] += weighted_recompose(U['diag_skew']['thalf'][:nbrow-1],
                                                              V['diag_skew']['thalf'][:nbrow-1],
                                                              K['diag_skew']['thalf'][:nbrow-1],
                                                              I['diag_skew']['thalf'][:nbrow-1],
                                                              W[:, hbwidth:])
    Yd[hbheight:d1-hbheight,d2-hbwidth:,:] += weighted_recompose(U['diag_skew']['thalf'][nbrow-1:],
                                                                 V['diag_skew']['thalf'][nbrow-1:],
                                                                 K['diag_skew']['thalf'][nbrow-1:],
                                                                 I['diag_skew']['thalf'][nbrow-1:],
                                                                 W[:, :hbwidth])
    # wide half blocks
    Yd[:hbheight,hbwidth:d2-hbwidth,:] += weighted_recompose(U['diag_skew']['whalf'][::2],
                                                             V['diag_skew']['whalf'][::2],
                                                             K['diag_skew']['whalf'][::2],
                                                             I['diag_skew']['whalf'][::2],
                                                             W[hbheight:, :])
    Yd[d1-hbheight:,hbwidth:d2-hbwidth,:] += weighted_recompose(U['diag_skew']['whalf'][1::2],
                                                                V['diag_skew']['whalf'][1::2],
                                                                K['diag_skew']['whalf'][1::2],
                                                                I['diag_skew']['whalf'][1::2],
                                                                W[:hbheight, :])
    # Corners
    Yd[:hbheight,:hbwidth,:] += weighted_recompose(U['diag_skew']['quarter'][:1],
                                                   V['diag_skew']['quarter'][:1],
                                                   K['diag_skew']['quarter'][:1],
                                                   I['diag_skew']['quarter'][:1],
                                                   W[hbheight:, hbwidth:])
    Yd[d1-hbheight:,:hbwidth,:] += weighted_recompose(U['diag_skew']['quarter'][1:2],
                                                      V['diag_skew']['quarter'][1:2],
                                                      K['diag_skew']['quarter'][1:2],
                                                      I['diag_skew']['quarter'][1:2],
                                                      W[:hbheight, hbwidth:])
    Yd[:hbheight,d2-hbwidth:,:] += weighted_recompose(U['diag_skew']['quarter'][2:3],
                                                      V['diag_skew']['quarter'][2:3],
                                                      K['diag_skew']['quarter'][2:3],
                                                      I['diag_skew']['quarter'][2:3],
                                                      W[hbheight:, :hbwidth])
    Yd[d1-hbheight:,d2-hbwidth:,:] += weighted_recompose(U['diag_skew']['quarter'][3:],
                                                         V['diag_skew']['quarter'][3:],
                                                         K['diag_skew']['quarter'][3:],
                                                         I['diag_skew']['quarter'][3:],
                                                         W[:hbheight:, :hbwidth])
    return np.asarray(Yd)


cpdef tv_norm(image):
    return np.sum(np.abs(image[:,:-1] - image[:,1:])) + np.sum(np.abs(image[:-1,:] - image[1:,:]))


cpdef spatial_test_statistic(component):
    d1, d2 = component.shape
    return (tv_norm(component) *d1*d2)/ (np.sum(np.abs(component)) * (d1*(d2-1) + d2 * (d1-1)))


cpdef temporal_test_statistic(signal):
    return np.sum(np.abs(signal[2:] + signal[:-2] - 2*signal[1:-1])) / np.sum(np.abs(signal))


cpdef pca_patch(R, bheight=None, bwidth=None, num_frames=None,
        spatial_thresh=None, temporal_thresh=None, max_components=50,
        consec_failures=3, n_iter=7):
    """ """

    # Preallocate Space For Outputs
    U = np.zeros((bheight*bwidth, max_components), dtype=np.float64)
    V = np.zeros((max_components, num_frames), dtype=np.float64)

    if np.sum(np.abs(R)) <= 0:
        return U, V, 0
    # Run SVD on the patch
    Ub, sb, Vtb = svd(R, n_components=max_components, n_iter=n_iter)

    # Iteratively Test & discard components
    fails = 0
    K = 0
    for k in range(max_components):
        spatial_stat = spatial_test_statistic(Ub[:,k].reshape((bheight,bwidth), order='F'))
        temporal_stat = temporal_test_statistic(Vtb[k,:])
        if (spatial_stat > spatial_thresh or temporal_stat > temporal_thresh):
            fails += 1
            if fails >= consec_failures:
                break
        else:
            U[:,K] = Ub[:,k]
            V[K,:] = Vtb[k,:] * sb[k]
            fails = 0
            K += 1
    return U, V, K


cpdef pca_decompose(const int d1,
                    const int d2,
                    const int t,
                    double[:, :, ::1] Y,
                    const int bheight,
                    const int bwidth,
                    const double spatial_thresh,
                    const double temporal_thresh,
                    const size_t max_components,
                    const size_t consec_failures):
    """ Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function
    factor_patch with OpenMP directives to parallelize batch processing.

    Parameters
    ----------
    d1 :
        height of video
    d2 :
        width of video
    t :
        frames of video
    Y :
        video data of shape (d1 x d2) x t
    bheight :
        block height
    bwidth :
        block width
    spatial_thresh :
        spatial threshold,
    temporal_thresh :
        temporal threshold,
    max_components :
        maximum number of components,
    consec_failures :
        number of failures before stopping

    Returns
    -------
    U :
        decomposed spatial matrix
    V :
        decomposed temporal matrix
    K :
        rank of each patch
    indices :
        location/index inside of patch grid
    """

    # Assert Evenly Divisible FOV/Block Dimensions
    if d1 % bheight != 0:
        raise ValueError(FOV_BHEIGHT_WARNING)
    if d2 % bwidth != 0:
        raise ValueError(FOV_BWIDTH_WARNING)

    # Initialize Counters
    # cdef size_t iu, ku
    # cdef int i, j, k, b
    cdef int bi, bj
    cdef int nbi = d1 // bheight
    cdef int nbj = d2 // bwidth
    cdef int num_blocks = nbi * nbj

    # Compute block-start indices and spatial cutoff
    indices = np.transpose([np.tile(range(nbi), nbj), np.repeat(range(nbj), nbi)])

    # cdef size_t good, fails
    # cdef double spatial_stat
    # cdef double temporal_stat

    # Copy Residual For Multiprocessing
    R = []
    for bj in range(nbj):
        for bi in range(nbi):
            R.append(np.reshape(Y[(bi * bheight):((bi+1) * bheight),
                                  (bj * bwidth):((bj+1) * bwidth), :],
                                (bheight*bwidth, t), order='F'))

    # Process In Parallel
    with multiprocessing.Pool(os.cpu_count()) as pool:
        results = pool.map(partial(pca_patch, bheight=bheight, bwidth=bwidth,
            num_frames=t, spatial_thresh=spatial_thresh,
            temporal_thresh=temporal_thresh, max_components=max_components,
            consec_failures=consec_failures), R)

    # Format Components & Return To Numpy Array
    U, V, K = zip(*results)
    return (np.array(U).reshape((num_blocks, bheight, bwidth, max_components), order='F'),
            np.array(V), np.array(K).astype(np.uint64), indices.astype(np.uint64))


cpdef overlapping_pca_decompose(const int d1, const int d2, const int t,
        double[:, :, ::1] Y, const int bheight, const int bwidth, const double
        spatial_thresh, const double temporal_thresh, const size_t
        max_components, const size_t consec_failures):
    """4x batch denoiser

    Parameters
    ----------
    d1 :
        height of video
    d2 :
        width of video
    t :
        frames of video
    Y :
        video data of shape (d1 x d2) x t
    spatial_thresh :
        spatial threshold,
    temporal_thresh :
        temporal threshold,
    max_components :
        maximum number of components,
    consec_failures :

    Returns
    -------
    U :
        spatial components matrix
    V :
        temporal components matrix
    K :
        rank of each patch
    I :
        location/index inside of patch grid
    W :
        weighting components matrix
    """

    # Assert Even Blockdims
    if (bheight & 1):
        raise ValueError("Block height must be an even integer.")
    if (bwidth & 1):
        raise ValueError("Block width must be an even integer.")

    # Assert Even Blockdims
    if d1 % bheight != 0:
        raise ValueError("Input FOV height must be an evenly divisible by block height.")
    if d2 % bwidth != 0:
        raise ValueError("Input FOV width must be evenly divisible by block width.")

    # Declare internal vars
    cdef int i,j
    cdef int hbheight = bheight/2
    cdef int hbwidth = bwidth/2
    # cdef int nbrow = d1/bheight
    # cdef int nbcol = d2/bwidth

    # -------------------- Construct Combination Weights ----------------------#

    # Generate Single Quadrant Weighting matrix
    cdef double[:,:] ul_weights = np.empty((hbheight, hbwidth), dtype=np.float64)
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = min(i, j)

    # Compute Cumulative Overlapped Weights (Normalizing Factor)
    cdef double[:,:] cum_weights = np.asarray(ul_weights) +\
            np.fliplr(ul_weights) + np.flipud(ul_weights) +\
            np.fliplr(np.flipud(ul_weights))

    # Normalize By Cumulative Weights
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = ul_weights[i,j] / cum_weights[i,j]


    # Construct Full Weighting Matrix From Normalize Quadrant
    cdef double[:,:] W = np.hstack([np.vstack([ul_weights,
                                               np.flipud(ul_weights)]),
                                    np.vstack([np.fliplr(ul_weights),
                                               np.fliplr(np.flipud(ul_weights))])])

    # Initialize Outputs
    cdef dict U = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict V = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict K = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict I = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}

    # ---------------- Handle Blocks Overlays One At A Time --------------#

    # ----------- Original Overlay -----------
    # Only Need To Process Full-Size Blocks
    U['no_skew']['full'],\
    V['no_skew']['full'],\
    K['no_skew']['full'],\
    I['no_skew']['full'] = pca_decompose(d1, d2, t, Y, bheight, bwidth,
                                           spatial_thresh,temporal_thresh,
                                           max_components, consec_failures)

    # ---------- Vertical Skew -----------
    # Full Blocks
    U['vert_skew']['full'],\
    V['vert_skew']['full'],\
    K['vert_skew']['full'],\
    I['vert_skew']['full'] = pca_decompose(d1 - bheight, d2, t,
                                             Y[hbheight:d1-hbheight,:,:],
                                             bheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # wide half blocks
    U['vert_skew']['half'],\
    V['vert_skew']['half'],\
    K['vert_skew']['half'],\
    I['vert_skew']['half'] = pca_decompose(bheight, d2, t,
                                             np.vstack([Y[:hbheight,:,:],
                                                        Y[d1-hbheight:,:,:]]),
                                             hbheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # --------------Horizontal Skew----------
    # Full Blocks
    U['horz_skew']['full'],\
    V['horz_skew']['full'],\
    K['horz_skew']['full'],\
    I['horz_skew']['full'] = pca_decompose(d1, d2 - bwidth, t,
                                             Y[:, hbwidth:d2-hbwidth,:],
                                             bheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # tall half blocks
    U['horz_skew']['half'],\
    V['horz_skew']['half'],\
    K['horz_skew']['half'],\
    I['horz_skew']['half'] = pca_decompose(d1, bwidth, t,
                                             np.hstack([Y[:,:hbwidth,:],
                                                        Y[:,d2-hbwidth:,:]]),
                                             bheight, hbwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # -------------Diagonal Skew----------
    # Full Blocks
    U['diag_skew']['full'],\
    V['diag_skew']['full'],\
    K['diag_skew']['full'],\
    I['diag_skew']['full'] = pca_decompose(d1 - bheight, d2 - bwidth, t,
                                             Y[hbheight:d1-hbheight,
                                               hbwidth:d2-hbwidth, :],
                                             bheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # tall half blocks
    U['diag_skew']['thalf'],\
    V['diag_skew']['thalf'],\
    K['diag_skew']['thalf'],\
    I['diag_skew']['thalf'] = pca_decompose(d1 - bheight, bwidth, t,
                                             np.hstack([Y[hbheight:d1-hbheight,
                                                          :hbwidth, :],
                                                        Y[hbheight:d1-hbheight,
                                                          d2-hbwidth:, :]]),
                                             bheight, hbwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # wide half blocks
    U['diag_skew']['whalf'],\
    V['diag_skew']['whalf'],\
    K['diag_skew']['whalf'],\
    I['diag_skew']['whalf'] = pca_decompose(bheight, d2 - bwidth, t,
                                             np.vstack([Y[:hbheight,
                                                          hbwidth:d2-hbwidth,
                                                          :],
                                                        Y[d1-hbheight:,
                                                          hbwidth:d2-hbwidth,
                                                          :]]),
                                             hbheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # Corners
    U['diag_skew']['quarter'],\
    V['diag_skew']['quarter'],\
    K['diag_skew']['quarter'],\
    I['diag_skew']['quarter'] = pca_decompose(bheight, bwidth, t,
                                                np.hstack([
                                                    np.vstack([Y[:hbheight,
                                                                 :hbwidth,
                                                                 :],
                                                               Y[d1-hbheight:,
                                                                 :hbwidth,
                                                                 :]]),
                                                    np.vstack([Y[:hbheight,
                                                                 d2-hbwidth:,
                                                                 :],
                                                               Y[d1-hbheight:,
                                                                 d2-hbwidth:,
                                                                 :]])
                                                               ]),
                                                hbheight, hbwidth,
                                                spatial_thresh, temporal_thresh,
                                                max_components, consec_failures)

    # Return Weighting Matrix For Reconstruction
    return U, V, K, I, W


# Temporary TODO: Optimize, streamline, & add more options (multiple
# simulations when block size large relative to FOV)
def determine_thresholds(mov_dims, block_dims, num_components, max_iters_main,
        max_iters_init, tol, d_sub, t_sub, conf, save_fig=False,
        save_fig_path=None, enable_temporal_denoiser=True,
        enable_spatial_denoiser=True):
    """Determine spatial and temporal threshold.

    Parameters
    ----------
    mov_dims :
        dimension of video (height x width x frames)
    block_dims :
        dimension of each block (height x width)
    num_components :
        number of components
    max_iters_main :
        maximum number of iterations refining a component
    max_iters_init :
        maximum number of iterations refining a component during decimated
        initialization
    tol :
        convergence tolerence
    d_sub :
        spatial downsampling factor
    t_sub :
        temporal downsampling factor
    conf :
        confidence level to determine threshold for the summary statistics
    save_fig :
        whether plot and save plot
    save_fig_path:
        path to save fig
    enable_temporal_denoiser : optional, default: True
        whether to enable temporal denoiser
    enable_spatial_denoiser : optional, default: True
        whether to enable spatial denoiser

    Returns
    -------
    spatial_thresh :
        spatial threshold
    temporal_thresh :
        temporal threshold
    """

    # Simulate Noise Movie
    noise_mov = np.ascontiguousarray(np.reshape(np.random.randn(np.prod(mov_dims)), mov_dims))

    # Perform Blockwise PMD Of Noise Matrix In Parallel
    # NOTE: the last return var is block_indices
    spatial_components,\
    temporal_components,\
    block_ranks,\
    _ = batch_decompose(mov_dims[0], mov_dims[1], mov_dims[2],
                        noise_mov, block_dims[0], block_dims[1],
                        1e3, 1e3,
                        num_components, num_components,
                        max_iters_main, max_iters_init, tol,
                        d_sub, t_sub,
                        enable_temporal_denoiser,
                        enable_spatial_denoiser)

    # Gather Test Statistics
    spatial_stat = []
    temporal_stat = []
    num_blocks = int((mov_dims[0] / block_dims[0]) * (mov_dims[1] / block_dims[1]))
    for block_idx in range(num_blocks):
        for k in range(int(block_ranks[block_idx])):
            spatial_stat.append(spatial_test_statistic(spatial_components[block_idx,:,:,k]))
            temporal_stat.append(temporal_test_statistic(temporal_components[block_idx,k,:]))

    # Compute Thresholds
    spatial_thresh = np.percentile(spatial_stat, conf)
    temporal_thresh = np.percentile(temporal_stat, conf)

    if save_fig:
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].scatter(spatial_stat, temporal_stat, marker='x', c='r', alpha =.2)
        ax[0, 0].axvline(spatial_thresh)
        ax[0, 0].axhline(temporal_thresh)
        ax[0, 1].hist(temporal_stat, bins=20, color='r')
        ax[0, 1].axvline(temporal_thresh)
        ax[0, 1].set_title("Temporal Threshold: {}".format(temporal_thresh))
        ax[1, 0].hist(spatial_stat, bins=20, color='r')
        ax[1, 0].axvline(spatial_thresh)
        ax[1, 0].set_title("Spatial Threshold: {}".format(spatial_thresh))
        if save_fig:
            plt.savefig(f'{save_fig_path}/thresholds_plot.png')
        else:
            plt.show()

    return spatial_thresh, temporal_thresh
