#include <stdlib.h>
#include <mkl.h>
#include <math.h>
#include "pmd.h"

/*----------------------------------------------------------------------------*
 *-------------------------- Downsampling Routines ---------------------------*
 *----------------------------------------------------------------------------*/


/* Compute downsampled signal as average in each (ds, ) neighborhood */
void downsample_1d(const int t, 
                   const int ds, 
                   const double* v, 
                   double* v_ds)
{
    int t_ds = t / ds;
    int i_ds;
    for (i_ds = 0; i_ds < t_ds; i_ds++){
        int i;
        v_ds[i_ds] = 0;
        for (i = 0; i < ds; i++){
            v_ds[i_ds] += v[i_ds * ds + i];
        }
        v_ds[i_ds] /= ds;
    }
}


/* Compute downsampled image pixels as average in each (ds,ds) neighborhood*/
void downsample_2d(const int d1, 
                   const int d2, 
                   const int ds, 
                   const double* u, 
                   double* u_ds)
{
    int d2_ds = d2 / ds;
    int d1_ds = d1 / ds;
    int j_ds;
    for (j_ds = 0; j_ds < d2_ds; j_ds++){
        int i_ds;
        for (i_ds = 0; i_ds < d1_ds; i_ds++){
            u_ds[i_ds + d1_ds * j_ds] = 0;
            int j;
            for (j = 0; j < ds; j++){
                int i;
                for (i = 0; i < ds; i++){
                    u_ds[i_ds + d1_ds * j_ds] += u[(i_ds*ds + i) + d1 * (j_ds * ds + j)];
                }
            }
            u_ds[i_ds + d1_ds * j_ds] /= ds * ds;
        }
    }
}


/*----------------------------------------------------------------------------*
 *-------------------------- Upsampling Routines -----------------------------*
 *----------------------------------------------------------------------------*/


/* Linearly Interpolate Between Missing Values In 1D Upsampled Array
 */
void upsample_1d_inplace(const int t, 
                         const int ds, 
                         double* v)
{
    /* Declare & Initialize Local Variables */
    int i, i_ds, t_ds = t / ds;
    double start, end = v[0];

    /* Fill Missing Elements Between Observed Samples*/
    for (i_ds = 0; i_ds < t_ds - 2; i_ds++){
        start = end;
        end = v[(i_ds + 1) * ds];
        for (i = 1; i < ds; i++) v[i_ds * ds + i] = start  + (i / ds) * (end - start); 
    }

    /* Continue Linear Trend For Trailing Elements */
    end = v[t - ds] + end - start; 
    start = v[t - ds];  
    for (i = 1; i < ds; i++) v[t - ds + i] = start  + (i / ds) * (end - start);
}


/* Fill Upsampled 1D Array From Downsampled 1D Array Via Linear Interpolation
 */
void upsample_1d(const int t, 
                 const int ds, 
                 double* v, 
                 const double* v_ds)
{
    /* Declare & Initialize Local Variables */
    int t_ds = t / ds;

    /* Copy Elements Of DS Array Into Respective Locs In US Array */
    cblas_dcopy(t_ds, v_ds, 1, v, ds);
    
    /* Upsample Missing Elements Of US Array */
    upsample_1d_inplace(t, ds, v);
}


/* Fill Upsampled 2D Array From Downsampled 2D Array Via Bilinear Interpolation
 */
void upsample_2d(const int d1, 
                 const int d2, 
                 const int ds, 
                 double* u, 
                 double* u_ds)
{
    /* Declare & Initialize Local Variables */
    int d2_ds = d2 / ds;
    int d1_ds = d1 / ds;

    /* Perform Inplace Transpose To Get Row Major Formatted Image */
    mkl_dimatcopy('C', 'T', d1, d2, 1.0, u, d1, d2);
    mkl_dimatcopy('C', 'T', d1_ds, d2_ds, 1.0, u_ds, d1_ds, d2_ds);

    /* Start By Interpolating Along Rows Where We Have Samples */
    int i_ds;
    for (i_ds = 0; i_ds < d1_ds; i_ds++) 
        upsample_1d(d2, ds, u + i_ds*ds*d2, u_ds + i_ds*d2_ds);

    /* Perform Inplace Transpose To Return To Column Major Format */
    mkl_dimatcopy('C', 'T', d2, d1, 1.0, u, d2, d1);
    mkl_dimatcopy('C', 'T', d2_ds, d1_ds, 1.0, u_ds, d2_ds, d1_ds);

    /* Finish By Interpolating Each Column */
    int j;
    for (j = 0; j < d2; j++) 
        upsample_1d_inplace(d1, ds, u + j*d1);
}


/*----------------------------------------------------------------------------*
 *-------------------------- Decimated PMD Routines --------------------------*
 *----------------------------------------------------------------------------*/


/* Solves: 
 *  u_k, v_k : min <u_k, R_k v_k> - lambda_tv ||u_k||_TV - lambda_tf ||v_k||_TF
 *  
 *  Returns:
 *      1: If we reject the null hypothesis that u_k is noise 
 *     -1: If we accept the null hypothesis that u_k is noise
 */
int rank_one_decomposition(const MKL_INT d1, 
                           const MKL_INT d2, 
                           const MKL_INT t,
                           const double* R_k, 
                           double* u_k, 
                           double* v_k,
                           const double lambda_tv,
                           const double spatial_thresh,
                           const MKL_INT max_iters,
                           const double tol,
                           void* FFT=NULL)
{
 
    /* Declare & Allocate Mem For Internal Vars */
    MKL_INT d, iters;
    double delta_u, delta_v, lambda_tf;
    double *z_k = (double *) malloc((t-2) * sizeof(double));

    /* Initialize Internal Variables */
    d = d1 * d2;
    lambda_tf = initialize_components(d, t, R_k, u_k, v_k, z_k, FFT);

    /* Loop Until Convergence Of Spatial & Temporal Components */
    for (iters = 0; iters < max_iters; iters++){
        
        /* Update Components */
        delta_u = update_spatial(d1, d2, t, R_k, u_k, v_k, lambda_tv);
        delta_v = update_temporal(d, t, R_k, u_k, v_k, z_k, &lambda_tf, FFT);

        /* Check Convergence */
        if (fmax(delta_u, delta_v) < tol){    
            /* Free Allocated Memory & Test Spatial Component Against Null */
            free(z_k);
            if (spatial_test_statistic(d1, d2, u_k) < spatial_thresh) 
                return -1;  // Discard Component
            return 1;  // Keep Component
        }

        /* Preemptive Check To See If We're Fitting Noise */
        if (iters == 9){
            if (spatial_test_statistic(d1, d2, u_k) < spatial_thresh){         
                /* Free Allocated Memory & Return */
                free(z_k); 
                return -1; // Discard Component
            } 
        }    
    }

    /* MAXITER EXCEEDED: Free Memory & Test Spatial Component Against Null */
    free(z_k);
    if (spatial_test_statistic(d1, d2, u_k) < spatial_thresh){ 
        return -1;  // Discard Component
    }
    return 1;  // Keep Component
}


/* Apply TF/TV Penalized Matrix Decomposition (PMD) to factor a (d1*d2)xT
 * column major formatted video into sptial and temporal components.
 */
size_t dec_pmd(const MKL_INT d1, 
               const int d1_ds,
               const MKL_INT d2, 
               const int d2_ds,
               const MKL_INT t,
               const int t_ds,
               double* R, 
               double* R_ds,
               double* U,
               double* V,
               const double lambda_tv,
               const double spatial_thresh,
               const size_t max_components,
               const size_t consec_failures,
               const size_t max_iters,
               const size_t max_iters_ds,
               const double tol,
               void* FFT=NULL)  /* Handle Provided For Threadsafe FFT */
{
    /* Declare & Intialize Internal Vars */
    int* keep_flag = (int *) malloc(consec_failures*sizeof(int));
    int max_keep_flag;
    size_t i, k, good = 0;
    MKL_INT d = d1 * d2;

    /* Allocate Space For Downsampled Components */
    int d_ds = d1_ds * d2_ds;
    double* u_ds = (double *) malloc(d_ds * sizeof(double));
    double* v_ds = (double *) malloc(t_ds * sizeof(double));

    /* Fill Keep Flags */
    for (i = 0; i < consec_failures; i++) keep_flag[i] = 1;

    /* Sequentially Extract Rank 1 Updates Until We Are Fitting Noise */
    for (k = 0; k < max_components; k++, good++) {
        
        /* Solve Decomposition For Downsampled Data */
        keep_flag[k % consec_failures] = rank_one_decomposition(d1_ds, d2_ds, 
                                                                t_ds, R_ds, 
                                                                u_ds, v_ds, 
                                                                lambda_tv, 
                                                                spatial_thresh, 
                                                                max_iters_ds, 
                                                                tol, FFT);
        /* Upsample Results */

        /* U[:,k] <- u_k, V[k,:] <- v_k' : 
         *    min <u_k, R_k v_k> - lambda_tv ||u_k||_TV - lambda_tf ||v_k||_TF
         */
        keep_flag[k % consec_failures] = rank_one_decomposition(d1, d2, t, R, 
                                                                U + good*d, 
                                                                V + good*t, 
                                                                lambda_tv, 
                                                                spatial_thresh, 
                                                                max_iters - max_iters_ds, 
                                                                tol, FFT);

        /* Check Component Quality: Terminate if we're fitting noise */
        if (keep_flag[k % consec_failures] < 0){
            max_keep_flag = -1;  // current component is a failure
            for (i=1; i < consec_failures; i++){  // check previous
                max_keep_flag = max(max_keep_flag, keep_flag[(k+i) % consec_failures]);
            } 
            if (max_keep_flag < 0){
                free(keep_flag);
                return good;
            }
        }
        /* Debias Components */
        regress_temporal(d, t, R, U + good*d, V + good*t);

        /* Update Residual: R_k <- R_k - U[:,k] V[k,:] */
        cblas_dger(CblasColMajor, d, t, -1.0, U + good*d, 1, V + good*t, 1, R, d);

        /* Downsample Components */
        // TODO

        /* Update Downsampled Residual: R_k <- R_k - U[:,k] V[k,:] */
        cblas_dger(CblasColMajor, d_ds, t_ds, -1.0, u_ds, 1, v_ds, 1, R_ds, d_ds);
        
        /* Make Sure We Overwrite Failed Components */
        if (keep_flag[k % consec_failures] < 0) good--;
    }

    /* MAXCOMPONENTS EXCEEDED: Terminate Early */
    free(keep_flag);
    return good; 
    return k;
}


/* Wrap TV/TF Penalized Matrix Decomposition with OMP directives to enable parallel, 
 * block-wiseprocessing of large datasets in shared memory.
 */
void batch_pmd(const MKL_INT bheight, 
               const MKL_INT bwidth, 
               const MKL_INT t,
               const MKL_INT b,
               double** Rpt, 
               double** Upt,
               double** Vpt,
               size_t* Kpt,
               const double lambda_tv,
               const double spatial_thresh,
               const size_t max_components,
               const size_t consec_failures,
               const size_t max_iters,
               const double tol)
{
    // Create FFT Handle So It can Be Shared Aross Threads
    DFTI_DESCRIPTOR_HANDLE FFT;
    MKL_LONG status;
    MKL_LONG L = 256;  /* Size Of Subsamples in welch estimator */
    status = DftiCreateDescriptor( &FFT, DFTI_DOUBLE, DFTI_REAL, 1, L );
    status = DftiSetValue( FFT, DFTI_PACKED_FORMAT, DFTI_PACK_FORMAT);
    status = DftiCommitDescriptor( FFT );
    if (status != 0) 
        fprintf(stderr, "Error while creating MKL_FFT Handle: %ld\n", status);

    // Loop Over All Patches In Parallel
    int m;
    #pragma omp parallel for shared(FFT) schedule(guided)
    for (m = 0; m < b; m++){
        //Use dummy vars for decomposition  
        Kpt[m] = pmd(bheight, bwidth, t, Rpt[m], Upt[m], Vpt[m], lambda_tv, 
                     spatial_thresh, max_components, consec_failures, max_iters, 
                     tol, &FFT);
    }

    // Free MKL FFT Handle
    status = DftiFreeDescriptor( &FFT ); 
    if (status != 0)
        fprintf(stderr, "Error while deallocating MKL_FFT Handle: %ld\n", status);
}