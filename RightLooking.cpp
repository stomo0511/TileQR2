/*
 * RightLooking.cpp
 *
 *  Created on: 2017/06/21
 *      Author: stomo
 */

#define _COUT
#define _GPU

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <omp.h>

#include <plasma.h>
#include <core_blas.h>

#ifdef _GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define GDEV_NUM 1
#define GDEV_ID  1
#endif

#include "TileMatrix.hpp"

using namespace std;

void tileQR( TileMatrix *A, TileMatrix *T )
{
	const int MT = A->mt();
	const int NT = A->nt();
	const int NB = A->mb(0,0);
	const int IB = A->ib();

	assert(NB == A->nb(0,0));

	#ifdef _GPU
	cudaError_t	    cuda_stat;
	cublasStatus_t	cublas_stat;
	cublasHandle_t *cublas_handle = new cublasHandle_t[NT];
	cudaStream_t   *cuda_stream   = new cudaStream_t[NT];

	cudaSetDevice(GDEV_ID);

	// Allocate device memory for Trailing matrix update
	double **dAkj  = new double*[NT];
	double **dAij  = new double*[NT];
	double **dWork = new double*[NT];

	for(int j=0; j<NT; j++)
	{
		cuda_stat = cudaMalloc( (void**) &dAkj[j], sizeof(double)*NB*NB );
		if( cuda_stat != cudaSuccess ){
			cerr << "Device memory allocate failure for dAkj[" << j << "]\n";
			exit(EXIT_FAILURE);
		}

		cuda_stat = cudaMalloc( (void**) &dAij[j], sizeof(double)*NB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dAij[" << j << "]\n";
			exit(EXIT_FAILURE);
		}

		cuda_stat = cudaMalloc( (void**) &dWork[j], sizeof(double)*IB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dwork[" << j << "]\n";
			exit(EXIT_FAILURE);
		}
	}

	// Allocate device memory for Translation matrix
	double **dAkk = new double*[MT];
	double **dTkk = new double*[MT];

	for(int i=0; i<MT; i++)
	{
		cuda_stat = cudaMalloc( (void**) &dAkk[i], sizeof(double)*NB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dAkk[" << i << "]\n";
			exit(EXIT_FAILURE);
		}

		cuda_stat = cudaMalloc( (void**) &dTkk[i], sizeof(double)*IB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dTkk[" << i << "]\n";
			exit(EXIT_FAILURE);
		}
	}

	// Create CUBLAS handle
	for (int j=0; j<NT; j++)
	{
		cublas_stat = cublasCreate(&cublas_handle[j]);
		if ( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << j << "-th CUBLAS initialization failed\n";
			exit(EXIT_FAILURE);
		}
	}

	// Create CUBLAS stream
	for(int j=0; j<NT; j++)
	{
		cuda_stat = cudaStreamCreate(&cuda_stream[j]);
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Stream create failure for stream[" << j << "]\n";
			exit(EXIT_FAILURE);
		}

		cublas_stat = cublasSetStream( cublas_handle[j], cuda_stream[j]);
		if ( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << j << "-th cublasSetStream failed\n";
			exit(EXIT_FAILURE);
		}
	}
	#endif

	double ttime = omp_get_wtime();

	//////////////////////////////////////////////////////////////////////
	// Right Looking tile QR
	for (int tk=0; tk < min(MT,NT); tk++ )
	{
		{
			double *Tau = new double [A->nb(tk,tk)];
			double *Work = new double [A->nb(tk,tk)*A->ib()];

			int info = core_dgeqrt( A->mb(tk,tk), A->nb(tk,tk), A->ib(),
					A->ttop(tk,tk), A->mb(tk,tk),
					T->ttop(tk,tk), T->mb(tk,tk),
					Tau, Work );
			if (info != PlasmaSuccess)
			{
				cerr << "core_dgeqrt() failed\n";
				exit (EXIT_FAILURE);
			}

			#ifdef _COUT
			#pragma omp critical
			// cout << omp_get_thread_num() << " : " << "GEQRT(" << tk << "," << tk << "," << tk << ") : " << omp_get_wtime() - ttime << "\n";
			cout << omp_get_thread_num() << " : " << "GEQRT(" << tk << "," << tk << "," << tk << ")\n";
			#endif

			delete[] Tau;
			delete[] Work;
		}

		cublas_stat[sid_k] =
				cublasSetMatrixAsync( Akk_m, Akk_n, sizeof(double),
						Akk, Akk_m,
						dAkk[sid_k], Akk_m, stream[sid_k]);
		if( cublas_stat[sid_k] != CUBLAS_STATUS_SUCCESS)
		{
			cout << "dAkkの転送失敗" << endl;
			cublasDestroy(handle[sid_k]);
		}
		//同期を使うので，無理に非同期にする必要はないかも
		cudaStreamSynchronize(stream[sid_k]);

		for (int tj=tk+1; tj < NT; tj++)
		{
			int nb = A->nb(tk,tj);
			double *Work = new double [nb*A->ib()];

			int info = core_dormqr( PlasmaLeft, PlasmaTrans,
					A->mb(tk,tj), A->nb(tk,tj), min(A->mb(tk,tk),A->nb(tk,tk)), A->ib(),
					A->ttop(tk,tk), A->mb(tk,tk),
					T->ttop(tk,tk), T->mb(tk,tk),
					A->ttop(tk,tj), A->mb(tk,tj),
					Work, nb);

			if (info != PlasmaSuccess)
			{
				cerr << "core_dormqr() failed\n";
				exit (EXIT_FAILURE);
			}

			#ifdef _COUT
			#pragma omp critical
			// cout << omp_get_thread_num() << " : " << "LARFB(" << tk << "," << tj << "," << tk << ") : " << omp_get_wtime() - ttime << "\n";
			cout << omp_get_thread_num() << " : " << "LARFB(" << tk << "," << tj << "," << tk << ")\n";
			#endif

			delete[] Work;
		}

		for (int ti=tk+1; ti < MT; ti++)
		{
			{
				double *Tau = new double [A->nb(ti,tk)];
				double *Work = new double [A->nb(ti,tk)*A->ib()];

				int info = core_dtsqrt( A->mb(ti,tk), A->nb(ti,tk), A->ib(),
						A->ttop(tk,tk), A->mb(tk,tk),
						A->ttop(ti,tk), A->mb(ti,tk),
						T->ttop(ti,tk), T->mb(ti,tk),
						Tau, Work);
				if (info != PlasmaSuccess)
				{
					cerr << "core_dtsqrt() failed\n";
					exit (EXIT_FAILURE);
				}

				#ifdef _COUT
				#pragma omp critical
				// cout << omp_get_thread_num() << " : " << "TSQRT(" << ti << "," << tk << "," << tk << ") : " << omp_get_wtime() - ttime << "\n";
				cout << omp_get_thread_num() << " : " << "TSQRT(" << ti << "," << tk << "," << tk << ")\n";
				#endif

				delete[] Tau;
				delete[] Work;
			}

			for (int tj=tk+1; tj < NT; tj++)
			{
				double *Work = new double [A->ib()*A->nb(tk,tj)];

				int info = core_dtsmqr( PlasmaLeft, PlasmaTrans,
						A->mb(tk,tj), A->nb(tk,tj), A->mb(ti,tj), A->nb(ti,tj), A->nb(ti,tk), A->ib(),
						A->ttop(tk,tj), A->mb(tk,tj),
						A->ttop(ti,tj), A->mb(ti,tj),
						A->ttop(ti,tk), A->mb(ti,tk),
						T->ttop(ti,tk), T->mb(ti,tk),
						Work,  A->ib());
				if (info != PlasmaSuccess)
				{
					cerr << "core_dtsmqr() failed\n";
					exit (EXIT_FAILURE);
				}

				#ifdef _COUT
				#pragma omp critical
				// cout << omp_get_thread_num() << " : " << "SSRFB(" << ti << "," << tj << "," << tk << ") : " << omp_get_wtime() - ttime << "\n";
				cout << omp_get_thread_num() << " : " << "SSRFB(" << ti << "," << tj << "," << tk << ")\n";
				#endif

				delete[] Work;

			} // j-LOOP END
		} // i-LOOP END
	} // k-LOOP END

	//////////////////////////////////////////////////////////////////////
	#ifdef _GPU
	for(int j=0; j<NT; j++)
	{
		cudaFree(dAkj[j]);
		cudaFree(dAij[j]);
		cudaFree(dWork[j]);
	}

	for(int i=0; i<MT; ++i)
	{
		cudaFree(dTkk[i]);
		cudaFree(dAkk[i]);
	}

	delete [] cublas_handle;
	delete [] cuda_stream;
	#endif
	//////////////////////////////////////////////////////////////////////
	// Right Looking tile QR END
	//////////////////////////////////////////////////////////////////////
}
