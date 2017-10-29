/*
 * GPURightLooking.cpp
 *
 *  Created on: 2017/10/27
 *      Author: stomo
 *
 *  *** Single Stream Version ***
 */

#define _COUT

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <omp.h>

#include <plasma.h>
#include <core_blas.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#define GDEV_NUM 1
#define GDEV_ID  1

#include "TileMatrix.hpp"

using namespace std;

void tileQR( TileMatrix *A, TileMatrix *T )
{
	const int MT = A->mt();
	const int NT = A->nt();
	const int NB = A->mb(0,0);
	const int IB = A->ib();

	// Assume that tile is square
	assert(NB == A->nb(0,0));

	// Initialize GPU
	cudaSetDevice(GDEV_ID);

	////////////////////////////////////////////////////////////////////////////
	cudaError_t	    cuda_stat;
//	cudaStream_t   *stream = new cudaStream_t[NT];
	cublasStatus_t	cublas_stat;
	cublasHandle_t *handle = new cublasHandle_t[NT];

	// Create device handler
	for (int j=0; j<NT; j++)
	{
		cublas_stat = cublasCreate(&handle[j]);
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << j << "-th CUDA handle initialize failure\n";
			cublasDestroy(handle[j]);
			exit(EXIT_FAILURE);
		}
	}

	// Allocate device memory
	double **dAk   = new double*[NT];
	double **dAi   = new double*[NT];
	double **dWork = new double*[NT];
	double  *dT;

	for (int j=0; j<NT; j++)
	{
		cuda_stat = cudaMalloc( (void**) &dAk[j], sizeof(double)*NB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dAk[" << j << "]\n";
			exit(EXIT_FAILURE);
		}

		cuda_stat = cudaMalloc( (void**) &dAi[j], sizeof(double)*NB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dAi[" << j << "]\n";
			exit(EXIT_FAILURE);
		}

		cuda_stat = cudaMalloc( (void**) &dWork[j], sizeof(double)*IB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dWork[" << j << "]\n";
			exit(EXIT_FAILURE);
		}
	}

	cuda_stat = cudaMalloc( (void**) &dT, sizeof(double)*IB*NB );
	if( cuda_stat != cudaSuccess )
	{
		cerr << "Device memory allocate failure for dT\n";
		exit(EXIT_FAILURE);
	}
	////////////////////////////////////////////////////////////////////////////

	double ttime = omp_get_wtime();

	////////////////////////////////////////////////////////////////////////////
	// Right Looking tile QR
	for (int tk=0; tk < min(MT,NT); tk++ )
	{
		//
		// GEQRT part
		//
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

		//
		// Set elements data of dAk[k]
		//
		{
			cublas_stat = cublasSetMatrix( A->mb(tk,tk), A->nb(tk,tk), sizeof(double),
					A->ttop(tk,tk), A->mb(tk,tk), dAk[tk], A->mb(tk,tk) );
			if( cublas_stat != CUBLAS_STATUS_SUCCESS)
			{
				cerr << "Data copy to dAk[k] failure\n";
				exit(EXIT_FAILURE);
			}
		}
		//
		// Set elements data of dT
		//
		{
			cublas_stat = cublasSetMatrix( T->mb(tk,tk), T->nb(tk,tk), sizeof(double),
					T->ttop(tk,tk), T->mb(tk,tk), dT, T->mb(tk,tk) );
			if( cublas_stat != CUBLAS_STATUS_SUCCESS)
			{
				cerr << "Data copy to dT failure (GEQRT)\n";
				exit(EXIT_FAILURE);
			}
		}

		for (int tj=tk+1; tj < NT; tj++)
		{
			//
			// Send elements data to dAk[j]
			//
			{
				cublas_stat = cublasSetMatrix( A->mb(tk,tj), A->nb(tk,tj), sizeof(double),
						A->ttop(tk,tj), A->mb(tk,tj), dAk[tj], A->mb(tk,tj) );
				if( cublas_stat != CUBLAS_STATUS_SUCCESS)
				{
					cerr << "Data copy to dAk[" << tj << "] failure\n";
					exit(EXIT_FAILURE);
				}
			}

			//
			// LARFB part
			//
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
			//
			// TSQRT part
			//
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

			//
			// Set elements data of dAi[k]
			//
			{
				cublas_stat = cublasSetMatrix( A->mb(ti,tk), A->nb(ti,tk), sizeof(double),
						A->ttop(ti,tk), A->mb(ti,tk), dAi[tk], A->mb(ti,tk) );
				if( cublas_stat != CUBLAS_STATUS_SUCCESS)
				{
					cerr << "Data copy to dAi[k] failure\n";
					exit(EXIT_FAILURE);
				}
			}
			//
			// Set elements data of dT
			//
			{
				cublas_stat = cublasSetMatrix( T->mb(ti,tk), T->nb(ti,tk), sizeof(double),
						T->ttop(ti,tk), T->mb(ti,tk), dT, T->mb(ti,tk) );
				if( cublas_stat != CUBLAS_STATUS_SUCCESS)
				{
					cerr << "Data copy to dT failure (TSQRT)\n";
					exit(EXIT_FAILURE);
				}
			}

			for (int tj=tk+1; tj < NT; tj++)
			{
				//
				// Send elements data to dAi[j]
				//
				{
					cublas_stat = cublasSetMatrix( A->mb(ti,tj), A->nb(ti,tj), sizeof(double),
							A->ttop(ti,tj), A->mb(ti,tj), dAi[tj], A->mb(ti,tj) );
					if( cublas_stat != CUBLAS_STATUS_SUCCESS)
					{
						cerr << "Data copy to dAi[" << tj << "] failure\n";
						exit(EXIT_FAILURE);
					}
				}

				//
				// SSRFB part
				//
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

				//
				// Send elements data of dAk[j] back
				//
				{
					cublas_stat = cublasGetMatrix( A->mb(tk,tj), A->nb(tk,tj), sizeof(double),
							dAk[tj], A->mb(tk,tj), A->ttop(tk,tj), A->mb(tk,tj) );
					if( cublas_stat != CUBLAS_STATUS_SUCCESS)
					{
						cerr << "Data copy to Akj failure\n";
						exit(EXIT_FAILURE);
					}
				}
				//
				// Send elements data of dAi[j] back
				//
				{
					cublas_stat = cublasGetMatrix( A->mb(ti,tj), A->nb(ti,tj), sizeof(double),
							dAi[tj], A->mb(ti,tj), A->ttop(ti,tj), A->mb(ti,tj) );
					if( cublas_stat != CUBLAS_STATUS_SUCCESS)
					{
						cerr << "Data copy to Aij failure\n";
						exit(EXIT_FAILURE);
					}
				}
			} // j-LOOP END
		} // i-LOOP END
	} // k-LOOP END

	//////////////////////////////////////////////////////////////////////
	for (int j=0; j<NT; j++)
	{
		cudaFree(dAk[j]);
		cudaFree(dAi[j]);
		cudaFree(dWork[j]);
		cublasDestroy(handle[j]);
	}
	cudaFree(dT);
	//////////////////////////////////////////////////////////////////////
	// Right Looking tile QR END
	//////////////////////////////////////////////////////////////////////
}



