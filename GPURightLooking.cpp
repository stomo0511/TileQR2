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

	assert(NB == A->nb(0,0));

	////////////////////////////////////////////////////////////////////////////
	cudaError_t	    cuda_stat;
	cublasStatus_t	cublas_stat;

	// Initialize GPU
	cudaSetDevice(GDEV_ID);

	// Allocate device memory for Translation matrix
	double *dAkk, *dTkk;

	cuda_stat = cudaMalloc( (void**) &dAkk, sizeof(double)*NB*NB );
	if( cuda_stat != cudaSuccess )
	{
		cerr << "Device memory allocate failure for dAkk\n";
		exit(EXIT_FAILURE);
	}

	cuda_stat = cudaMalloc( (void**) &dTkk, sizeof(double)*IB*NB );
	if( cuda_stat != cudaSuccess )
	{
		cerr << "Device memory allocate failure for dTkk\n";
		exit(EXIT_FAILURE);
	}

	// Allocate device memory for Trailing matrix update
	double *dAkj, *dAij, *dWork;

	cuda_stat = cudaMalloc( (void**) &dAkj, sizeof(double)*NB*NB );
	if( cuda_stat != cudaSuccess ){
		cerr << "Device memory allocate failure for dAkj\n";
		exit(EXIT_FAILURE);
	}

	cuda_stat = cudaMalloc( (void**) &dAij, sizeof(double)*NB*NB );
	if( cuda_stat != cudaSuccess )
	{
		cerr << "Device memory allocate failure for dAij\n";
		exit(EXIT_FAILURE);
	}

	cuda_stat = cudaMalloc( (void**) &dWork, sizeof(double)*IB*NB );
	if( cuda_stat != cudaSuccess )
	{
		cerr << "Device memory allocate failure for dWork\n";
		exit(EXIT_FAILURE);
	}
	////////////////////////////////////////////////////////////////////////////

	double ttime = omp_get_wtime();

	//////////////////////////////////////////////////////////////////////
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
		// Set the elements of dAkk, dTkk
		//
		{

		}

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
	cudaFree(dTkk);
	cudaFree(dAkk);

	cudaFree(dAkj);
	cudaFree(dAij);
	cudaFree(dWork);
	//////////////////////////////////////////////////////////////////////
	// Right Looking tile QR END
	//////////////////////////////////////////////////////////////////////
}



