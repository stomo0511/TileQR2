//
//  main.cpp
//
//  Created by T. Suzuki on 2017/06/21.
//
//#define DEBUG

//#define DEBUG  // To check residure norm and orthogonality
#define COUT   // To display the value of parameters
#define GPU    // To use GPUs

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <omp.h>

#include <plasma.h>
#include <core_blas.h>
#include <mkl_cblas.h>

#ifdef GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define GDEV_NUM 1
#define GDEV_ID  1
#endif

#include "TileMatrix.hpp"
#include "Check_Accuracy.hpp"

using namespace std;

extern void tileQR( TileMatrix *A, TileMatrix *T );

int main(int argc, const char * argv[])
{
	if (argc < 5)
	{
		cerr << "Usage: a.out [M] [N] [NB] [IB]\n";
		exit (EXIT_FAILURE);
	}
	
	const int M =  atoi(argv[1]);  // n. of rows of the matrix
	const int N =  atoi(argv[2]);  // n. of columns of the matrix
	const int NB = atoi(argv[3]);  // tile size
	const int IB = atoi(argv[4]);  // inner blocking size

	assert( M >= N );
	assert( NB >= IB );
	
	////////////////////////////////////////////////////////////////////////////
	// Definitions and Initialize
	TileMatrix *A = new TileMatrix(M,N,NB,NB,IB);
	
	const int MT = A->mt();
	const int NT = A->nt();
	
	#ifdef COUT
	cout << "M = " << M << ", N = " << N << ", NB = " << NB << ", IB = " << IB;
	cout << ", MT = " << MT << ", NT = " << NT << endl;
	#endif

	// refered in workspace.c of PLASMA
	TileMatrix *T = new TileMatrix(MT*IB,NT*NB,IB,NB,IB);
	
	// Initialize matrix A
	A->Set_Rnd( 20170621 );

	#ifdef DEBUG
	// Copy the elements of TMatrix class A to double array mA
	double *mA = new double [ M*N ];
	for (int i=0; i<M; i++)
		for (int j=0; j<N; j++)
			mA[ i+j*M ] = A->Get_Val(i,j);
	#endif
	// End of Definitions and Initialize

	////////////////////////////////////////////////////////////////////////////
	// Setting up GPUs
	#ifdef GPU
	cudaError_t		cuda_stat;

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
			return EXIT_FAILURE;
		}

		cuda_stat = cudaMalloc( (void**) &dAij[j], sizeof(double)*NB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dAij[" << j << "]\n";
			return EXIT_FAILURE;
		}

		cuda_stat = cudaMalloc( (void**) &dWork[j], sizeof(double)*IB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dwork[" << j << "]\n";
			return EXIT_FAILURE;
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
			return EXIT_FAILURE;
		}

		cuda_stat = cudaMalloc( (void**) &dTkk[i], sizeof(double)*IB*NB );
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Device memory allocate failure for dTkk[" << i << "]\n";
			return EXIT_FAILURE;
		}
	}

	cublasStatus_t	cublas_stat;
	cublasHandle_t *cublas_handle = new cublasHandle_t[NT];
	cudaStream_t   *cuda_stream   = new cudaStream_t[NT];

	// Create CUBLAS handle
	for (int j=0; j<NT; j++)
	{
		cublas_stat = cublasCreate(&cublas_handle[j]);
		if ( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << j << "-th CUBLAS initialization failed\n";
			return EXIT_FAILURE;
		}
	}

	// Create CUBLAS stream
	for(int j=0; j<NT; j++)
	{
		cuda_stat = cudaStreamCreate(&cuda_stream[j]);
		if( cuda_stat != cudaSuccess )
		{
			cerr << "Stream create failure for stream[" << j << "]\n";
			return EXIT_FAILURE;
		}

		cublas_stat = cublasSetStream( cublas_handle[j], cuda_stream[j]);
		if ( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << j << "-th cublasSetStream failed\n";
			return EXIT_FAILURE;
		}
	}
	#endif
	// End of device memory allocateion and stream creation
	////////////////////////////////////////////////////////////////////////////

	// Timer start
	double time = omp_get_wtime();

	////////////////////////////////////////////////////////////////////////////
	// tile QR variants
	tileQR(A,T);
	////////////////////////////////////////////////////////////////////////////
	
	// Timer stop
	time = omp_get_wtime() - time;
	cout << M << ", " << NB << ", " << IB << ", " << time << endl;
	
	#ifdef DEBUG
	////////////////////////////////////////////////////////////////////////////
	// Regenerate Q
	TileMatrix *Q = new TileMatrix(M,M,NB,NB,IB);

	// Set to the identity matrix
	for (int i=0; i<M; i++)
		for (int j=0; j<N; j++)
		{
			double val = (i==j) ? 1.0 : 0.0;
			Q->Set_Val(i,j,val);
		}

	// Make Orthogonal matrix Q
	int qMT = Q->mt();
	for (int tk = qMT - 1; tk+1 >= 1; tk--)
	{
		for (int ti = qMT - 1; ti > tk; ti--)
		{
			#pragma omp parallel for
			for (int tj = tk; tj < qMT; tj++)
			{
				double *Work = new double [Q->ib()*Q->nb(tk,tj)];
				int info = core_dtsmqr( PlasmaLeft, PlasmaNoTrans,
						Q->mb(tk,tj), Q->nb(tk,tj), Q->mb(ti,tj), Q->nb(ti,tj), A->nb(ti,tk), A->ib(),
						Q->ttop(tk,tj), Q->mb(tk,tj),
						Q->ttop(ti,tj), Q->mb(ti,tj),
						A->ttop(ti,tk), A->mb(ti,tk),
						T->ttop(ti,tk), T->mb(ti,tk),
						Work,  Q->ib());
				if (info != PlasmaSuccess)
				{
					cerr << "core_dtsmqr() in DEBUG failed\n";
					exit (EXIT_FAILURE);
				}

				delete[] Work;
			}
		}
		#pragma omp parallel for
		for (int tj = tk; tj < qMT; tj++)
		{
			double *Work = new double [A->ib()*Q->nb(tk,tj)];

			int info = core_dormqr( PlasmaLeft, PlasmaNoTrans,
					Q->mb(tk,tj), Q->nb(tk,tj), min(A->mb(tk,tk),A->nb(tk,tk)), A->ib(),
					A->ttop(tk,tk), A->mb(tk,tk),
					T->ttop(tk,tk), T->mb(tk,tk),
					Q->ttop(tk,tj), Q->mb(tk,tj),
					Work, Q->nb(tk,tj));  // plasma-2.8.0
//					Work, A->ib());  // plasma-17.1
			if (info != PlasmaSuccess)
			{
				cerr << "core_dormqr() in DEBUG failed\n";
				exit (EXIT_FAILURE);
			}

			delete[] Work;
		}
	}
	// Regenerate Q END
	//////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////
	// Check Accuracy
	double *mQ = new double [ M*M ];
	double *mR = new double [ M*N ];

	// Copy from TileMatrix to Array
	for (int i=0; i<M; i++)
	{
		for (int j=0; j<M; j++)
			mQ[ i + j*Q->m() ] = Q->Get_Val(i,j);
		for (int j=0; j<N; j++)
			mR[ i + j*A->m() ] = (i <= j) ? A->Get_Val(i,j) : 0.0;
	}

	Check_Accuracy(M,N,mA,mQ,mR );
	// Check Accuracy END
	//////////////////////////////////////////////////////////////////////

	delete [] mA;
	delete [] mQ;
	delete [] mR;

	cout << "Done\n";
  	#endif

	delete A;
	delete T;

	//////////////////////////////////////////////////////////////////////
	#ifdef GPU
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

	return EXIT_SUCCESS;
}
