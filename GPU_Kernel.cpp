#include "GPU_Kernel.hpp"

#include <iostream>
#include <algorithm>

const double one = 1.0;
const double mone = -1.0;
const double zero = 0.0;

using namespace std;

void GPU_dLARFB(cublasHandle_t& handle, int M, int N, int K, int ib,
                double* dAkk, int lda,
                double* dTkk, int ldt,
                double* dAkj, int ldc,
                double* dwork, int ldw)
{
	cublasStatus_t	cublas_stat;
	int kb, m, Mib;

	for(int i=0; i<K; i+=ib)
	{

		//適用するサイズ
		kb = min( ib, max(0,K-i));

		m = M-i;

		Mib = m-ib;

		//W = (V1)**T * (A1)
		cublas_stat = cublasDtrmm(handle,
				CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
				CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
				kb, N,               // m, n
				&one,                // alpha
				&dAkk[lda*i+i], lda, // *A, lda
				&dAkj[i],ldc,        // *B, ldb
				dwork,ldw);          // *C, ldc
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasDtrmm 1 in LARFB failure\n";
			cublasDestroy(handle);
			exit(EXIT_FAILURE);
		}

		//下の部分がある時
		if( m > ib )
		{
			//W = V2**T * A2 + W
			cublas_stat = cublasDgemm(handle,
					CUBLAS_OP_T, CUBLAS_OP_N,  // transA, transB
					kb, N, Mib,                // m, n, k
					&one,                      // alpha
					&dAkk[lda*i+i+kb], lda,    // *A, lda
					&dAkj[i+kb],ldc,           // *B, ldb
					&one,                      // beta
					dwork, ldw);               // *C, ldc
			if( cublas_stat != CUBLAS_STATUS_SUCCESS)
			{
				cerr << "cublasDgemm 1 in LARFB failure\n";
				cublasDestroy(handle);
				exit(EXIT_FAILURE);
			}
		}

		//W = T*W
		cublas_stat = cublasDtrmm(handle,
				CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
				CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                kb, N,             // m, n
                &one,              // alpha
                &dTkk[ldt*i], ldt, // *A, lda
                dwork, ldw,        // *B, ldb
                dwork, ldw);       // *C, ldc
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasDtrmm 2 in LARFB failure\n";
			cublasDestroy(handle);
			exit(EXIT_FAILURE);
		}

		//Akj - V * W

		//(A1) = (A1) - (V1) * W
		//(A2)   (A2) - (V2)

		//A2 = A2 - V2 * W
		if( m > ib )
		{
			cublas_stat = cublasDgemm(handle,
					CUBLAS_OP_N, CUBLAS_OP_N, // transA, transB
					Mib, N, kb,               // m, n, k
					&mone,                    // alpha
					&dAkk[lda*i+i+kb], lda,   // *A, lda
					dwork, ldw,               // *B, ldb
					&one,                     // beta
					&dAkj[i+kb], ldc);        // *C, ldc
			if( cublas_stat != CUBLAS_STATUS_SUCCESS)
			{
				cerr << "cublasDgemm 2 in LARFB failure\n";
				cublasDestroy(handle);
				exit(EXIT_FAILURE);
			}
		}

		//W = V1 * W
		cublas_stat = cublasDtrmm(handle,
				CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
				CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
				kb, N,               // m,n
				&one,                // alpha
				&dAkk[lda*i+i], lda, // *A, lda
				dwork, ldw,          // *B, ldb
				dwork, ldw);         // *C, ldc
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasDtrmm 3 in LARFB failure\n";
			cublasDestroy(handle);
			exit(EXIT_FAILURE);
		}

		//A1 = A1 - W
		cublas_stat = cublasDgeam(handle,
				CUBLAS_OP_N, CUBLAS_OP_N, // transA, transB
				kb, N,                    // m, n
				&one,                     // alpha
				&dAkj[i], ldc,            // *A, lda
				&mone,                    // beta
				dwork, ldw,               // *B, ldb
				&dAkj[i], ldc);           // *C, ldc
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasDgeam in LARFB failure\n";
			cublasDestroy(handle);
			exit(EXIT_FAILURE);
		}
	}
}

void GPU_dSSRFB(cublasHandle_t& handle, int M1, int N1,
                int M2, int N2, int K,int ib,
                double* dAkj, int lda1, double* dAij, int lda2,
                double* dAik, int ldv, double* dTik, int ldt,
                double* dwork, int ldw)
{
	cublasStatus_t	cublas_stat;
	int kb;

	for(int i=0; i<K; i+=ib)
	{
		kb = min( ib,K-i);

		//dwork に Akjの kb*N1をコピー
		// for(int j=0; j<kb; ++j)
		//   cublasDcopy(handle, N1, &dAkj[ i+j ], lda1, &dwork[j], ldw);
		cublas_stat = cublasDgeam(handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				kb, N1,
				&one,
				&dAkj[i], lda1,
				&zero,
				dwork, ldw,
				dwork, ldw);
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
				{
					cerr << "cublasDgeam in SSRFB failure\n";
					cublasDestroy(handle);
					exit(EXIT_FAILURE);
				}

		//dwork = V**T * dAij + dAkj
		cublas_stat = cublasDgemm(handle,
				CUBLAS_OP_T, CUBLAS_OP_N,
				kb, N1, M2,
				&one,
				&dAik[ ldv*i ], ldv,
				dAij, lda2,
				&one,
				dwork, ldw);
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasDgemm 1 in SSRFB failure\n";
			cublasDestroy(handle);
			exit(EXIT_FAILURE);
		}

		//dAkj = dAkj - T**T * dwork
		cublas_stat = cublasDgemm(handle,
				CUBLAS_OP_T, CUBLAS_OP_N,
				kb, N1, kb,
				&mone,
				&dTik[ ldt*i ], ldt,
				dwork, ldw,
				&one,
				&dAkj[i], lda1);
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasDgemm 2 in SSRFB failure\n";
			cublasDestroy(handle);
			exit(EXIT_FAILURE);
		}

		//dAij = dAij - V * T**T * dwork
		//dwork = T**T * dwork
		cublas_stat = cublasDtrmm(handle,
				CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
				CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
				kb, N1,
				&one,
				&dTik[ ldt*i ], ldt,
				dwork, ldw,
				dwork, ldw);
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasDtrmm in SSRFB failure\n";
			cublasDestroy(handle);
			exit(EXIT_FAILURE);
		}

		//dAij = dAij - V * dwork
		cublasDgemm(handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				M2, N2, kb,
				&mone,
				&dAik[ ldv*i ], ldv,
				dwork, ldw,
				&one,
				dAij, lda2);
		if( cublas_stat != CUBLAS_STATUS_SUCCESS)
		{
			cerr << "cublasDgemm 3 in SSRFB failure\n";
			cublasDestroy(handle);
			exit(EXIT_FAILURE);
		}
	}
}
