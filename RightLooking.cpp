/*
 * RightLooking.cpp
 *
 *  Created on: 2017/06/21
 *      Author: stomo
 */

//#define COUT

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <omp.h>

#include <plasma.h>
#include <core_blas.h>

#include "TileMatrix.hpp"

using namespace std;

void tileQR( TileMatrix *A, TileMatrix *T )
{
	const int MT = A->mt();
	const int NT = A->nt();

	double ttime = omp_get_wtime();

	//////////////////////////////////////////////////////////////////////
	// Right Looking tile QR Task version
	for (int tk=0; tk < min(MT,NT); tk++ )
	{
		{
			int nb = A->nb(tk,tk);
			double *Tau = new double [nb];
			double *Work = new double [nb*A->ib()];

			int info = core_dgeqrt( A->mb(tk,tk), A->nb(tk,tk), A->ib(),
					A->ttop(tk,tk), A->mb(tk,tk),
					T->ttop(tk,tk), T->mb(tk,tk),
					Tau, Work );
			if (info != PlasmaSuccess)
			{
				cerr << "core_dgeqrt() failed\n";
				exit (EXIT_FAILURE);
			}

			#ifdef COUT
			#pragma omp critical
			cout << omp_get_thread_num() << " : " << "GEQRT(" << tk << "," << tk << "," << tk << ") : " << omp_get_wtime() - ttime << "\n";
			#endif

			delete[] Tau;
			delete[] Work;
		}

		#pragma omp parallel for
		for (int tj=tk+1; tj < NT; tj++)
		{
			int nb = max( A->mb(tk,tk), T->mb(tk,tk) );
			double *Work = new double [nb*A->ib()];

			int info = core_dormqr( PlasmaLeft, PlasmaTrans,
					A->mb(tk,tj), A->nb(tk,tj), A->nb(tk,tk), A->ib(),
					A->ttop(tk,tk), A->mb(tk,tk),
					T->ttop(tk,tk), T->mb(tk,tk),
					A->ttop(tk,tj), A->mb(tk,tj),
					Work, nb );
			if (info != PlasmaSuccess)
			{
				cerr << "core_dormqr() failed\n";
				exit (EXIT_FAILURE);
			}

			#ifdef COUT
			#pragma omp critical
			cout << omp_get_thread_num() << " : " << "LARFB(" << tk << "," << tj << "," << tk << ") : " << omp_get_wtime() - ttime << "\n";
			#endif

			delete[] Work;
		}

		for (int ti=tk+1; ti < MT; ti++)
		{
			{
				assert( A->nb(tk,tk) == A->nb(ti,tk) );

				int nb = A->nb(ti,tk);
				double *Tau = new double [nb];
				double *Work = new double [nb*A->ib()];

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

				#ifdef COUT
				#pragma omp critical
				cout << omp_get_thread_num() << " : " << "TSQRT(" << ti << "," << tk << "," << tk << ") : " << omp_get_wtime() - ttime << "\n";
				#endif

				delete[] Tau;
				delete[] Work;
			}

			#pragma omp parallel for
			for (int tj=tk+1; tj < NT; tj++)
			{
				assert( A->nb(tk,tj) == A->nb(ti,tj) );

				double *Work = new double [ A->nb(tk,tj)*A->ib()];

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

				#ifdef COUT
				#pragma omp critical
				cout << omp_get_thread_num() << " : " << "SSRFB(" << ti << "," << tj << "," << tk << ") : " << omp_get_wtime() - ttime << "\n";
				#endif

				delete[] Work;

			} // j-LOOP END
		} // i-LOOP END
	} // k-LOOP END
	// Right Looking tile QR END
	//////////////////////////////////////////////////////////////////////
}
