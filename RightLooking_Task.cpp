/*
 * RightLooking_Task.cpp
 *
 *  Created on: 2017/06/21
 *      Author: stomo
 */

#define DEBUG
#define COUT

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <omp.h>

#include <plasma.h>
#include <core_blas.h>

#include "TileMatrix.hpp"

using namespace std;

void tileQR( TileMatrix& A, TileMatrix& T )
{
	const int MT = A.mt();
	const int NT = A.nt();

	double ttime = omp_get_wtime();

	//////////////////////////////////////////////////////////////////////
	// Right Looking tile QR Task version
	#pragma omp parallel firstprivate(ttime)
	{
		#pragma omp single
		{
			for (int tk=0; tk < min(MT,NT); tk++ )
			{
				double *Akk_top = A.ttop(tk,tk);
				int Akk_m = A.mb(tk,tk);
				int Akk_n = A.nb(tk,tk);
				int ib = A.ib(tk,tk);

				double *Tkk_top = T.ttop(tk,tk);
				int Tkk_m = T.mb(tk,tk);

				#pragma omp task depend(inout:Akk_top[:Akk_m*Akk_n]) \
								 depend(out:Tkk_top[:Tkk_m*Akk_n])
				{
					double *Tau = new double [Akk_n];
					double *Work = new double [Akk_n*ib];

					int info = core_dgeqrt( Akk_m, Akk_n, ib,
							Akk_top, Akk_m,
							Tkk_top, Tkk_m,
							Tau, Work );
					if (info != PlasmaSuccess)
					{
						cerr << "core_dgeqrt() failed\n";
						exit (EXIT_FAILURE);
					}

					#ifdef COUT
					#pragma omp critical
					cout << "GEQRT(" << tk << "," << tk << "," << tk << ") : " << omp_get_thread_num() << " : " << omp_get_wtime() - ttime << "\n";
					#endif

					delete[] Tau;
					delete[] Work;
				}

				for (int tj=tk+1; tj < NT; tj++)
				{
					double *Akj_top = A.ttop(tk,tj);
					int Akj_m = A.mb(tk,tj);
					int Akj_n = A.nb(tk,tj);

					assert( Akk_m == Akj_m );

					#pragma omp task depend(in:Akk_top[:Akk_m*Akk_n]) \
									 depend(in:Tkk_top[:Tkk_m*Akk_n]) \
									 depend(inout:Akj_top[:Akj_m*Akj_n])
					{
						double *Work = new double [Akk_n*ib];

						int info = core_dormqr( PlasmaLeft, PlasmaTrans,
								Akk_m, Akk_n, min(Akk_m,Akk_n), ib,
								Akk_top, Akk_m,
								Tkk_top, Tkk_m,
								Akj_top, Akj_m,
								Work, ib);
						if (info != PlasmaSuccess)
						{
							cerr << "core_dormqr() failed\n";
							exit (EXIT_FAILURE);
						}

						#ifdef COUT
						#pragma omp critical
						cout << "LARFB(" << tk << "," << tj << "," << tk << ") : " << omp_get_thread_num() << " : " << omp_get_wtime() - ttime << "\n";
						#endif

						delete[] Work;
					}
				}

				for (int ti=tk+1; ti < MT; ti++)
				{
					double *Aik_top = A.ttop(ti,tk);
					int Aik_m = A.mb(ti,tk);
					int Aik_n = A.nb(ti,tk);

					double *Tik_top = T.ttop(ti,tk);
					int Tik_m = T.mb(ti,tk);

					#pragma omp task depend(inout:Akk_top[:Akk_m*Akk_n]) \
									 depend(inout:Aik_top[:Aik_m*Aik_n]) \
									 depend(out:Tik_top[:Tik_m*Aik_n])
					{
						double *Tau = new double [Aik_n];
						double *Work = new double [Aik_n*ib];

						int info = core_dtsqrt( Aik_m, Aik_n, ib,
						                Akk_top, Akk_m,
						                Aik_top, Aik_m,
						                Tik_top, Tik_m,
						                Tau, Work);
						if (info != PlasmaSuccess)
						{
							cerr << "core_dtsqrt() failed\n";
							exit (EXIT_FAILURE);
						}

						#ifdef COUT
						#pragma omp critical
						cout << "TSQRT(" << ti << "," << tk << "," << tk << ") : " << omp_get_thread_num() << " : " << omp_get_wtime() - ttime << "\n";
						#endif

						delete[] Tau;
						delete[] Work;
					}

					for (int tj=tk+1; tj < NT; tj++)
					{
						double *Akj_top = A.ttop(tk,tj);
						int Akj_m = A.mb(tk,tj);
						int Akj_n = A.nb(tk,tj);

						double *Aij_top = A.ttop(ti,tj);
						int Aij_m = A.mb(ti,tj);
						int Aij_n = A.nb(ti,tj);

						#pragma omp task depend(inout:Akj_top[:Akj_m*Akj_n]) \
										 depend(inout:Aij_top[:Aij_m*Aij_n]) \
										 depend(in:Aik_top[:Aik_m*Aik_n]) \
										 depend(in:Tik_top[:Tik_m*Aik_n])
						{
							double *Work = new double [Akj_n*ib];

							int info = core_dtsmqr( PlasmaLeft, PlasmaTrans,
				                                   Akj_m, Akj_n, Aij_m, Aij_n, Aik_n, ib,
				                                   Akj_top, Akj_m,
				                                   Aij_top, Aij_m,
				                                   Aik_top,  Aik_m,
				                                   Tik_top,  ib,
				                                   Work,  ib);

							#ifdef COUT
							#pragma omp critical
							cout << "SSRFB(" << ti << "," << tj << "," << tk << ") : " << omp_get_thread_num() << " : " << omp_get_wtime() - ttime << "\n";
							#endif

							delete[] Work;
						}
					} // j-LOOP END
				} // i-LOOP END
			} // k-LOOP END
		} // parallel section END
	}
	// Right Looking tile QR END
	//////////////////////////////////////////////////////////////////////

}
