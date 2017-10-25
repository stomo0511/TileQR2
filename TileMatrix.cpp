/*
 * TileMatrix.cpp
 *
 *  Created on: 2017/06/23
 *      Author: stomo
 */

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <algorithm>

#include "TileMatrix.hpp"

using namespace std;

/**
 * Constructor
 *
 * @param m number of lows of the matrix
 * @param n number of columns of the matrix
 */
TileMatrix::TileMatrix(	const int m, const int n )
: m_(m), n_(n), mb_(m), nb_(n), mt_(1), nt_(1), ib_(n)
{
	assert( m > 0 && n > 0 );

	try
	{
		top_ = new double[ m * n ];
	}
	catch (char *eb)
	{
		cerr << "Can't allocate memory space for TileMatrix class: " << eb << endl;
		exit(EXIT_FAILURE);
	}
}

/**
 * Constructor
 *
 * @param m number of lows of the matrix
 * @param n number of columns of the matrix
 * @param mb number of lows of the tile
 * @param nb number of columns of the tile
 */
TileMatrix::TileMatrix(	const int m, const int n, const int mb, const int nb, const int ib )
: m_(m), n_(n), mb_(mb), nb_(nb),
  mt_(m % mb == 0 ? m / mb : m / mb + 1),
  nt_(n % nb == 0 ? n / nb : n / nb + 1),
  ib_(ib)
{
	assert( m > 0 && n > 0 && mb > 0 && nb > 0 && ib > 0);
	assert( mb <= m && nb <= n );

	try
	{
		top_ = new double[ m * n ];
	}
	catch (char *eb)
	{
		cerr << "Can't allocate memory space for TileMatrix class: " << eb << endl;
		exit(EXIT_FAILURE);
	}
}

/**
 * Destructor
 */
TileMatrix::~TileMatrix()
{
	delete [] top_;
}

/*
 * return pointer to the top address of tile (ti,tj)
 *
 * @param ti tile index
 * @param tj tile index
 *
 * @return pointer to the top address of tile (ti,tj)
 */
double* TileMatrix::ttop( const int ti, const int tj ) const
{
	assert( ti >= 0 && ti < mt_ );
	assert( tj >= 0 && tj < nt_ );

	// column major x column major
	int nb = TileMatrix::nb(ti,tj);
	return top_ + ti* (mb_*nb) + tj * (m_*nb_);
}

/*
 * get mb of (ti,tj) tile
 */
int TileMatrix::mb( const int ti, const int tj ) const
{
	assert( ti < mt_ && tj < nt_ );

	if ((m_ % mb_ != 0) && (ti == mt_ -1))
		return m_ % mb_;
	else
		return mb_;
}

/*
 * get nb of (ti,tj) tile
 */
int TileMatrix::nb( const int ti, const int tj ) const
{
	assert( ti < mt_ && tj < nt_ );

	if ((n_ % nb_ != 0) && (tj == nt_ -1))
		return n_ % nb_;
	else
		return nb_;
}

/*
 *  Assign random numbers to the elements
 *  @param seed seed of random number generator
 */
void TileMatrix::Set_Rnd( const unsigned seed )
{
	assert( seed >= 0 );

	srand(seed);
	for (int i = 0; i < m_ * n_; i++)
		top_[i] = (double)rand() / RAND_MAX;
}

// Set value
void TileMatrix::Set_Val( const int i, const int j, const double val )
{
	// 当該タイルの位置
	int ti = i/mb_;
	int tj = j/nb_;

	// 当該タイルのサイズ
	int mb = TileMatrix::mb(ti,tj);
	int nb = TileMatrix::nb(ti,tj);

	// グローバル配列の位置
	int pos = 0;

	pos += (mb_*nb)*ti + (i%mb_);     // i方向位置
	pos += (nb_*m_)*tj + (j%nb_)*mb;  // j方向位置

	top_[ pos ] = val;
}

// Get value
double TileMatrix::Get_Val( const int i, const int j ) const
{
	// 当該タイルの位置
	int ti = i/mb_;
	int tj = j/nb_;

	// 当該タイルのサイズ
	int mb = TileMatrix::mb(ti,tj);
	int nb = TileMatrix::nb(ti,tj);

	// グローバル配列の位置
	int pos = 0;

	pos += (mb_*nb)*ti + (i%mb_);     // i方向位置
	pos += (nb_*m_)*tj + (j%nb_)*mb;  // j方向位置

	return top_[ pos ];
}

/*
 * Display all elements
 */
void TileMatrix::Show_all() const
{
	for (int i=0; i<m_; i++)
	{
		for (int j=0; j<n_; j++)
			cout << TileMatrix::Get_Val(i,j) << ", ";
		cout << endl;
	}
	cout << endl;
}

/*
 * Display (i,j) tile elements
 */
void TileMatrix::Show_tile(const int ti, const int tj) const
{
	double *t = TileMatrix::ttop(ti,tj);
	const int mb = TileMatrix::mb(ti,tj);
	const int nb = TileMatrix::nb(ti,tj);

	for (int i=0; i<mb; i++) {
		for (int j=0; j<nb; j++)
			cout << t[i + mb*j] << ", ";
		cout << endl;
	}
	cout << endl;
}




