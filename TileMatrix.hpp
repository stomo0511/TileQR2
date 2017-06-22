/*
 * TileMatrix.hpp
 *
 *  Created on: 2017/06/20
 *      Author: stomo
 */

#ifndef TILEMATRIX_HPP_
#define TILEMATRIX_HPP_

#include <cassert>
#include <algorithm>

using namespace std;

/**
 *  @class TileMatrix
 */
class TileMatrix
{
private:
	double *top_;		// pointer for top address
	const int m_;		// number of lows of the matrix
	const int n_;		// number of columns of the matrix
	int mb_;			// number of lows of the tile
	int nb_;			// number of columns of the tile
	int mt_;			// number of low tiles
	int nt_;			// number of column tiles
	int ib_;			// inner block size

public:
	/**
	 * Constructor
	 *
	 * @param m number of lows of the matrix
	 * @param n number of columns of the matrix
	 */
	TileMatrix(	const int m, const int n )
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
	TileMatrix(	const int m, const int n, const int mb, const int nb, const int ib )
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
	~TileMatrix()
	{
		delete [] top_;
	}

	/*
	 * Getters
	 */
	double* top() { return top_; }
	int m() const { return m_; }
	int n() const { return n_; }
	int mt() const { return mt_; }
	int nt() const { return nt_; }
	int ib() const { return ib_; }

	/*
	 * get mb of (ti,tj) tile
	 */
	int mb( const int ti, const int tj ) const
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
	int nb( const int ti, const int tj ) const
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
	void Set_Rnd( const unsigned seed )
	{
		assert( seed >= 0 );

		srand(seed);
		for (int i = 0; i < m_ * n_; i++)
			top_[i] = (double)rand() / RAND_MAX;
	}

	// Set value
	void Set_Val( const int i, const int j, const double val )
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
	double Get_Val( const int i, const int j ) const
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
	 * return pointer to the top address of tile (ti,tj)
	 *
	 * @param ti tile index
	 * @param tj tile index
	 *
	 * @return pointer to the top address of tile (ti,tj)
	 */
	double* ttop( const int ti, const int tj ) const
	{
		assert( ti >= 0 && ti < mt_ );
		assert( tj >= 0 && tj < nt_ );

		// column major x column major
		return top_ + ti* (mb_*nb_) + tj * (m_*nb_);
	}
};

#endif /* TILEMATRIX_HPP_ */
