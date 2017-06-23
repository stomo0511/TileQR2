/*
 * TileMatrix.hpp
 *
 *  Created on: 2017/06/20
 *      Author: stomo
 */

#ifndef TILEMATRIX_HPP_
#define TILEMATRIX_HPP_

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
	TileMatrix(	const int m, const int n );

	/**
	 * Constructor
	 *
	 * @param m number of lows of the matrix
	 * @param n number of columns of the matrix
	 * @param mb number of lows of the tile
	 * @param nb number of columns of the tile
	 */
	TileMatrix(	const int m, const int n, const int mb, const int nb, const int ib );

	/**
	 * Destructor
	 */
	~TileMatrix();

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
	 * return pointer to the top address of tile (ti,tj)
	 *
	 * @param ti tile index
	 * @param tj tile index
	 *
	 * @return pointer to the top address of tile (ti,tj)
	 */
	double* ttop( const int ti, const int tj ) const;

	/*
	 * get mb of (ti,tj) tile
	 */
	int mb( const int ti, const int tj ) const;

	/*
	 * get nb of (ti,tj) tile
	 */
	int nb( const int ti, const int tj ) const;

	/*
	 *  Assign random numbers to the elements
	 *  @param seed seed of random number generator
	 */
	void Set_Rnd( const unsigned seed );

	// Set value
	void Set_Val( const int i, const int j, const double val );

	// Get value
	double Get_Val( const int i, const int j ) const;

	/*
	 * Display all elements
	 */
	void Show_all() const;

	/*
	 * Display (i,j) tile elements
	 */
	void Show_tile(const int i, const int j) const;
};

#endif /* TILEMATRIX_HPP_ */
