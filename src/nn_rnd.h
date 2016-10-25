#ifndef _NN_RND_
#define _NN_RND_

/**
 *
 * @brief Initialize vector with Un([0,1]) random values
 *
 * @param vec       vector
 * @param n         # of elements
 *
 **/
void rnd_vec_gen (double_  *vec, const size_t n);

/**
 *
 * @brief Initialize matrix with Un([0,1]) random values
 *
 * @param mtx       matrix
 * @param n         # of rows
 * @param m         # of columns
 *
 **/
void rnd_mtx_gen (double_ **mtx, const size_t n, const size_t m);

#endif
