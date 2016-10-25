#ifndef _NN_ALLOC_
#define _NN_ALLOC_

/**
 *
 * @brief Allocate/free space for/from matrix of n rows and m columns
 *
 * @param n       # of rows
 * @param m       # of columns
 * @param init    1 to initialize matrix with 0.0 values
 *
 * @return pointer to the mtx[0][0]
 *
 **/
double_ **alloc_mtx (const size_t n, const size_t m, const int init);
void       free_mtx (double_ **mtx, const size_t n);

#endif
