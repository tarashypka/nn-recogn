#include <stdlib.h>
#include <time.h>

#include "nn_impl.h"

#define SEED	          time (NULL)

static int RNG = 0;    /* flag if rng was already initialized */

static void rng_init_ (void)
{
  srand (SEED);
  RNG = 1;
}

static const double_ rnd_gen_ (void)
{
  return ((double_) rand() / (double_) RAND_MAX);
}

void rnd_vec_gen (double_ *vec, const size_t n)
{
  if (! RNG)
    rng_init_();

  for (size_t i = 0; i < n; i++)
    vec[i] = rnd_gen_();
}

void rnd_mtx_gen (double_ **mtx, const size_t n, const size_t m)
{
  if (! RNG)
    rng_init_();

  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < m; j++)
      mtx[i][j] = rnd_gen_();
}
