#include <stdlib.h>
#include <stdio.h>

#include "nn_impl.h"

void free_mtx (double_ **mtx, const size_t n)
{
  for (size_t i = 0; i < n; i++)
    free (mtx[i]);
  free (mtx);
}

static const char *ALLOC_MTX_ERR_MSG[] =
  {
    "alloc_mtx(): could not allocate space for matrix"
  };
double_ **alloc_mtx (const size_t n, const size_t m, const int init)
{
  double_ **mtx;

  if ((mtx = malloc (n * sizeof *mtx)) == NULL)
    {
      fprintf (stderr, "%s (n=%ld)\n", ALLOC_MTX_ERR_MSG[0], n);
      return mtx;
    }

  for (size_t i = 0; i < n; i++)
    if ((mtx[i] = init ? calloc (m,  sizeof *mtx[i]) 
                       : malloc (m * sizeof *mtx[i])) == NULL)
      {
        fprintf (stderr, "%s (n=%ld, m=%ld)\n", ALLOC_MTX_ERR_MSG[0], n, m);
        free_mtx (mtx, i);
        return NULL;
      }

  return mtx;
}
