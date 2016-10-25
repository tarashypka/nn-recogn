#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "nn_impl.h"
#include "nn_rnd.h"
#include "nn_alloc.h"
#include "nn_params.h"

#if WITH_THPOOL
#include "../lib/thpool.h"
#endif

/**
 *
 * Functions that set input/expected output for the network
 *
 * It could be functions that will 
 *  - parse a data set from the file
 *  - fetch a data set from the db
 *  ...
 *
 * Here, rnd_mtx_gen is a stub function 
 * that will generate random input/expected output
 *
 **/
static void (*SETINP) (double_ **, const size_t, const size_t) = &rnd_mtx_gen;
static void (*SETOUTP)(double_ **, const size_t, const size_t) = &rnd_mtx_gen;

int main (void)
{
  clock_t cbegin, cend;
  time_t  tbegin, tend;
  void train_networks_ (void);

  cbegin = clock();
  time (&tbegin);

  train_networks_();

  cend = clock();
  time (&tend);

  float cseconds = (float)          (cend - cbegin) / CLOCKS_PER_SEC,
        tseconds = (float) difftime (tend,  tbegin);

  printf ("All jobs were completed and it took %.1f cloak secs\n", cseconds);
  printf ("All jobs were completed and it took %.1f watch secs\n", tseconds);
}

static void main_exit_ (void)
{
  fprintf (stderr, "main(): %s\n", "Could not allocate memory ...");
  exit (1);
}

static double_ **getinp_ (
    const size_t nexamples, const size_t nfeatures,
    void (*setinp)(double_ **, const size_t, const size_t))
{
  double_ **inp;

  /* Generate random input values */
  if ((inp = alloc_mtx (nexamples, nfeatures, 0)) == NULL)
    main_exit_();

  setinp (inp, nexamples, nfeatures);
  return inp;
}

static double_ **getoutp_ (
    const size_t nexamples, const size_t nlabels,
    void (*setoutp)(double_ **, const size_t, const size_t))
{
  double_ **outp;

  /* Generate random output values */
  if ((outp = alloc_mtx (nexamples, nlabels, 0)) == NULL)
    main_exit_();

  setoutp (outp, nexamples, nlabels);
  return outp;
}

typedef struct backprop_params_
{
  size_t        id;
  nnetwork    netw;
  double_    **inp;
  double_   **outp;
  nnparams nparams;

} bprop_params_;

static bprop_params_ *alloc_bparams_ (const size_t i)
{
  bprop_params_ *bs = malloc (sizeof *bs);
  bs->id   = i;
  bs->netw = nn_alloc (
      NINP_UNITS[i], NOUTP_UNITS[i], NHID_LAYERS[i], NHID_UNITS[i]);
  bs->inp  = getinp_  (NEXAMPLES[i], NFEATURES[i], SETINP);
  bs->outp = getoutp_ (NEXAMPLES[i], NLABELS[i],   SETOUTP);
  bs->nparams = nn_alloc_nparams (
      NEXAMPLES[i], NITERS[i], LEARN_PARAMS[i], REGUR_PARAMS[i], DIST_FUNCS[i]);
  return bs;
}

static void free_bparams_ (bprop_params_ *bs)
{
  nn_destroy         (bs->netw);
  nn_destroy_nparams (bs->nparams);
  free_mtx (bs->inp,  NEXAMPLES[bs->id]);
  free_mtx (bs->outp, NEXAMPLES[bs->id]);
  free (bs);
}

static void backprop_ (void *bparams)
{
  bprop_params_ *bs = (bprop_params_ *)bparams;
  nn_backprop (bs->netw, bs->inp, bs->outp, bs->nparams);
}

void train_networks_ (void)
{
  /* Train network 1 */
  bprop_params_ *bs0 = alloc_bparams_ (0);

  #if WITH_THPOOL
    const threadpool thpool = thpool_init (NTHREADS);

    /* Train networks 2, 3, ... */
    bprop_params_ *bs[NNETWORKS];
    bs[0] = bs0;

    for (size_t i = 0; i < NNETWORKS; i++)
      {
        bs[i] = alloc_bparams_ (i);

        /* Add new job to the thread pool */
        thpool_add_work (thpool, &backprop_, (void *)bs[i]);
        printf ("[%ld]: Added new job ...\n", i);
      }

    /* Wait for thread pool to finish all jobs */
    thpool_wait (thpool);

    /* Free all resources */
    for (size_t i = 0; i < NNETWORKS; i++)
      free_bparams_ (bs[i]);

    thpool_destroy (thpool);
  #else
    nn_backprop (bs0->netw, bs0->inp, bs0->outp, bs0->nparams);
    free_bparams_ (bs0);
  #endif
}
