#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "nn_impl.h"
#include "nn_rnd.h"
#include "nn_alloc.h"

#define BIAS_ACTIVATION   1.0   /* static bias unit activation value */

#define N_INP_LAYERS      1     /* # of  input layers */
#define N_OUTP_LAYERS     1     /* # of output layers */
#define N_BIAS            1     /* # of bias units in each layer */


/* ========================== STRUCTURES ============================= */

/**
 *
 * Amount of hidden layers in the network (here nhid)
 * and amounts of units in each hidden layer (here nunits)
 * define the neural network structure
 *
 **/

typedef struct nnlayer_ 
{
  struct nnlayer_ *prev;    /* previos layer, NULL if layer is input */
  struct nnlayer_ *next;    /*    next layer, NULL if layer is output */
  double_        *units;    /* non-bias units activation values */
  size_t         nunits;    /* # of units in layer */
  double_     **weights;    /* outcoming weights from units in this layer 
                                                   to units in next layer */
} nnlayer_;

typedef struct nnetwork_ 
{
  nnlayer_    *inp;       /*  input layer */
  nnlayer_   *outp;       /* output layer */
  double_ *expoutp;       /* expected result for particular input */
  size_t      nhid;       /* # of hidden layers */

} nnetwork_;

typedef struct nnparams_ 
{
  size_t nexamples;       /* # of training examples */
  size_t    niters;       /* # of iterations to backpropagate */
  double_  learn_p;       /*       learning parameter */
  double_  regur_p;       /* regularization parameter, 0 if non-regularized */
  dist_f      dist;       /* distance function */

} nnparams_;

/* ====================== NETWORK INITIALIZATION ======================== */

void nn_destroy (nnetwork_ *netw_p)
{
  /* Destroy input layer */
  nnlayer_ *inp  = netw_p->inp;
  nnlayer_ *next = inp->next;
  free_mtx (inp->weights, inp->next->nunits);
  free (inp);

  /* Destroy hidden layers */
  for (nnlayer_ *hid = next; hid != netw_p->outp; hid = next)
    {
      free_mtx (hid->weights, hid->next->nunits);
      free (hid->units);
      next = hid->next;
      free (hid);
    }

  /* Destroy output layer */
  free (netw_p->outp->units);
  free (netw_p->outp);

  free (netw_p);
  puts ("Network successfully destroyed");
}

void nn_destroy_nparams (nnparams_ *nparams_p)
{
  free (nparams_p);
}

static void nn_exit_ (nnetwork_ *netw_p)
{
  if (netw_p != NULL)
    nn_destroy (netw_p);
  fprintf (stderr, "nn_exit(): %s\n", "Could not allocate memory ...");
  exit (1);
}

static int nn_example_prop_ (nnetwork_ *netw_p, double *inp, double *outp)
{
  if (inp == NULL || outp == NULL)
    {
      fprintf (stderr, "nn_example_prop(): inp or outp is NULL\n");
      return 1;
    }
  netw_p->inp->units = inp;
  netw_p->expoutp = outp;
  return 0;
}

static void nn_alloc_layers_units_ (nnetwork_ *netw_p)
{
  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    {
      double_ *units;
      if ((units = malloc (curr->next->nunits * sizeof *units)) == NULL)
        nn_exit_ (netw_p);
      curr->next->units = units;
    }
}

static void nn_alloc_layers_weights_ (nnetwork_ *netw_p)
{
  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    {
      double_ **ws;
      size_t ncurr = curr->nunits;
      size_t nnext = curr->next->nunits;
      if ((ws = alloc_mtx (nnext, N_BIAS + ncurr, 0)) == NULL)
        nn_exit_ (netw_p);
      curr->weights = ws;
    }
}

static void 
nn_alloc_layers_ (nnetwork_ *netw_p, 
                 const size_t  ninpunits, 
                 const size_t *nhidunits, 
                 const size_t noutpunits)
{
  /* Allocate input layer */
  nnlayer_ *inp;
  if ((inp = malloc (sizeof *inp)) == NULL)
    nn_exit_ (netw_p);
  inp->prev = NULL;
  inp->nunits = ninpunits;

  nnlayer_ *prev = netw_p->inp = inp;

  /* Allocate hidden layers */
  nnlayer_ *curr;
  for (size_t i = 0; i < netw_p->nhid; i++)
    {
      if ((curr = malloc (sizeof *curr)) == NULL)
        nn_exit_ (netw_p);
      prev->next = curr;
      curr->prev = prev;
      curr->nunits = nhidunits[i];
      prev = curr;
    }

  /* Alocate output layer */
  nnlayer_ *outp;
  if ((outp = malloc (sizeof *outp)) == NULL)
    nn_exit_ (netw_p);

  outp->prev = prev;
  outp->nunits = noutpunits;
  prev->next = netw_p->outp = outp;

  /* Alocate units for hidden and output layers */
  nn_alloc_layers_units_ (netw_p);

  /* Allocate weights for input and hidden layers */
  nn_alloc_layers_weights_ (netw_p);
}

static void nn_rnd_weights_alloc_ (nnetwork_ *netw_p)
{
  /* Generate for input layer */
  nnlayer_ *inp = netw_p->inp;
  rnd_mtx_gen (inp->weights, inp->next->nunits, N_BIAS + inp->nunits);

  /* Generate for hidden layers */
  for (nnlayer_ *hid = inp->next; hid != netw_p->outp; hid = hid->next)
    rnd_mtx_gen (hid->weights, hid->next->nunits, N_BIAS + hid->nunits);
}

void nn_weights_init (nnetwork_ *netw_p, double_ ***ws)
{
  /* Input and hidden layers weights */
  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    curr->weights = *ws++;

  /* Output layer weights */
  netw_p->outp->weights = NULL;
}

nnetwork_ *
nn_alloc (const size_t ninpunits, const size_t noutpunits,
          const size_t nhid,      const size_t *nhidunits)
{
  nnetwork_ *netw_p;
  if ((netw_p = malloc (sizeof *netw_p)) == NULL)
    nn_exit_ (netw_p);

  netw_p->nhid = nhid;

  /* Allocate and define layers */
  nn_alloc_layers_ (netw_p, ninpunits, nhidunits, noutpunits);

  /* Set random Un([0,1]) weights */
  nn_rnd_weights_alloc_ (netw_p);

  return netw_p;
}

nnparams_ *
nn_alloc_nparams (const size_t nexamples, const size_t niters, 
                  const double_ learn_p,  const double_ regur_p, 
									const dist_f dist)
{
  nnparams_ *ps = malloc (sizeof *ps);
  ps->nexamples = nexamples;
  ps->niters    = niters;
  ps->learn_p   = learn_p;
  ps->regur_p   = regur_p;
  ps->dist      = dist;
  return ps;
}

/* ========================= COST FUNCTION ============================ */

static const double_ sigmoid_ (const double_ x)
{
  return 1 / (1 + exp (-x));
}

static const double_ sigmoid_grad_ (const double_ s)
{
  return s * (1 - s);
}

static void sigmoid_map_ (double_ *a_i, const size_t n)
{
  for (size_t j = 0; j < n; j++)
    a_i[j] = sigmoid_ (a_i[j]);
}

/**
 *
 * gsl/gsl_blas matrix multiplication alternative solution doesn't provide 
 * better performance when compared to straightforward multiplication
 *
 **/
static void linear_prop_ (nnlayer_ *lay)
{
  nnlayer_ *prev = lay->prev;

  for (size_t i = 0; i < lay->nunits; i++)
    {
      double_ unit_i = BIAS_ACTIVATION * prev->weights[i][0];
      for (size_t j = N_BIAS; j < lay->nunits; j++)
        unit_i += prev->units[j-1] * prev->weights[i][j];
      lay->units[i] = unit_i;
    }
}

static void feedforward_ (nnlayer_ *lay)
{
  linear_prop_ (lay);
  sigmoid_map_ (lay->units, lay->nunits);
}

static void compute_hypotheses_ (nnetwork_ *netw_p)
{
  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    feedforward_ (curr->next);
}

static const double_ costfunc_example (
    nnetwork_ *netw_p, dist_f dist)
{
  /* Feedforward propagation: set output layer units activations */
  compute_hypotheses_ (netw_p);

  double_ cost = 0.0;
  for (size_t k = 0; k < netw_p->outp->nunits; k++)
    {
      double_ y_k = netw_p->expoutp[k];
      double_ h_k = netw_p->outp->units[k];
      cost -= dist (y_k, h_k);
    }
  return cost;
}

static const double_ nn_regur_ (nnetwork_ *netw_p)
{
  int regur = 0.0;

  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    {
      size_t ncurr = curr->nunits;
      size_t nnext = curr->next->nunits;
      for (size_t i = 0; i < nnext; i++)   
        /* don't regularize bias unit */
        for (size_t j = N_BIAS; j < N_BIAS + ncurr; j++)
          {
            double_ weight_i_j = curr->weights[i][j];
            regur += weight_i_j * weight_i_j;
          }
    }
  return regur / 2;
}

const double_ sqdist (const double_ x, const double_ y)
{
  return (x-y)*(x-y);
}

const double_ logdist (const double_ x, const double_ y)
{
  return x * log (y) + (1-x) * log (1-y);
}

const double_ 
nn_costfunc (nnetwork_ *netw_p, double_ **inps, double_ **outps, 
             nnparams_ *nparams_p)
{
  double_ cost = 0.0;

  for (size_t i = 0; i < nparams_p->nexamples; i++)
    {
      if (nn_example_prop_ (netw_p, inps[i], outps[i]) != 0)
        continue;
      cost += costfunc_example (netw_p, nparams_p->dist);
    }

  if (nparams_p->regur_p > 0)
    cost += nparams_p->regur_p * nn_regur_ (netw_p);

  return cost / nparams_p->nexamples;
}

/* =================== BACKPROPAGATION AND GRADIENT ==================== */

static void compute_deltas_ (nnetwork_ *netw, double_ **deltas)
{
  size_t ndeltas = netw->nhid + N_OUTP_LAYERS;

  /* Set delta vector for output layer */
  for (size_t i = 0; i < netw->outp->nunits; i++)
    deltas[ndeltas-1][i] = netw->outp->units[i] - netw->expoutp[i];

  /* Set delta vectors for all hidden layers */
  nnlayer_ *curr = netw->outp->prev;
  for (size_t k = ndeltas-1; k > 0; k--, curr = curr->prev)
    for (size_t i = 0; i < curr->nunits; i++)
      {
        double_ deltas_i = 0.0;
        for (size_t j = 0; j < curr->next->nunits; j++)
          deltas_i += curr->weights[j][N_BIAS+i] * deltas[k][j];
        deltas[k-1][i] = deltas_i * sigmoid_grad_ (curr->units[i]);
      }
}

static void 
acc_dweights_ (nnetwork_ *netw, nnparams_ *nparams_p,
             double_ **deltas, double_ ***dweights)
{
  size_t ndweights = N_INP_LAYERS + netw->nhid;

  nnlayer_ *curr = netw->inp;
  for (size_t k = 0; k < ndweights; k++, curr = curr->next)
    for (size_t i = 0; i < curr->next->nunits; i++)
      for (size_t j = 0; j < curr->nunits; j++)
        {
          double_ dweight_j = deltas[k][i] * curr->units[j];
          if (nparams_p->regur_p > 0)
            dweight_j += nparams_p->regur_p * curr->weights[i][N_BIAS+j];
          dweights[k][i][j] += 
            nparams_p->learn_p * dweight_j / nparams_p->nexamples;
        }
}

static void 
backprop_iter_ (nnetwork_ *netw_p, double_ **inps, double_ **outps,
               nnparams_ *nparams_p, double_ **deltas, double_ ***dweights)
{
  /* Backpropagation */
  for (size_t m = 0; m < nparams_p->nexamples; m++)
    {
      if (nn_example_prop_ (netw_p, inps[m], outps[m]) != 0)
        continue;

      /* Feedforward propagation: set output layer units activations */
      compute_hypotheses_ (netw_p);

      /* Set delta values for all hidden and output layers */
      compute_deltas_ (netw_p, deltas);

      /* Accumulate dweights matrices according to computed deltas */
      acc_dweights_ (netw_p, nparams_p, deltas, dweights);
    }
}

static double_ **alloc_deltas_ (nnetwork_ *netw_p)
{
  size_t ndeltas = netw_p->nhid + N_OUTP_LAYERS;
  double_ **deltas = malloc (ndeltas * sizeof *deltas);
  size_t i = 0;

  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    if ((deltas[i++] = malloc (curr->nunits * sizeof *deltas[i])) == NULL)
      nn_exit_ (netw_p);
  return deltas;
}

static void free_deltas_ (nnetwork_ *netw_p, double_ **deltas)
{
  free_mtx (deltas, netw_p->nhid + N_OUTP_LAYERS);
}

static double_ ***alloc_dweights_ (nnetwork_ *netw_p)
{
  size_t ndweights = N_INP_LAYERS + netw_p->nhid;
  double_ ***dweights = malloc (ndweights * sizeof *dweights);
  size_t i = 0;

  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    {
      size_t nnext = curr->next->nunits;
      size_t ncurr = curr->nunits;
      if ((dweights[i++] = alloc_mtx (nnext, ncurr + N_BIAS, 1)) == NULL)
        nn_exit_ (netw_p);
    }
  return dweights;
}

static void free_dweights_ (nnetwork_ *netw_p, double_ ***dweights)
{
  double_ ***dweights_p = dweights;
  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    free_mtx (*dweights_p++, curr->next->nunits);
}

static void reset_weights_ (nnetwork_ *netw_p, double_ ***dweights)
{
  size_t k = 0;
  for (nnlayer_ *curr = netw_p->inp; curr != netw_p->outp; curr = curr->next)
    {
      size_t ncurr = curr->nunits;
      size_t nnext = curr->next->nunits;
      for (size_t i = 0; i < nnext; i++)
        for (size_t j = 0; j < N_BIAS + ncurr; j++)
          curr->weights[i][j] -= dweights[k][i][j];
      k++;
    }
}

void
nn_backprop (nnetwork_ *netw_p, double_ **inps, double_ **outps,
             nnparams_ *nparams_p)
{
  double_  **deltas   = alloc_deltas_   (netw_p);
  double_ ***dweights = alloc_dweights_ (netw_p);

  puts ("Training neural network ...");
  for (size_t i = 0; i < nparams_p->niters; i++)
    {
      printf ("Iteration %4ld | cost = %g\n",
               i+1, nn_costfunc (netw_p, inps, outps, nparams_p));

      /* Feedforward and then backpropagate to find dweights */
      backprop_iter_ (netw_p, inps, outps, nparams_p, deltas, dweights);

      /* Modify network weights according to computed dweights */
      reset_weights_ (netw_p, dweights);
    }

  /* Free memory from delta vectors and dweights matrices */
  free_deltas_  (netw_p, deltas);
  free_dweights_ (netw_p, dweights);
}
