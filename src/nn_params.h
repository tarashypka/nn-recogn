#ifndef _NN_PARAMS_
#define _NN_PARAMS_

#include "nn_impl.h"

/**
 *
 * Train
 *   - 1 network  in 1 thread
 *   - n networks in n threads
 *
 * To use thread pool, copy thpool.h and thpool.c 
 * from https://github.com/Pithikos/C-Thread-Pool to ../lib
 *
 **/
#define WITH_THPOOL           0

/* ========================== NEURAL NETWORK 1 ============================= */

#define N1_LEARN_PARAM        0.003
#define N1_REGUR_PARAM        0
#define N1_NITERS             50

#define N1_NEXAMPLES          10000
#define N1_NFEATURES          20*20
#define N1_NLABELS            10

#define N1_NINP_UNITS         N1_NFEATURES
#define N1_NOUTP_UNITS        N1_NLABELS

#define N1_NHID_LAYERS        1
#define N1_DIST_FUNC	      logdist

/* If HIDL_SIZES array is empty, then NHID_LAYERS should be 0 */
const size_t N1_NHID_UNITS[N1_NHID_LAYERS] = { 20 };

#if WITH_THPOOL

  #define NTHREADS      4           /* # of logical cores */
  #define NNETWORKS     NTHREADS

  /* ========================= NEURAL NETWORK 2 ============================ */

  #define N2_LEARN_PARAM        0.1
  #define N2_REGUR_PARAM        1
  #define N2_NITERS             50

  #define N2_NEXAMPLES          10000
  #define N2_NFEATURES          20*20
  #define N2_NLABELS            10

  #define N2_NINP_UNITS         N2_NFEATURES
  #define N2_NOUTP_UNITS        N2_NLABELS

  #define N2_NHID_LAYERS        1
  #define N2_DIST_FUNC		sqdist

  const size_t N2_NHID_UNITS[N2_NHID_LAYERS] = { 25 };

  /* ========================= NEURAL NETWORK 3 ============================ */

  #define N3_LEARN_PARAM        0.1
  #define N3_REGUR_PARAM        2
  #define N3_NITERS             50

  #define N3_NEXAMPLES          10000
  #define N3_NFEATURES          20*20
  #define N3_NLABELS            10

  #define N3_NINP_UNITS         N3_NFEATURES
  #define N3_NOUTP_UNITS        N3_NLABELS

  #define N3_NHID_LAYERS        1
  #define N3_DIST_FUNC		sqdist

  const size_t N3_NHID_UNITS[N3_NHID_LAYERS] = { 30 };

  /* ========================= NEURAL NETWORK 4 ============================ */

  #define N4_LEARN_PARAM        0.1
  #define N4_REGUR_PARAM        3
  #define N4_NITERS             50

  #define N4_NEXAMPLES          10000
  #define N4_NFEATURES          20*20
  #define N4_NLABELS            10

  #define N4_NINP_UNITS         N4_NFEATURES
  #define N4_NOUTP_UNITS        N4_NLABELS
  
  #define N4_NHID_LAYERS        1
  #define N4_DIST_FUNC		logdist

  const size_t N4_NHID_UNITS[N4_NHID_LAYERS] = { 35 };

  /* ========================== HELPER ARRAYS ============================== */

  const size_t LEARN_PARAMS[NNETWORKS] = 
    { 
      N1_LEARN_PARAM, 
      N2_LEARN_PARAM, 
      N3_LEARN_PARAM, 
      N4_LEARN_PARAM 
    };

  const size_t REGUR_PARAMS[NNETWORKS] =
    {
      N1_REGUR_PARAM,
      N2_REGUR_PARAM,
      N3_REGUR_PARAM,
      N4_REGUR_PARAM
    };

  const size_t NITERS[NNETWORKS] =
    {
      N1_NITERS,
      N2_NITERS,
      N3_NITERS,
      N4_NITERS
    };

  const size_t NEXAMPLES[NNETWORKS] =
    {
      N1_NEXAMPLES,
      N2_NEXAMPLES,
      N3_NEXAMPLES,
      N4_NEXAMPLES
    };

  const size_t NFEATURES[NNETWORKS] =
    {
      N1_NFEATURES,
      N2_NFEATURES,
      N3_NFEATURES,
      N4_NFEATURES
    };

  const size_t NLABELS[NNETWORKS] =
    {
      N1_NLABELS,
      N2_NLABELS,
      N3_NLABELS,
      N4_NLABELS
    };

  const size_t NINP_UNITS[NNETWORKS] =
    {
      N1_NINP_UNITS,
      N2_NINP_UNITS,
      N3_NINP_UNITS,
      N4_NINP_UNITS
    };

  const size_t NOUTP_UNITS[NNETWORKS] =
    {
      N1_NOUTP_UNITS,
      N2_NOUTP_UNITS,
      N3_NOUTP_UNITS,
      N4_NOUTP_UNITS
    };

  const size_t NHID_LAYERS[NNETWORKS] =
    {
      N1_NHID_LAYERS,
      N2_NHID_LAYERS,
      N3_NHID_LAYERS,
      N4_NHID_LAYERS
    };

  const size_t *NHID_UNITS[NNETWORKS] =
    {
      N1_NHID_UNITS,
      N2_NHID_UNITS,
      N3_NHID_UNITS,
      N4_NHID_UNITS
    };

  const dist_f DIST_FUNCS[NNETWORKS] =
    {
      N1_DIST_FUNC,
      N2_DIST_FUNC,
      N3_DIST_FUNC,
      N4_DIST_FUNC
    };

#else

  const size_t LEARN_PARAMS[1] = { N1_LEARN_PARAM };
  const size_t REGUR_PARAMS[1] = { N1_REGUR_PARAM };
  const size_t       NITERS[1] = { N1_NITERS };
  const size_t    NEXAMPLES[1] = { N1_NEXAMPLES };
  const size_t    NFEATURES[1] = { N1_NFEATURES };
  const size_t      NLABELS[1] = { N1_NLABELS };
  const size_t   NINP_UNITS[1] = { N1_NINP_UNITS };
  const size_t  NOUTP_UNITS[1] = { N1_NOUTP_UNITS };
  const size_t  NHID_LAYERS[1] = { N1_NHID_LAYERS };
  const size_t  *NHID_UNITS[1] = { N1_NHID_UNITS };
  const dist_f   DIST_FUNCS[1] = { N1_DIST_FUNC };

#endif

#endif
