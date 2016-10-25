#ifndef _NN_IMPL_
#define _NN_IMPL_

/**
 *
 * l_i - the i'th network layer 
 *
 *    |<-- hidden layers -->|
 *
 * b_0   b_1   b_2 ... b_n-1   b_n           <-- bias units
 * |   \ |   \ |         \ |   \ |     \ |
 * |    \|    \|          \|    \|      \|
 * *   --*   --*         --*   --*     --*   <-- activation units
 * |    /|    /|          /|    /|      /|       .
 * |   / |   / |         / |   / |     / |       .
 * l_0   l_1   l_2 ... l_n-1   l_n   l_n+1       .
 * |   \ |   \ |         \ |   \ |     \ |       .
 * |    \|    \|          \|    \|      \|       .
 * *   --*   --*         --*   --*     --*   <-- activation units
 * |    /|    /|          /|    /|      /|       .
 * |   / |   / |         / |   / |     / |       .
 * |     |<--- hidden layers --->|       |       .
 * input                            output
 * layer                             layer
 *
 **/

typedef struct nnetwork_* nnetwork;
typedef struct nnparams_* nnparams;
typedef double double_;

/**
 *
 * Distance between hypothesis and expected result,
 * is used to measure cost function
 *
 **/
typedef const double_ (*dist_f)(const double_, const double_);

/**
 *
 * @brief Initialize neural network with random Un([0,1]) weights
 *
 * @param ninpunits       # of units in  input layer
 * @param noutpunits      # of units in output layer
 * @param nhid            # of hidden layers
 * @param nhidunits       array of #nhid elements,
 *                        containts sizes for all hidden layers
 *
 * @return nnetwork struct with randomly initialized weights,
 *         it could be further passed to nn_costfunc(), nn_backprop()
 *
 **/
nnetwork 
nn_alloc (const size_t ninpunits, const size_t noutpunits,
          const size_t nhid,      const size_t *nhidunits);

/**
 *
 * @brief Initialize neural network parameters
 *
 * @param nexamples   # of examples that will be used to train the network
 * @param niters      # of iterations to backpropagate when training
 * @param learn_p     learning parameter
 * @param regur_p     regularization parameter, 0 if non-regularized
 * @param dist        distance function (see nn_costfunc())
 *
 * @return nnparams struct, it could be further passed to nn_backprop()
 *
 **/
nnparams
nn_alloc_nparams (const size_t nexamples, const size_t niters, 
                  const double_ learn_p,  const double_ regur_p, 
									const dist_f dist);

/**
 *
 * @brief Initialize network with already computed params
 *
 * @param netw      neural network
 * @param ws        matrices of weights for all hidden layers
 *
 * Assuming, l_i   - i'th layer, i = 0,1,2,...,n
 *           s_i   - # of (non-bias) units in i'th layer
 *           b_i   - i'th layer bias unit
 *           u_i_j - j'th unit in i'th layer, j = 1,2,...,s_i
 *
 * ws[i]      - matrix of s_i+1 rows and s_i + 1 columns
 * ws[i]      - matrix of weights between layers l_i and l_i+1
 * ws[i][j-1] - vector of weights between layer  l_i and u_i+1_j
 *
 * F.e. ws[2]       - matrix of weights that connects l_2   with l_3
 *      ww[2][3]    - vector of weights that connects l_2   with u_3_4
 *      ws[2][3][0] -           weights  that connects b_2   with u_3_4
 *      ws[2][3][4] -           weights  that connects u_2_4 with u_3_4
 *
 **/
void nn_weights_init (nnetwork netw, double_ ***ws);

/**
 *
 * @brief Free memory from 
 *          - nnetwork struct;
 *          - allocated weights for  input and hidden layers;
 *          - allocated   units for hidden and output layers; 
 *
 * @note The memory pointed to by the input units will not be freed,
 *       since it should be allocated outside and assigned
 *       to the network internally
 *
 **/
void nn_destroy (nnetwork netw);

/**
 *
 * @brief Free memory from nnparams struct pointed to by ps
 *
 **/
void nn_destroy_nparams (nnparams ps);

/**
 *
 * @brief Compute cost function of the network
 *
 *  J(W)    = 1/M * Sum (m, 1, nexamples, _distf_ (y_m,    h_m   ))
 *  _costf_ =       Sum (k, 1, nlabels,    distf  (y_m(k), h_m(k)))
 *  h_m     = hypof (W, x_m)
 *
 * Here, 
 *   W       - #
 *   M       - # of examples
 *   y_m (k) - expected result for m'th input example and k'th possible label
 *   h_m (k) - hypothesis      for m'th input example and k'th possible label
 *
 *   hypof (W, x)     - hypothesis function
 *   distf (y_m, h_m) - function, that determines the distance between
 *                      expected result and hypothesis
 *
 * Possible distance functions:
 *   distf (x, y) = (x - y)^2
 *   distf (x, y) =  x * log (y) + (1 - x) * log (1 - y)
 *
 * Possible hypothesis functions:
 *   hypof (W, x) = sigmoid (W*x) = 1 / (1 + exp (-W*x))
 *
 * @param netw      neural network
 * @param inps      inputs in training set
 * @param outps     expected outputs for each input
 * @param distf     distance function
 * @param ps        training parameters (see nn_alloc_params)
 *
 * @return cost function value for the network current wights
 *
 **/
const double_ 
nn_costfunc (nnetwork netw, double_ **inps, double_ **outps, nnparams ps);

/**
 *
 * @brief Functions that determine the distance between two values,
 *        could be used to find the difference between 
 *        hypothesis and expected result
 *
 * @param x   ~ expected result
 * @param y   ~ hypothesis
 *
 **/
const double_  sqdist (const double_ x, const double_ y);
const double_ logdist (const double_ x, const double_ y);

/**
 * @brief Modify network weights according to train set
 *        and backpropgation method
 *
 * @param netw      neural network
 * @param inps      inputs in training set
 * @param outps     expected outputs for each input
 * @param ps        training parameters (see nn_alloc_nparams())
 *
 **/
void 
nn_backprop (nnetwork netw, double_ **inps, double_ **outps, nnparams ps);

#endif
