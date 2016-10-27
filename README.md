## nn-recogn

######*Simple but highly customizable neural network with backpropagation*

0. Dependencies

   * Copy `thpool.h` and `thpool.c` from [Pithikos/C-Thread-Pool](https://github.com/Pithikos/C-Thread-Pool) to `./lib` [1]

1. Configure desired network parameters in `./src/nn_params.h`

   * Alternatively, configure multiple neural networks to simultaneously train them
   using thread pool (`WITH_THPOOL`) [2]
   

2. Compile with gcc

   * Without thread pool
   ```
   $ gcc -Wall \
          -o ./build/nn.o \
          -g ./src/{nn.c,nn_impl.c,nn_alloc.c,nn_rnd.c} \
          -lm
   ```
   * With thread pool
    ```
    $ gcc -Wall \
          -o ./build/nn.o \
          -g ./src/{nn.c,nn_impl.c,nn_alloc.c,nn_rnd.c} ./lib/thpool.c \
          -lm -pthread
    ```

3. Train neural network(s)
    ```
    $ ./build/nn.o
    ```

[1] Another Thread pool for C ([mbrossard/threadpool](https://github.com/mbrossard/threadpool)) gives almost the same performance results.  
[2] The result of using 4 threads instead of one and training 4 neural networks simultaneously leads to ~2x increase in the watch time and ~2x decrease in the clock time. 
