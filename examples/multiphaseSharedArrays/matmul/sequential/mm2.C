#include "params.h"
#include <stdlib.h>

/*
  double A[M][K];
  double B[K][N];
  double C[M][N];
*/

main()
{
    double* A[M];
    double* B[K];
    double* C[M];
    int i, j, k, l;

    for (i=0; i<M; i++) {
        A[i] = (double *) malloc(K*sizeof(double));
        for (j=0; j<K; j++)
            A[i][j] = (double) i * 1000.0 + (double) j;
    }

    for (i=0; i<K; i++) {
        B[i] = (double *) malloc(N*sizeof(double));
        for (j=0; j<N; j++)
            B[i][j] = (double) i * 1000.0 + (double) j;
    }

    for (i=0; i<M; i++) {
        C[i] = (double *) malloc(N*sizeof(double));
        for (j=0; j<N; j++)
            C[i][j] = 0.0;
    }

    for (i=0; i<M; i++)
        for (j=0; j<N; j++)
            for (k=0; k<K; k++)
                C[i][j] += A[i][k] * B[k][j];

}
