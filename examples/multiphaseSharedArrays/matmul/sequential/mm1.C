#include "params.h"

double A[M][K];
double B[K][N];
double C[M][N];

main()
{
  int i, j, k;

  for (i=0; i<M; i++)
    for (k=0; k<K; k++)
      A[i][k] = 1.0;

  for (k=0; k<K; k++)
    for (j=0; j<N; j++)
      B[k][j] = 1.0;

  for (i=0; i<M; i++)
    for (j=0; j<N; j++)
      for (k=0; k<K; k++)
        C[i][j] += A[i][k] * B[k][j];
}
