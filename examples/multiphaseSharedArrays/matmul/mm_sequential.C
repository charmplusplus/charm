#include <sys/times.h>
/**
   The first algo compiles fine, but not on lemieux with g++ -O3 where
   it generates an 880 MB a.out and does not run.  The second algo
   compiles fine on lemieux also.

   I ran both algos on skill and times were almost the same.
 */

// #define BIGGER_ALGO
// #define GLOBAL_VAR_VERSION
// #define SKIP_INIT

// incomplete
double CmiWallTimer()
{
  struct tms temp;
  double currenttime;
  int now;

  now = times(&temp);
  return now;
}


#ifdef BIGGER_ALGO

#include <stdio.h>
#include <stdlib.h>

const unsigned int ROWS1=2000;
const unsigned int COLS1=5000;
const unsigned int ROWS2=COLS1;
const unsigned int COLS2=300;

// const unsigned int ROWS1=2048;
// const unsigned int COLS1=2048;
// const unsigned int ROWS2=COLS1;
// const unsigned int COLS2=2048;

// debugging
#define false 0;
#define true 1;
const int verbose = false;
const int do_test = false;  // If true, tests results, etc.

#ifdef GLOBAL_VAR_VERSION
  double arr1[ROWS1][COLS1];
  double arr2[ROWS2][COLS2];
  double prod[ROWS1][COLS2];

  void malloc_arrays(){}

#else
  double **arr1;
  double **arr2;
  double **prod;

  //================================================================
  // http://remus.rutgers.edu/cs314/s2004/decarlo/lec/twod.c
  
  /* Dynamically allocate a 2D "array" of integers, that is rows X cols
   * and accessed as arr[i][j] where i=0..rows-1, and j=0..cols-1
   *
   * It is stored as a 1D array of ROWS*COLS integers, with an array
   * of ROWS integer pointers which refer to the appropriate places in this
   * larger array
   */
  double **make2DarrayFlat(int rows, int cols)
  {
      int i;
      double **p, *base;
      /* Allocate array of pointers and 1D block of integers */
      p = (double **)malloc(rows * sizeof(double *));
      base = (double *)malloc(rows * cols * sizeof(double));    
      if (p == NULL || base == NULL)
          return NULL;
      /* Set pointers for each row */
      for (i = 0; i < rows; i++) {
          p[i] = &base[cols * i];
      }
      return p;
  }
  
  //================================================================
  
  void malloc_arrays()
  {
          arr1 = make2DarrayFlat(ROWS1, COLS1);
          arr2 = make2DarrayFlat(ROWS2, COLS2);
          prod = make2DarrayFlat(ROWS1, COLS2);
  }

#endif

void init_arrays()
{
	unsigned int i, j;

	for(i = 0; i < ROWS1; i++)
		for(j = 0; j < COLS1; j++)
			arr1[i][j] = 1.0;

	for(i = 0; i < ROWS2; i++)
		for(j = 0; j < COLS2; j++)
			arr2[i][j] = 1.0;

}

void multiply()
{
    if(verbose) printf("Multiplying\n");

    unsigned int i, j, k;
    for(j = 0; j < COLS2; j++)
	for(i = 0; i < ROWS1; i++)
            {
                double r = 0.0;
                for(k = 0; k < ROWS2; k++)
                    r += arr1[i][k] * arr2[k][j];
                prod[i][j] = r;
            }
}

void test_result()
{
    unsigned int i, j, k, msg;
    msg = 1;
    for(i = 0; i < ROWS1; i++)
        for(j = 0; j < COLS2; j++)
            if(msg && (prod[i][j] != 1.0*ROWS2))
                {
                    printf("Element [%d][%d] inconsistent\n", i, j);
                    msg = 0;
                }
}

int main()
{
#ifdef GLOBAL_VAR_VERSION
    printf("global_var ");
#else
    printf("malloc_version ");
#endif
#ifdef SKIP_INIT
    printf("skip_init ");
#else
    printf("do_init ");
#endif

    printf("%d %d %d\n", ROWS1, COLS1, COLS2);

    malloc_arrays();
#ifndef SKIP_INIT
    init_arrays();
#endif
    multiply();

    if (do_test)
        test_result();

    return 0;
}

#else

#include <stdio.h>

#define M 2000
#define K 5000
#define N 300

double A[M][K];
double B[K][N];
double C[M][N];

main()
{
  int i, j, k;

  /*
  // Check if values are auto-initialized to 0
  unsigned int numNonZero = 0;
  for (i=0; i<M; i++)
    for (k=0; k<K; k++)
      if (A[i][k] != 0)
          numNonZero++;

  for (k=0; k<K; k++)
    for (j=0; j<N; j++)
        if (B[k][j] != 0)
            numNonZero++;
  printf("Num non zero = %d\n", numNonZero);
  */

#ifndef SKIP_INIT
  for (i=0; i<M; i++)
    for (k=0; k<K; k++)
      A[i][k] = 1.0;

  for (k=0; k<K; k++)
    for (j=0; j<N; j++)
      B[k][j] = 1.0;
#endif

  for (i=0; i<M; i++)
    for (j=0; j<N; j++)
      for (k=0; k<K; k++)
        C[i][j] += A[i][k] * B[k][j];
}
#endif
