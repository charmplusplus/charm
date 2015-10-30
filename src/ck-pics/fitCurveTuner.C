#include <stdlib.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_poly.h>
#include <vector>
#include "fitCurveTuner.h"
using namespace std;

double FittingCurve::linearFitting(int nV, int n, double *rawx, double *rawy, vector < vector<double> > &minX)
{
    double *params = (double*)malloc(sizeof(double)*nV+1); 
    multiFitting(nV, n, rawx, rawy, 2, params);
}

double FittingCurve::quadraticFitting(int nV, int n, double *rawx, double *rawy, vector < vector<double> > &minX)
{
    double *params = (double*)malloc(2*sizeof(double)*nV+1); 
    int i;
    //vector<double> minX;
    multiFitting(nV, n, rawx, rawy, 3, params);
    for(i=0; i<nV; i++)
    {
        minX[i].push_back(-params[1+i*2]/(2*params[2+2*i]));
        printf("------#########----- min X  %f \n", -params[1+i*2]/(2*params[2+2*i] ));
    }
}

double FittingCurve::cubicFitting(int nV, int n, double *rawx, double *rawy, vector < vector<double> > &minX)
{
    //vector<double> minX;
    int i;
    double *params = (double*)malloc(3*sizeof(double)*nV+1);
    double *codev = (double*)malloc(3*sizeof(double));
    multiFitting(nV, n, rawx, rawy, 4, params);
    for(i=0; i<nV; i++)
    {
        codev[0] = params[1+i*3];
        codev[1] = 2*params[2+i*3];
        codev[2] = 3*params[3+i*3];
        minimumPolynomialValue(3, codev, minX[i]);
    }
    free(params);
    free(codev);
}

double FittingCurve::quarticFitting(int nV, int n, double *rawx, double *rawy, vector < vector<double> > &minX)
{
    //vector<double> minX;
    int i;
    double *params = (double*)malloc(4*sizeof(double)*nV+1);
    double *codev = (double*)malloc(4*sizeof(double));
    multiFitting(nV, n, rawx, rawy, 5, params);
    for(i=0; i<nV; i++)
    {
        codev[0] = params[1+i*4];
        codev[1] = 2*params[2+i*4];
        codev[2] = 3*params[3+i*4];
        codev[3] = 4*params[4+i*4];
        minimumPolynomialValue(4, codev, minX[i]);
    }
    free(params);
    free(codev);
}

// root   of polynomial n is the number of coeffiency 
double FittingCurve::minimumPolynomialValue(int n, double *a, vector<double>& roots)
{
    int i;
    double *z = (double*)malloc( sizeof(double)*(n-1)*2);
    gsl_poly_complex_workspace * w = gsl_poly_complex_workspace_alloc (n);

    gsl_poly_complex_solve (a, n, w, z);
    gsl_poly_complex_workspace_free (w);
    for (i = 0; i < n-1; i++)
    {
        printf ("z%d = %+.18f %+.18f\n", i, z[2*i], z[2*i+1]);
        //only keep the real values, no complex
        if(z[2*i+1] == 0)
        {
           roots.push_back(z[2*i]); 
        }
    }

    free(z);
}

double FittingCurve::multiFitting(int nV, int n, double *rawx, double *rawy, int dim, double *params)
{
    int i, j, k;
    double  chisq;
    gsl_matrix *X, *cov;
    gsl_vector *y, *w, *c;
    int mdim = 1+(dim-1)*nV;
    X = gsl_matrix_alloc (n, mdim);
    y = gsl_vector_alloc (n);
    w = gsl_vector_alloc (n);

    c = gsl_vector_alloc (mdim);
    cov = gsl_matrix_alloc (mdim, mdim);

    for(i=0; i<n; i++)
    {
        double mul = 1.0;
        gsl_matrix_set (X, i, 0, 1.0);
        for(k=0; k<nV; k++)
        {
            mul = 1.0;
            for(j=0; j<dim-1; j++)
            {
                mul *= rawx[k*n+i];
                gsl_matrix_set (X, i, 1+k*(dim-1)+j, mul);
            }
        }
        gsl_vector_set (y, i, rawy[i]);
    }

    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (n, mdim);
    gsl_multifit_linear (X, y, c, cov, &chisq, work);
    printf(" chisq is %f fitting function is ", chisq);
    gsl_multifit_linear_free (work);

#define C(i) (gsl_vector_get(c,(i)))
#define COV(i,j) (gsl_matrix_get(cov,(i),(j)))

    printf(" %g ", C(0));
    params[0] = C(0);
    for(i=0; i<nV; i++)
        for(j=0; j<dim-1; j++)
        {
            k = 1+i*(dim-1)+j;
            printf(" %g X^%d ", C(k), j+1);
            params[k] = C(k);
        }
    printf(" \n");
    //printf ("# covariance matrix:\n");
    //printf ("[ %+.5e, %+.5e, %+.5e  \n",
    //    COV(0,0), COV(0,1), COV(0,2));
    //printf ("  %+.5e, %+.5e, %+.5e  \n", 
    //           COV(1,0), COV(1,1), COV(1,2));
    //printf ("  %+.5e, %+.5e, %+.5e ]\n", 
    //           COV(2,0), COV(2,1), COV(2,2));
    //printf ("# chisq = %g\n", chisq);

    gsl_matrix_free (X);
    gsl_vector_free (y);
    gsl_vector_free (w);
    gsl_vector_free (c);
    gsl_matrix_free (cov);

}

#if 0
double FittingCurve::minimize()
{
    int status;
    int iter = 0, max_iter = 100;
    const gsl_min_fminimizer_type *T;
    gsl_min_fminimizer *s;
    double m = 2.0, m_expected = M_PI;
    double a = 0.0, b = 6.0;
    gsl_function F;

    F.function = &fn1;
    F.params = 0;

    T = gsl_min_fminimizer_brent;
    s = gsl_min_fminimizer_alloc (T);
    gsl_min_fminimizer_set (s, &F, m, a, b);

    printf ("using %s method\n",
        gsl_min_fminimizer_name (s));

    printf ("%5s [%9s, %9s] %9s %10s %9s\n",
      "iter", "lower", "upper", "min",
          "err", "err(est)");

    printf ("%5d [%.7f, %.7f] %.7f %+.7f %.7f\n",
        iter, a, b,
          m, m - m_expected, b - a);

    do
    {
      iter++;
      status = gsl_min_fminimizer_iterate (s);

      m = gsl_min_fminimizer_x_minimum (s);
      a = gsl_min_fminimizer_x_lower (s);
      b = gsl_min_fminimizer_x_upper (s);

      status 
        = gsl_min_test_interval (a, b, 0.001, 0.0);

      if (status == GSL_SUCCESS)
        printf ("Converged:\n");

      printf ("%5d [%.7f, %.7f] "
              "%.7f %+.7f %.7f\n",
              iter, a, b,
              m, m - m_expected, b - a);
    }
  while (status == GSL_CONTINUE && iter < max_iter);

  gsl_min_fminimizer_free (s);

  return status; 
}

#endif
