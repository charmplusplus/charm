#ifndef __FITTINGCURVE__H__
#define __FITTINGCURVE__H__
#include <vector>
using namespace std;
class FittingCurve{
public:
    static double linearFitting(int nV, int n, double *rawx, double *rawy, vector < vector<double> > &minX);
    static double quadraticFitting(int nV, int n, double *rawx, double *rawy, vector < vector<double> > &minX);
    static double cubicFitting(int nV, int n, double *rawx, double *rawy, vector< vector<double> > &minX);
    static double quarticFitting(int nV, int n, double *rawx, double *rawy, vector< vector<double> > &minX);
    static double minimumPolynomialValue(int n, double *a, std::vector<double>& roots);
    static double multiFitting(int nV, int n, double *rawx, double *rawy, int dim, double *params);
};

#endif
