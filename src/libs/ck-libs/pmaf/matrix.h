/// Matrix class with some fundmental operations
/** This is probably a bad thing to do, but given my luck with STL of late,
    I don't want to have to deal with anyone else's ill- or un-documented
    code.  Matrices are here to get Delaunay flippable tests done. */
class Matrix {
 protected:
  /// this mtx can be of any size
  double **mtx;
  /// dimension of matrix (when using square matrices)
  int order;
  
 public:
  /// Perform basic initialization
  /** Allocate arrays for a square matrix of size dim X dim, and initialize
      order to dim. */
  Matrix(int dim) {
    mtx = (double **)malloc(dim*sizeof(double *));
    for (int i=0; i<dim; i++)
      mtx[i] = (double *)malloc(dim*sizeof(double));
    order = dim;
  }
  
  /// Sets element i,j of matrix
  void setElement(int i, int j, double value) { 
    CmiAssert((i < order) && (j < order));
    mtx[i][j] = value; 
  }

  /// Accesses matrix and returns mtx[i][j] (value)
  const double &elem(int i, int j) const { 
    CmiAssert((i < order) && (j < order));
    return mtx[i][j]; 
  }
  
  /// Calculate and return the matrix' determinant
  double determinant(void) const {
    double d = 0.0;
    if (order <= 1) return elem(0, 0);
    for (int i=0; i<order; i++) {
      double sign = (i%2) ? -1 : 1;
      d += sign * elem(i, 0) * cofactor(i, 0).determinant();
    }
    return d;
  }

  /// Helper to determinant but may eventually be generally useful
  /** Returns a new matrix with dim one smaller than current, with 
      row aI and column aJ removed.  */
  Matrix cofactor(int aI, int aJ) const {
    CmiAssert((aI < order) && (aJ < order));
    Matrix a(order-1);
    for (int i=0, k=0; i<order; i++)
      if (i != aI) {
        for (int j=0, l=0; j<order; j++) {
          if (j != aJ) {
            a.setElement(k, l, elem(i, j));
            l++;
          }
        }
        k++;
      }
    return a;
  }
};
