/**
Iterative Linear Solvers: C-callable interface.

Orion Sky Lawlor, olawlor@acm.org, 1/16/2003
*/
#ifndef __UIUC_CHARM_ILSI_C_H
#define __UIUC_CHARM_ILSI_C_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This struct gives the input and output parameters
 * for an IDXL_Solver. Because of fortran, it must be 
 * layout-compatible with an array of 20 doubles,   
 */
typedef struct {
/* Input parameters: */
  double maxResidual; /* (1) If nonzero, upper bound on total residual error. */
  double maxIterations; /* (2) If nonzero, upper bound on number of iterations to take. */
  double solverIn[8]; /* Solver-specific input parameters (normally 0) */
/* Output parameters: */
  double residual; /* (11) Residual error of final solution (estimate) */
  double iterations; /* (12) Number of iterations actually taken. */
  double solverOut[8]; /* Solver-specific output parameters (normally 0) */
} ILSI_Param;

/** Set default values for solver parameters. */
void ILSI_Param_new(ILSI_Param *param);


/** An ILSI_Comm is actually a C++ class--see ilsi.h */
typedef struct ILSI_Comm ILSI_Comm;


/**
 * An ILSI_Solver is a routine that computes, in parallel,
 * the solution x to the linear equation A x = b.
 * ILSI_Solvers must be written in C++, since ILSI_Comm is C++.
 *
 *   @param param Assorted input and output parameters for solver.
 *   @param comm The partitioned parallel matrix A and 
 *                communication system for the solver.
 *   @param n The length of the local part of the solution vectors.
 *   @param b The local part of the known vector.  Never modified.
 *   @param x On input, the initial guess for the solution.
 *            During execution, the intermediate solution values.
 *            On output, the final solution.
 */
typedef void (*ILSI_Solver)(ILSI_Param *param, ILSI_Comm *comm,
	int n, const double *b, double *x);

/** Conjugate-gradient solver: requires symmetric positive definite matrix */
void ILSI_CG_Solver(ILSI_Param *param, ILSI_Comm *comm,
	int n, const double *b, double *x);


#ifdef __cplusplus
}
#endif

#endif

