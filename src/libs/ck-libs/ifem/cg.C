/**
Iterative Linear Solver: Conjugate Gradient method

As an algorithm reference, I recommend J. Shewchuk's excellent
article
	"An Introduction to the Conjugate Gradient Method
	   Without the Agonizing Pain"

Orion Sky Lawlor, olawlor@acm.org, 1/16/2003
*/
#include <stdio.h>
#include <math.h>
#include "ilsi.h"
#include "ilsi_vec.h"

class cgSolver {
	int n; //Length of vectors
	const real *b; //Target vector
	real *x_k; //Current solution estimate (user-allocated memory)
	allocVector r_k; //Residual-- should equal b-A x_k
	allocVector s_k; //Search direction
	allocVector tmp; //Temporary vector (avoids dynamic allocation)
	ILSI_Comm &comm;
	double residualMagSq; //Total magnitude of residual vector, squared
	
public:
	//Set up solver to handle vectors of length n (collective)
	cgSolver(int n,real *x,const real *b,ILSI_Comm &comm_);
	
	//Return the current solution vector
	real *getSolution(void) {
		return x_k;
	}
	
	//Return the magnitude of the residual vector
	double getResidual(void) const { return sqrt(residualMagSq); }
	
	//Advance the solution one step (collective).
	void iterate(void);
};

cgSolver::cgSolver(int n_,real *x_,const real *b_,ILSI_Comm &comm_)
	:n(n_), b(b_), x_k(x_), r_k(n), s_k(n), tmp(n), comm(comm_)
{
	residualMagSq=1.0e20;
	comm.matrixVectorProduct(x_k, tmp);
	sub(n,r_k, b,tmp); //Compute initial residual vector
	residualMagSq=comm.dotProduct(r_k,r_k);
	copy(n,s_k, r_k); //Initial search direction is residual
}

//Advance the solution one step.
// Stops and returns true if the error is less than the given value
void cgSolver::iterate(void)
{
	//Decide how far to advance
	comm.matrixVectorProduct(s_k,tmp);
	double s_kDot;
	s_kDot=comm.dotProduct(s_k,tmp);
	double alpha=residualMagSq/s_kDot;
	
	//Advance solution along the search direction
	fma(n,x_k, x_k,alpha,s_k);
	
	//Update residual
	double oldMagSq=residualMagSq;
	fma(n,r_k, r_k,-alpha,tmp);
	residualMagSq=comm.dotProduct(r_k,r_k);
	
	//printf("residualMagSq=%g\n",residualMagSq);
	//if (sqrt(fabs(residualMagSq))<=maxErr) return true; //We're done
	
	//Update search direction
	double beta=residualMagSq/oldMagSq;
	fma(n,s_k, r_k,beta,s_k);
}

CDECL void ILSI_CG_Solver(ILSI_Param *param, ILSI_Comm *comm,
	int n, const real *b, real *x)
{
	int maxIterations=1000000000;
	if (param->maxIterations>=1) maxIterations=(int)param->maxIterations;
	int nIterations=0;
	cgSolver cg(n,x,b,*comm);
	while (cg.getResidual()>param->maxResidual) {
		cg.iterate();
		nIterations++;
		if (nIterations>=maxIterations) break;
	}
	param->residual=cg.getResidual();
	param->iterations=nIterations;
}

// This nastiness lets f90 people pass ILSI_CG_Solver as a parameter:
FORTRAN_NAME_SOLVER(ILSI_CG_SOLVER,ILSI_CG_Solver,ilsi_cg_solver)

