/**
Tiny serial test for linear solvers.
For a real parallel example program, see
	charm/pgms/charm++/fem/matrix/

Orion Sky Lawlor, olawlor@acm.org, 1/16/2003
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ilsi.h"
#include "ilsi_vec.h"

double nextDouble(void) {
	return (rand()%8192)*(1.0/8192.0);
}

/// An ILSI_Comm that holds a single, serial, dense matrix.
class serial_comm : public ILSI_Comm {
	int n;
	const double *matrix; //n x n doubles of matrix, row-major order
	double at(int r,int c) const {
		return matrix[r*n+c];
	}
public:
	serial_comm(int n_,const double *matrix_) 
		:n(n_), matrix(matrix_) {}
	
	virtual void matrixVectorProduct(const double *src,double *dest) {
		for (int r=0;r<n;r++) {
			double sum=0;
			for (int c=0;c<n;c++)
				sum+=at(r,c)*src[c];
			dest[r]=sum;
		}
	}

	virtual double dotProduct(const double *a,const double *b) {
		double sum=0;
		for (int r=0;r<n;r++) sum+=a[r]*b[r];
		return sum;
	}
};

int main(int argc,char *argv[]) {
	int n=4; //Solve an n x n system
	if (argc>1) n=atoi(argv[1]);
	int r,c;
	//Generate a random matrix and "unknown" x
	double *A=new double[n*n], *true_x=new double[n];
	srand(2);
	for (r=0;r<n;r++) {
		for (c=0;c<n;c++) {
			if (c<r)
				A[r*n+c]=A[c*n+r]; //Symmetric copy
			else
				A[r*n+c]=nextDouble();
		}
		true_x[r]=nextDouble();
	}
	serial_comm comm(n,A);
	
	//Compute the target vector as b = A x
	double *b=new double[n], *x=new double[n];
	comm.matrixVectorProduct(true_x,b);
	
	//Use solver to try to recover true_x
	ILSI_Param param;
	ILSI_Param_new(&param);
	const double tolerance=1.0e-5;
	param.maxResidual=tolerance;
	
	printf("Solving...\n");
	ILSI_CG_Solver(&param,&comm, n,b,x);
	
	//Compare the true and estimated solutions
	double totalErr=0;
	for (r=0;r<n;r++) {
		double err=fabs(x[r]-true_x[r]);
		if (err>tolerance) {
			printf("Big difference %g found at position %d!\n",err,r);
			exit(1);
		}
		totalErr+=err;
	}
	printf("Solved: est. res=%g, err=%g, iter=%d\n",
		param.residual, totalErr, (int)param.iterations);
	
	exit(0);
}

