/**
Iterative Linear Solver Interface, which is used to 
iteratively solve parallel linear systems of equations
of the form
	A x = b
where 
A is a matrix, typically represented implicitly 
   by a partitioned parallel set of finite elements; 
x is the unknown vector, and 
b is the known solution vector.

This is the C++ interface used by actual Solvers.
A Solver is a single subroutine, with the interface
specified by ILSI_Solver in ilsi_c.h, that uses an ILSI_Comm
to (presumably iteratively) solve a linear system.

Oddly enough, because the parallelism is encapsulated in 
ILSI_Comm, solvers do *not* need to do any parallel work;
they are presented with an apparently serial interface.

Orion Sky Lawlor, olawlor@acm.org, 1/16/2003
*/
#ifndef __UIUC_CHARM_ILSI_H
#define __UIUC_CHARM_ILSI_H

#include "charm-api.h"
#include "ilsic.h"

/**
 * An ILSI_Comm is an ILSI_Solver's interface to the 
 * (parallel) problem matrix and communication system.
 */
class ILSI_Comm {
public:
	virtual ~ILSI_Comm();
	
	/// Compute dest = A src, where A is the square matrix.
	///  This product must take into account values from other
	///  chunks, as well as this one.  This is a collective call.
	///  It is not valid for src to equal dest.
	virtual void matrixVectorProduct(const double *src,double *dest) =0;
	
	/// Do a global dot product of these two vectors.  
	/// This dot product must take into account values from other
	/// processors.  It is valid for a to equal b.
	/// All chunks are guaranteed to get the same return value.
	virtual double dotProduct(const double *a,const double *b) =0;
};

/** 
 * This macro lets f90 callers reference C solvers. If link
 *  aliases were more widely supported, this would make the f90
 *  symbol name a link alias of the C version.
 */
#define FORTRAN_NAME_SOLVER(CAPITALNAME,Cname,lowercasename) \
CDECL void FTN_NAME(CAPITALNAME,lowercasename) \
	(ILSI_Param *param, ILSI_Comm *comm,int n, const double *b, double *x) \
{ Cname(param,comm,n,b,x); }


#endif

