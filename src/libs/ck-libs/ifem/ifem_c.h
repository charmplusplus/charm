/**
Implicit FEM: C-callable interface.

Orion Sky Lawlor, olawlor@acm.org, 1/17/2003
*/
#ifndef __UIUC_CHARM_IFEM_C_H
#define __UIUC_CHARM_IFEM_C_H

#include "ilsi_c.h" /* For ILSI_Param, ILSI_Solver */

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Performs a parallel matrix-vector multiply.
 * This type of routine is typically written by the user,
 * and directly applies element stiffness functions to the
 * source data.
 *
 *  @param ptr User-defined pointer value.
 *  @param length Number of entries in dest and src vectors.
 *  @param width Number of double per entries in dest and src vectors.
 *  @param src Source vector--multiply this with the matrix.
 *  @param dest Destination vector--initially zero; fill this 
 *              with the product of the local elements and src.
 */
typedef void (*IFEM_Matrix_product_c)(void *ptr, 
	int length,int width,const double *src, double *dest);
typedef void (*IFEM_Matrix_product_f)(void *ptr, 
	const int *length,const int *width,const double *src, double *dest);

/**
 * Solve the matrix equation A x = b for the unknown node value x.
 * The matrix is assumed to be partitioned by element, and
 * the unknown and known vectors are listed by node.
 *
 * This version uses the shared-node solution formulation; so
 * the matrix-product function takes a vector of local node values
 * and returns a vector of local node values including contributions
 * from all local elements.  IFEM_Solve_shared sums this output vector 
 * with the contributions from remote elements to arrive at a complete 
 * matrix-vector product.
 * 
 *   @param s The solver to use (e.g., ILSI_CG_Solver)
 *   @param param Assorted input and output parameters for the solver.
 *   @param fem_mesh Readable FEM mesh object to solve over.
 *   @param fem_entity FEM mesh entity to solve over (typically FEM_NODE).
 *   @param length The number of shared entities.
 *   @param width The number of unknowns per shared entity.  b and x
 *                must have length*width entries.
 *   @param A The user function that applies the matrix.
 *   @param ptr User-defined pointer value passed to A.
 *   @param b The local part of the known vector.  Never modified.
 *   @param x On input, the initial guess for the solution.
 *            During execution, the intermediate solution values.
 *            On output, the final solution.
 */
void IFEM_Solve_shared(ILSI_Solver s,ILSI_Param *p,
	int fem_mesh, int fem_entity,int length,int width,
	IFEM_Matrix_product_c A, void *ptr, 
	const double *b, double *x);


#ifdef __cplusplus
};
#endif

#endif
