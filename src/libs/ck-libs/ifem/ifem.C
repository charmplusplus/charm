/**
Implicit FEM: C- and fortran-callable interface.

Orion Sky Lawlor, olawlor@acm.org, 1/17/2003
*/
#include "charm++.h"
#include "charm-api.h"
#include "ifemc.h"
#include "ilsi.h"
#include "fem.h"
#include "idxlc.h"


/// Sum the dot product of the fields of a and b marked with 1's in goodRecords
double localDotProduct(int nRecords,int nFields,const unsigned char *goodRecord,
		const double *a,const double *b)
{
	double sum=0.0;
	for (int r=0;r<nRecords;r++) 
		if (goodRecord[r]) //We are responsible for this record:
			for (int f=0;f<nFields;f++) {
				int i=r*nFields+f;
				sum+=a[i]*b[i];
			}
	return sum;
}

/**
 * Interface between the basic "matrix-style" interface expected by
 * the ILSI solvers, the FEM framework, and the local matrix-multiply 
 * function passed in by the user.
 */
class IFEM_Solve_shared_comm : public ILSI_Comm {
	int mesh,entity; //FEM identifier for our shared entity
	const int length,width; //Size of array we're solving for
	unsigned char *primary; //Marker indicating that we are responsible for dot product summation
	IDXL_Layout_t shared_fid, reduce_fid;
	IDXL_t shared_idxl;
	
	//User-written matrix-vector product routine
	IFEM_Matrix_product_c A_c;
	IFEM_Matrix_product_f A_f;
	void *ptr;
public:
	IFEM_Solve_shared_comm(int mesh_,int entity_, int length_,int width_)
		:mesh(mesh_), entity(entity_), length(length_), width(width_),
		 A_c(0), A_f(0), ptr(0)
	{
		//sanity checks on inputs:
		if (length!=FEM_Mesh_get_length(mesh,entity))
			CkAbort("IFEM_Solve_shared: vector length must equal number of nodes!");
		if (width<1) CkAbort("IFEM_Solve_shared: number of unknowns per node < 1!");
		if (width>100) CkAbort("IFEM_Solve_shared: do you really want that many unknowns per node?");
		
		// Prepare the fields we'll need during the run:
		shared_fid=IDXL_Layout_create(IDXL_DOUBLE,width);
		reduce_fid=IDXL_Layout_create(IDXL_DOUBLE,1);
		primary=new unsigned char[length];
		FEM_Mesh_get_data(mesh,entity,FEM_NODE_PRIMARY, primary,
			0,length, FEM_BYTE,1);
		shared_idxl=FEM_Comm_shared(mesh,entity);
	}
	
	// You have to register one of these
	void set_c(IFEM_Matrix_product_c A,void *ptr_) {
		A_c=A; ptr=ptr_;
	}
	void set_f(IFEM_Matrix_product_f A,void *ptr_) {
		A_f=A; ptr=ptr_;
	}
	
	/// Compute dest = A src, where A is the square stiffness matrix.
	virtual void matrixVectorProduct(const double *src,double *dest) {
		//Call user routine to do product
		if (A_c) (A_c)(ptr,length,width,src,dest);
		else /*A_f*/ (A_f)(ptr,&length,&width,src,dest);
	}
	
	/// Do a parallel dot product of these two vectors
	virtual double dotProduct(const double *a,const double *b) {
		// First do sum over local, primary nodes:
		double sum=localDotProduct(length,width,primary,a,b);
		// Now call FEM to sum over parallel values:
		double gsum;
		FEM_Reduce(reduce_fid,&sum,&gsum,FEM_SUM);
		return gsum; //Return global sum of values
	}
	
	~IFEM_Solve_shared_comm() {
		delete[] primary;
		IDXL_Layout_destroy(reduce_fid);
		IDXL_Layout_destroy(shared_fid);
	}
};

CDECL void 
IFEM_Solve_shared(ILSI_Solver s,ILSI_Param *p,
	int fem_mesh, int fem_entity,int length,int width,
	IFEM_Matrix_product_c A, void *ptr, 
	const double *b, double *x)
{
	IFEM_Solve_shared_comm comm(fem_mesh,fem_entity,length,width);
	comm.set_c(A,ptr);
	
	int n=length*width;
	(s)(p,&comm,n,b,x);
}

FDECL void 
FTN_NAME(IFEM_SOLVE_SHARED,ifem_solve_shared)
	(ILSI_Solver s,ILSI_Param *p,
	int *fem_mesh, int *fem_entity,int *length,int *width,
	IFEM_Matrix_product_f A, void *ptr, 
	const double *b, double *x)
{
	IFEM_Solve_shared_comm comm(*fem_mesh,*fem_entity,*length,*width);
	comm.set_f(A,ptr); //<- this is the only difference between C and Fortran versions.
	
	int n=*length* *width;
	(s)(p,&comm,n,b,x);
}

/************* Boundary Condition wrapper **************/

/**
 * Wrapper for user matrix-vector multiply that applies
 * "essential" boundary conditions.
 *
 * The basic idea is that given a problem
 *    A x = b
 * where x is only partially unknowns--that is, if x=u+c (Unknowns and boundary Conditions)
 *    A u + A c = b
 * so there's an equivalent all-unknown system
 *    A u = b - A c = b'
 * Now for A u = b', we have to just zero out all the boundary conditions.
 */
class BCapplier {
	//User-written matrix-vector product routine to call
	IFEM_Matrix_product_c A_c;
	IFEM_Matrix_product_f A_f;
	void *ptr;
	
	inline void userMultiply(int length,int width,const double *src,double *dest) {
		//Call user routine to do product
		if (A_c) (A_c)(ptr,length,width,src,dest);
		else /*A_f*/ (A_f)(ptr,&length,&width,src,dest);
	}
	
	/**
	 * Definition of boundary conditions:
	 *  for (i=0;i<bcCount;i++)
	 *     assert(x[at(i)]==bcValue[i])
	 */
	int bcCount; //Length of below arrays
	int idxBase;
	const int *bcDOF; //Degree-of-freedom (vector index) to apply BC to
	inline int at(int bcIdx) { return bcDOF[bcIdx]-idxBase; }
	const double *bcValue; //Imposed value of BC
	
public:
	BCapplier(int cnt_,int base_,const int *dof_,const double *value_) 
		:A_c(0), A_f(0), ptr(0), bcCount(cnt_), idxBase(base_),
		 bcDOF(dof_), bcValue(value_)
	{
	}
	
	// You have to register one of these
	void set_c(IFEM_Matrix_product_c A,void *ptr_) {
		A_c=A; ptr=ptr_;
	}
	void set_f(IFEM_Matrix_product_f A,void *ptr_) {
		A_f=A; ptr=ptr_;
	}
	
	// Do the calculation to compute x:
	void solve(ILSI_Solver s,ILSI_Param *p,
		int fem_mesh, int fem_entity,int length,int width,
		const double *b,double *x);
	
	// Do an A u matrix-vector multiply
	void multiply(int length,int width,const double *u,double *Au) {
		int i;
		// Make sure u[bc's] is zero--like zeroing out column of A
		//  for (i=0;i<bcCount;i++) assert(u[at(i)]==0.0);
		
		//Call user routine to do ordinary multiply
		userMultiply(length,width,u,Au);
		
		// In-place zero out boundary parts of Au--
		//  This is like zeroing out the BC rows of A
		for (i=0;i<bcCount;i++) Au[at(i)]=0.0;
	}
};

extern "C" void
BCapplier_multiply(void *ptr,int length,int width,const double *src,double *dest)
{
	BCapplier *bc=(BCapplier *)ptr;
	bc->multiply(length,width,src,dest);
}

void BCapplier::solve(ILSI_Solver s,ILSI_Param *p,
		int fem_mesh, int fem_entity,int length,int width,
		const double *b,double *x)
{
// Subtract off boundary conditions (converts b into b'==b-A c)
	int i,DOFs=length*width;
	// Compute A c (response of system to fixed boundary conditions)
	double *c=new double[DOFs];
	for (i=0;i<DOFs;i++) c[i]=0;
	for (i=0;i<bcCount;i++) c[at(i)]=bcValue[i];
	double *Ac=new double[DOFs];
	userMultiply(length,width,c,Ac);
	
	// Compute new boundary conditions b'
	double *bPrime=c; //Re-use storage
	for (i=0;i<DOFs;i++) bPrime[i]=b[i]-Ac[i];
	delete[] Ac;
	
	// Zero out known parts of bPrime and x
	for (i=0;i<bcCount;i++) {
		x[at(i)]=0.0;
		bPrime[at(i)]=0.0;
	}
	
// Solve A u = b'
	IFEM_Solve_shared(s,p, fem_mesh,fem_entity,length,width,
		BCapplier_multiply,this,bPrime,x);
	
	delete[] c;
	
// Insert known boundary values back into x
	for (i=0;i<bcCount;i++) x[at(i)]=bcValue[i];
}

CDECL void 
IFEM_Solve_shared_bc(ILSI_Solver s,ILSI_Param *p,
	int fem_mesh, int fem_entity,int length,int width,
	int bcCount, const int *bcDOF, const double *bcValue,
	IFEM_Matrix_product_c A, void *ptr, 
	const double *b, double *x)
{
	BCapplier bc(bcCount,0,bcDOF,bcValue);
	bc.set_c(A,ptr);
	bc.solve(s,p,fem_mesh,fem_entity,length,width,b,x);
}

FDECL void 
FTN_NAME(IFEM_SOLVE_SHARED_BC,ifem_solve_shared_bc)
	(ILSI_Solver s,ILSI_Param *p,
	int *fem_mesh, int *fem_entity,int *length,int *width,
	int *bcCount,const int *bcDOF,const double *bcValue,
	IFEM_Matrix_product_f A, void *ptr, 
	const double *b, double *x)
{
	BCapplier bc(*bcCount,1,bcDOF,bcValue);
	bc.set_f(A,ptr);
	bc.solve(s,p,*fem_mesh,*fem_entity,*length,*width,b,x);
}


