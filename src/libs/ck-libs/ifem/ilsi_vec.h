/**
Tiny, highly efficient vector operations.
All of these allow simple argument aliasing.

Orion Sky Lawlor, olawlor@acm.org, 1/16/2003
*/
#ifndef __UIUC_CHARM_ILSI_VEC_H
#define __UIUC_CHARM_ILSI_VEC_H

typedef double real;

//Vector operations:
/// c=a
inline void copy(int n,real *c,const real *a)
{ for (int i=0;i<n;i++) c[i]=a[i]; }

/// c=a-b
inline void sub(int n,real *c,const real *a,const real *b)
{ for (int i=0;i<n;i++) c[i]=a[i]-b[i]; }

/// c=a+k*b  (floating-point multiply-add)
inline void fma(int n,real *c,const real *a,double k,const real *b)
{ 
	for (int i=0;i<n;i++) 
		c[i]=k*b[i]+a[i]; 
}

/// Take the dot product of these two vectors:
inline real dot(int n,const real *a,const real *b)
{ 
	real sum=(real)0;
	for (int i=0;i<n;i++) 
		sum+=a[i]*b[i]; 
	return sum;
}


/// Tiny, silly utility class for dynamically allocating vectors
class allocVector {
	real *sto;
public:
	allocVector(int n) {sto=new real[n];}
	~allocVector() {delete[] sto;}
	
	operator real *() {return sto;}
	operator const real *() const {return sto;}
};


#endif


