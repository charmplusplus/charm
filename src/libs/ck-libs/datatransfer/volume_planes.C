/*********************************** From VINCI lass ***************/
/*  http://www.lix.polytechnique.fr/Labo/Andreas.Enge/Vinci.html   */
/* Authors: Benno Bueeler (bueeler@ifor.math.ethz.ch)              */
/*          and                                                    */
/*          Andreas Enge (enge@ifor.math.ethz.ch)                  */
/*          Institute for Operations Research                      */
/*	    Swiss Federal Institute of Technology Zurich           */
/*	    Switzerland                                            */
/*
  (GPL CODE) 
  Lasserre's volume computation method 
  
     Modified by Orion Sky Lawlor, olawlor@acm.org, 2004/7/23
        - Wrapped implementation routines into a class
        - Remove global variables
	- Remove dynamic allocation by fixing dimension G_d to 3.
	- Dramatic simplification for 3D by removing caching.
     WARNING: if you need fast higher dimensions (>3D), don't
             use this routine, use the original VINCI code with caching!
	- Templated on dimension 'd' (save 20% of runtime)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   /* for fabs */
#include <string.h> /* for memcpy */
#include "volume_planes.h"

class VINCI_Lass
{
public:

/// Number of spatial dimensions.  Compile-time constant.
enum {G_d=3};

/// Maximum number of hyperplanes
enum {MAX_G_m=20};

/// Real number type used throughout
typedef double rational;

private:

#define MIN_PIVOT_LASS 0.1
#define EPSILON 1e-10

#define MAXIMUM 1.0e150   /* define the maximum used for testing infinity */
#define	EPSILON_LASS EPSILON   /* Numbers smaller than this are treated as zero in rhs*/
#define	EPS1    EPSILON    /* Numbers smaller than this are treated as zero in coefficient */
#define	EPS_NORM EPSILON   /* EPSILON used for constraint normalization*/
#define LaShiftLevel 0    /* Shifting is possible if d>=LaShiftLevel */
#define LaShift 1         /* shift polytope to make at least d components of
                             the rhs zero if there are less than d-LaShift zeros */
/* #define ReverseLass */ /* perform recursion from last to first constraint;
                             if undefined the recursion starts with the first constraint */
/* #define verboseFirstLevel */ /* output the intermediate volume of the first level */


/******************/
/*global variables*/
/******************/

rational planescopy[MAX_G_m * (G_d+1)];  /* needed in shift_P in the lasserre-code */

/***************/
/*help routines*/
/***************/

template <int d>
void rm_constraint(rational* A, int *LastPlane_, int rm_index)
/* removes the constraints given in rm_index and adjusts *LastPlane */

{   register rational *p1, *p2; 
    register int i;

    p1=A+rm_index*(d+1);
    p2=A+(rm_index+1)*(d+1);
    for (i=0; i<(((*LastPlane_)-rm_index)*(d+1)); i++) {
	*p1=*p2;
	p1++;
	p2++;
    };
    (*LastPlane_)--;
}

/***************/
/*Core routines*/
/***************/


bool notInPivot(int * pivot, int col, int i)
{ register int h;
  for (h=0;h<col;h++)
   if (pivot[h]==i) return false;
  return true;
}

template <int d>
void shift_P(rational *A, int LastPlane_)
/*  shift one vertex of the polytope into the origin, that
    is, make at least d components of b equal zero */

{   register rational  *p1, *p2, *p3, d1, d2, d3;
    register int col, i, j;
    int pivot[d+1]; /* contains the pivot row of each column */

    #ifdef STATISTICS
	Stat_CountShifts ++;
    #endif
    
    p1=A;                         /* search pivot of first column */
    pivot[0]=0; 
    d3=fabs(d1=*p1);
    for (i=0; i<=LastPlane_; i++) {
        d2=fabs(*p1);
#if PIVOTING_LASS == 0
	if (d2>=MIN_PIVOT_LASS) {pivot[0]=i; d1=*p1; break;};
#endif
	if (d2>d3) { pivot[0]=i; d1=*p1; d3=d2; };
	p1+=(d+1);
    }
    /* copy pivot row into planescopy */
    p1=A+pivot[0]*(d+1)+1;   
    p2=planescopy+pivot[0]*(d+1)+1;
    for (i=1,d2=1.0/d1; i<=d; i++,p1++,p2++) *p2 = (*p1)*d2;
    /* complete first pivoting and copying */
    p1=A+1;                          
    p2=planescopy+1;
    for (i=0; i<=LastPlane_; i++, p1++, p2++) {
	if (i==pivot[0]) {
	    p1+=d;
	    p2+=d;
	    continue;   /* pivot row already done */
	}
	d1=*(p1-1); 
	p3=planescopy+pivot[0]*(d+1)+1;
	for (j=1; j<=d; j++, p1++, p2++, p3++) (*p2)=(*p1)-d1*(*p3);
    }
    
    /* subsequent elimination below */
  
    for (col=1;col<d;col++) {
	for (i=0;i<=LastPlane_;i++)       /* search first row not already used as pivot row*/
	    if (notInPivot(pivot,col,i)) {
		pivot[col]=i; 
		break; 
	    }
	p1=planescopy+i*(d+1)+col;               /* search subsequent pivot row */
	d3=fabs(d1=*p1);
	for (; i<=LastPlane_; i++, p1+=(d+1))  
	    if (notInPivot(pivot,col,i)) {
	        d2=fabs(*(p1));
#if PIVOTING_LASS == 0
		if (d2>=MIN_PIVOT_LASS) {
		    pivot[col]=i; 
		    d1=*p1;
		    break; 
		}
#endif
		if (d2>d3) {
		    pivot[col]=i;
		    d1=*p1;
		    d3=d2;
		}
	    };
	/* update pivot row */
	p1=planescopy+pivot[col]*(d+1)+col+1;
	d2=1.0/d1;
	for (j=col+1; j<=d; j++, p1++) (*p1) *= d2;
	if (col==(d-1)) break;   /* the rest is not needed in the last case */
        /* update rest of rows */
        p1=planescopy+col+1;
        p2=planescopy+pivot[col]*(d+1)+col+1;
	for (i=0; i<=LastPlane_; i++, p1+=(col+1)) {
	    if (!notInPivot(pivot,col+1,i)) {
	        p1+=d-col;
		continue;
	    }
	    d1=*(p1-1);
	    for (j=col+1; j<=d; j++, p1++, p2++) *p1=(*p1)-d1*(*p2);
	    p2-=d-col;
	}
    };

    /* compute x* by backward substitution; result goes into rhs of planescopy */

    for (i=d-2; 0<=i; i--){
        p1=planescopy+pivot[i]*(d+1)+d;
	p2=p1-d+i+1;
	for (j=i+1; j<d; j++, p2++)
	    *(p1)-= (*p2)*(*(planescopy+pivot[j]*(d+1)+d));
    }
 
    /* compute shifted b  */

    for (i=0; i<=LastPlane_; i++) {
        p1=A+i*(d+1);
        p2=p1+d;
	if (notInPivot(pivot,d,i)) 
	    for (j=0; j<d; j++,p1++) {
		*p2 -= (*p1)*(*(planescopy+pivot[j]*(d+1)+d));
	    }
	else *p2=0;
    }
}

template <int d>
rational dot(rational *A,rational *B) {
	rational sum=0.0;
	for (int i=0;i<d;i++) sum+=A[i]*B[i];
	return sum;
}

template <int d>
int norm_and_clean_constraints(rational* A, int *LastPlane_)
/* Other (simpler) implementation of version lasserre-v15.
   Finally (up to the sign) identical constraints in A are detected. If they are
   identical the back one is removed, otherwise the system is infeasible. LastPlane_
   is reduced accordingly to the elimination process as well as insertion of the
   corresponding original indices into Del_index if Index_needed is true. */

{   register int i, j, row = 0;
    register rational r0, *p1, *p2;

    /* find nonzero[][] and maximal elements and normalize */
    p1=A;                                  /* begin of first constraint */
    while (row<=(*LastPlane_)) {           /* remove zeros and normalize */
	r0=dot<d>(p1,p1);               /* compute euclidean norm */
        if (r0<EPS_NORM*EPS_NORM) { /* very short normal */
            if ((p1[d])<-100000*EPS1){      /* if negative rhs */
		return 1;                  /* infeasible constraint */
	    }
	    rm_constraint<d>(A, LastPlane_, row);
	}
	else { /* make normals have unit magnitude */
	    r0=1.0/sqrt(r0);
	    for (j=0; j<=d; j++,p1++) (*p1)*=r0;
	    row++; 
	}
    }

    /* detect identical or reverse constraints */
    for (row=0; row<(*LastPlane_); row++) {
	for (i=row+1;i<=*LastPlane_;i++) 
	{ /* test all subsequent rows i for equality to row */
            p1=A+row*(d+1);
	    p2=A+i*(d+1);
	    r0=dot<d>(p1,p2); /* cosine of angle between vectors */
	    if (r0>=1.0-EPS_NORM) 
	    { /* nearly parallel constraint normals-- remove duplicate */
	    	if (p1[d]>p2[d]){
	    	    rm_constraint<d>(A, LastPlane_,row);
	    	    i=row;
            	}
	    	else {
	    	    if (i<(*LastPlane_)) 
	    		rm_constraint<d>(A, LastPlane_,i);
	    	    else (*LastPlane_)--;
	    	    i--;
            	}
	    }
	    else if (r0<=-1.0+EPS_NORM)
	    { /* nearly opposite constraints-- check for infeasible */
	    	if (p1[d]>0){
	    	    if (p2[d]<EPS1-p1[d]) return 1; 
	    	 }
	    	 else {
	    	     if (p1[d]<EPS1-p2[d]) return 1; 
	    	 }
	    }
	}
    }
    return 0;  /* elimination succesful */
}

/** 
  Recursive function to compute volume.
    lastPlane+1 is the number of halfplanes; the number of rows of A.
    d is the dimension of the polytope (always >1 here; 1 case is below);
    A has d+1 columns.
*/
template <int d>
rational lass(rational *A, int LastPlane_)
/* A has exact dimension (LastPlane_+1)*(d+1). The function returns
   the volume; an underscore is appended to LastPlane_ and d */

{
    int i, j;
    int baserow = 0, basecol = 0, col;
    int row; 
    bool i_balance = false;
    rational ma, mi,*realp1, *realp2;

    ma=0;                                         /* used to sum up the summands */
    if (norm_and_clean_constraints<d>(A, &LastPlane_)!=0)
        return 0.0;

    /* if appropriate shift polytope */

    if (d>=LaShiftLevel) {
	realp1=A+d;
	realp2=realp1+LastPlane_*(d+1);
	j=0;
	while (realp1<=realp2) {
	    if (fabs(*realp1)<EPSILON_LASS) j++;
	    realp1+=d+1;
	}
	if (d-j>=LaShift) shift_P<d>(A, LastPlane_);
    }

    /* redA = A reduced by one dimension and constraint */
    rational redA[(MAX_G_m-1)*G_d]; // should be size [LastPlane_* d]
    
#ifdef ReverseLass
    for (row=LastPlane_; row>=0; row--) {
#else
    for (row=0; row<=LastPlane_; row++) {
#endif
	if (fabs(A[row*(d+1)+d])<EPSILON_LASS) 
            continue;                        /* skip this constraint if b_row == 0 */
	rational pivotrow[G_d+1];  /* copy of pivot row */
	memcpy(&pivotrow[0], A+row*(d+1), sizeof(rational)*(d+1));
	col=0;                               /* search for pivot column */
	for (i=0; i<d; i++) {
#if PIVOTING_LASS == 0
	    if (fabs(pivotrow[i])>=MIN_PIVOT_LASS) {col=i; break;};
#endif
	    if (fabs(pivotrow[i])>fabs(pivotrow[col])) col=i;
	};
        
        /* copy A onto redA and at the same time perform pivoting */
	 
	mi=1.0/pivotrow[col];
	for (i=0; i<=d; i++) pivotrow[i]*=mi;
	realp1=A;
	realp2=redA;
	for (i=0; i<=LastPlane_; i++) {
	    if (i==row) {
		realp1+=d+1;
		continue;
	    };
	    mi=A[(i*(d+1))+col];
	    for (j=0; j<=d; j++) {
		if (j==col) {
		    realp1++;
		    continue;
		};
		*realp2=(*realp1)-pivotrow[j]*mi;
		realp1++;
		realp2++;
	    };
	};
	ma+= A[row*(d+1)+d]/(d*fabs(A[row*(d+1)+col]))
	     *lass<d-1>(redA, LastPlane_-1);
         #ifdef verboseFirstLevel
            if (d==G_d) 
	        printf("\nVolume accumulated to iteration %i is %20.12f",row,ma );
        #endif
    };
    
    return ma;
}

/**
  Base case for lass recursion.
    if d==1 compute the volume and give it back 
*/
template <>
rational lass<1>(rational *A, int LastPlane_)
{
    int i;
    rational ma, mi;

    ma=-MAXIMUM;
    mi= MAXIMUM;
    for (i=0; i<=LastPlane_; i++,A+=2) { 
    	if (*A>EPSILON_LASS) { if ((*(A+1)/ *A)<mi) mi=(*(A+1)/ *A); }
    	else if (*A<-EPSILON_LASS) { if ((*(A+1)/ *A)>ma) ma=*(A+1)/ *A; } 
    	else if ((*(A+1))<-(100000*EPSILON_LASS)) return 0; 
    }
    if ((ma<-.5*MAXIMUM)||(mi>.5*MAXIMUM)) {
    	printf("\nVolume is unbounded!\n");
    	exit(0);
    }
    if ((mi-ma)>EPSILON_LASS) {
    	return mi-ma;
    }
    return 0;
}

/*************************** External Interface ************************************/
public:

rational computeVolume(rational *planes,int nPlanes)
{
   memcpy(planescopy,planes,sizeof(rational)*nPlanes*(G_d+1));
   return lass<G_d>(planes, nPlanes-1);
}

};

/**
 Compute the volume of this set of halfspaces (hyperplanes).
  \param planes Values of halfplanes: array of (nPlanes) (G_d+1) rationals.
       We're inside the volume if, for each row A=&planes[i*4],
          A[0] * x + A[1] * y + A[2] * z <= A[3]
  \param nPlanes Number of halfspaces (hyperplanes).
*/
double computeVolumePlanes(const double *planes,int nPlanes)
{
   VINCI_Lass c;
   return c.computeVolume((double *)planes,nPlanes);
}

