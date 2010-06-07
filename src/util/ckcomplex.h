#ifndef __CKCOMPLEX_H__
#define __CKCOMPLEX_H__

#include "pup.h"

#if USE_FFTW_DECLS
#include "fftw.h"
typedef fftw_real  RealType;
#else
typedef double     RealType;
#endif

struct ckcomplex {
    RealType  re;
    RealType  im;   

 
    inline ckcomplex(RealType _re=0., RealType _im=0.): re(_re), im(_im){}
    //    inline ckcomplex(RealType r) {re=r; im=0;}
    //    inline ckcomplex(RealType r,RealType i) {re=r; im=i;}
    
    inline ~ckcomplex() {}

    inline RealType getMagSqr(void) const { 
      return re*re+im*im; 
    }

    inline ckcomplex operator+(ckcomplex a) { 
      return ckcomplex(re+a.re,im+a.im); 
    }

    // note: not a member
    inline friend ckcomplex operator-(ckcomplex lhs, ckcomplex rhs) {
      return ckcomplex(lhs.re - rhs.re, lhs.im - rhs.im);
    }

    inline ckcomplex conj(void) { 
        return ckcomplex(re, -im); 
    }

    inline void operator+=(ckcomplex a) { 
      re+=a.re; im+=a.im; 
    }
    
    // note: not a member
    inline friend ckcomplex operator*(RealType lhs, ckcomplex rhs) {
      return ckcomplex(rhs.re*lhs, rhs.im*lhs);
    }

    inline ckcomplex operator*(RealType a) { 
      return ckcomplex(re*a, im*a); 
    } 

    inline bool notzero() const { return( (0.0 != re) ? true : (0.0 != im)); }

    inline void operator*=(ckcomplex a) {        
        RealType treal, tim;
        treal = re * a.re - im * a.im;
        tim = re * a.im + im * a.re;
        re = treal;
        im = tim;
    }

    inline ckcomplex operator*(ckcomplex a) {
      return ckcomplex( re * a.re - im * a.im, re * a.im + im * a.re); 
    }


    inline void operator -= (ckcomplex a) {
      re -= a.re; im -= a.im;
    }


    inline ckcomplex multiplyByi () {
        return ckcomplex(-im, re);
    }
        
    inline void * operator new[] (size_t size){
        void *buf = malloc(size);
        return buf;
    }
    
    inline void operator delete[] (void *buf){
        free(buf);
    }
};

typedef ckcomplex complex;

PUPbytes(ckcomplex)

#endif
