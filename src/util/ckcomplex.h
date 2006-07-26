#ifndef __CKCOMPLEX_H__
#define __CKCOMPLEX_H__

/*
  #if defined(__INTEL_COMPILER)
  #include <emmintrin.h>
  #define PAIR_USE_SSE 0
  #endif
*/

#include "conv-mach.h"

#ifndef CMK_VERSION_BLUEGENE
#if CMK_MALLOC_USE_GNU_MALLOC 
extern "C" void *memalign(size_t boundary, size_t size);
#endif
#endif

struct ckcomplex {
    double re;
    double im;
    
    inline ckcomplex(double _re=0., double _im=0.): re(_re), im(_im){}
  //    inline ckcomplex(double r) {re=r; im=0;}
  //    inline ckcomplex(double r,double i) {re=r; im=i;}
    
    inline ~ckcomplex() {}

    inline double getMagSqr(void) const { 
#if PAIR_USE_SSE      
        double ret;
        __m128d dreg1 = _mm_loadu_pd((double *) this);
        __m128d dreg2 = _mm_loadu_pd((double *) this);

        dreg1 = _mm_mul_pd(dreg1, dreg2);
        dreg2 = _mm_unpackhi_pd(dreg1, dreg1);

        dreg2 = _mm_add_sd(dreg1, dreg2);        
        _mm_storel_pd(&ret, dreg2);
        return ret;                      
#else
        return re*re+im*im; 
#endif
    }

    inline ckcomplex operator+(ckcomplex a) { 
#if PAIR_USE_SSE    
        __m128d dreg1 = _mm_loadu_pd((double *)this);
        __m128d dreg2 = _mm_set_pd(a.im, a.re);
  
        dreg2 = _mm_add_pd(dreg1, dreg2);
        ckcomplex ret;
        _mm_storeu_pd((double *)&ret, dreg2);
        return ret;
#else         
        return ckcomplex(re+a.re,im+a.im); 
#endif
    }

    inline ckcomplex conj(void) { 
        return ckcomplex(re, -im); 
    }

    inline void operator+=(ckcomplex a) { 
        
#if PAIR_USE_SSE
        __m128d dreg1 = _mm_loadu_pd((double *)this);
        __m128d dreg2 = _mm_loadu_pd((double *)&a);
        
        _mm_storeu_pd((double *)this, _mm_add_pd(dreg1, dreg2));
#else
        re+=a.re; im+=a.im; 
#endif

    }
    
    inline ckcomplex operator*(double a) { 
#if PAIR_USE_SSE
        __m128d dreg1 = _mm_loadu_pd((double *)this);
        __m128d dreg2 = _mm_set_pd(a, a);
        
        dreg2 = _mm_mul_pd(dreg1, dreg2);
        
        ckcomplex ret;
        _mm_storeu_pd((double *)&ret, dreg2);
        return ret;
        
#else
        return ckcomplex(re*a, im*a); 
#endif
    } 

       inline bool notzero() const { return( (0.0 != re) ? true : (0.0 != im)); }

    inline void operator*=(ckcomplex a) {        
#if PAIR_USE_SSE
        __m128d dreg2 = _mm_set_pd(re, re); //re, re        
        __m128d dreg3 = _mm_set_pd(a.im, a.re);   //a.im, a.re
        __m128d dreg4 = _mm_mul_pd(dreg2, dreg3); //a.im *re, a.re * re
                
        dreg2 = _mm_set_pd(im, im); //im, im        
        dreg3 = _mm_set_pd(a.re, a.im); 
        dreg2 = _mm_mul_pd(dreg2,dreg3); //a.im *im, a.re * im
        dreg2 = _mm_mul_pd(dreg2, _mm_set_pd(1.0, -1.0));

        dreg4 =  _mm_add_pd(dreg4, dreg2);
        //a.im * re + a.re * im, a.re * re - a.im * im


        _mm_storeu_pd((double *)this, dreg4);         
#else
        double treal, tim;
        treal = re * a.re - im * a.im;
        tim = re * a.im + im * a.re;
        re = treal;
        im = tim;
#endif
    }

    inline ckcomplex operator*(ckcomplex a) {
#if PAIR_USE_SSE  
        __m128d dreg2 = _mm_set_pd(re, re); //re, re
        
        __m128d dreg3 = _mm_set_pd(a.im, a.re);   //a.im, a.re
        __m128d dreg4 = _mm_mul_pd(dreg2, dreg3); //a.im *re, a.re * re
                
        dreg2 = _mm_set_pd(im, im); //im, im        
        dreg3 = _mm_set_pd(a.re, a.im); 
        dreg2 = _mm_mul_pd(dreg2,dreg3); //a.im *im, a.re * im
        dreg2 = _mm_mul_pd(dreg2, _mm_set_pd(1.0, -1.0));

        dreg4 =  _mm_add_pd(dreg4, dreg2);
        //a.im * re + a.re * im, a.re * re - a.im * im

        ckcomplex ret;
        _mm_storeu_pd((double *)&ret, dreg4);
        return ret;
#else
        return ckcomplex( re * a.re - im * a.im, re * a.im + im * a.re); 
#endif
    }


    inline void operator -= (ckcomplex a) {
#if PAIR_USE_SSE 
        __m128d dreg1 = _mm_loadu_pd((double *)this);
        __m128d dreg2 = _mm_loadu_pd((double *)&a);
        
        _mm_storeu_pd((double *)this, _mm_sub_pd(dreg1, dreg2));
#else
        re -= a.re; im -= a.im;
#endif
    }


    inline ckcomplex multiplyByi () {
        return ckcomplex(-im, re);
    }
        
    inline void * operator new[] (size_t size){
#if CMK_MALLOC_USE_GNU_MALLOC 
        void *buf = memalign(16, size);
#else 
        void *buf = malloc(size);
#endif
        return buf;
    }
    
    inline void operator delete[] (void *buf){
        free(buf);
    }
};

typedef ckcomplex complex;

PUPbytes(ckcomplex);

// Backward compatability:
// Assume that you only have ckcomplex's definition of complex 
// Unless WRAP_COMPLEX is defined, in which case you have a 
// complex from elsewhere and want a distinct ckcomplex.

// This allows old codes which used ckcomplex to just work.


#ifndef CKCOMPLEX_ISNOT_COMPLEX
typedef ckcomplex complex;
#endif

#endif
