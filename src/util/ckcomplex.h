#ifndef __CKCOMPLEX_H__
#define __CKCOMPLEX_H__

#include "fftw.h"

#if defined(__INTEL_COMPILER)
#include <emmintrin.h>
#define PAIR_USE_SSE 0
#endif

struct complex {
    fftw_real re;
    fftw_real im;
    
    inline complex() {re=0; im=0;}
    inline complex(fftw_real r) {re=r; im=0;}
    inline complex(fftw_real r,fftw_real i) {re=r; im=i;}
    
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

    inline complex operator+(complex a) { 
#if PAIR_USE_SSE    
        __m128d dreg1 = _mm_loadu_pd((double *)this);
        __m128d dreg2 = _mm_set_pd(a.im, a.re);
  
        dreg2 = _mm_add_pd(dreg1, dreg2);
        complex ret;
        _mm_storeu_pd((double *)&ret, dreg2);
        return ret;
#else         
        return complex(re+a.re,im+a.im); 
#endif
    }

    inline complex conj(void) { 
        return complex(re, -im); 
    }

    inline void operator+=(complex a) { 
        
#if PAIR_USE_SSE
        __m128d dreg1 = _mm_loadu_pd((double *)this);
        __m128d dreg2 = _mm_loadu_pd((double *)&a);
        
        _mm_storeu_pd((double *)this, _mm_add_pd(dreg1, dreg2));
#else
        re+=a.re; im+=a.im; 
#endif

    }
    
    inline complex operator*(double a) { 
#if PAIR_USE_SSE
        __m128d dreg1 = _mm_loadu_pd((double *)this);
        __m128d dreg2 = _mm_set_pd(a, a);
        
        dreg2 = _mm_mul_pd(dreg1, dreg2);
        
        complex ret;
        _mm_storeu_pd((double *)&ret, dreg2);
        return ret;
        
#else
        return complex(re*a, im*a); 
#endif
    } 

       inline bool notzero() const { return( (0.0 != re) ? true : (0.0 != im)); }

    inline void operator*=(complex a) {        
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

    inline complex operator*(complex a) {
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

        complex ret;
        _mm_storeu_pd((double *)&ret, dreg4);
        return ret;
#else
        return complex( re * a.re - im * a.im, re * a.im + im * a.re); 
#endif
    }


    inline void operator -= (complex a) {
#if PAIR_USE_SSE 
        __m128d dreg1 = _mm_loadu_pd((double *)this);
        __m128d dreg2 = _mm_loadu_pd((double *)&a);
        
        _mm_storeu_pd((double *)this, _mm_sub_pd(dreg1, dreg2));
#else
        re -= a.re; im -= a.im;
#endif
    }


    inline complex multiplyByi () {
        return complex(-im, re);
    }
    
    
    void pup(PUP::er &p) {
        p|re;
        p|im;
    }
    
    void * operator new[] (size_t size){
        void *buf = malloc(size);
        return buf;
    }
    
    void operator delete[] (void *buf){
        free(buf);
    }
};



#endif
