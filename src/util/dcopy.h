
#ifndef __CMI_DMEMCPY_128_H__
#define __CMI_DMEMCPY_128_H__

/**************************************************************
*  Optimized version of memory copy designed for Blue Gene/L  *
*  It inlines short memory copy operations and optimizes      *
*  8 byte aligned copies larger than 128 bytes                *
*                  - Sameer (04/07)                           *
*  It has two functions :                                     *
*  inlined opt_memcopy (cmimemcpy.h) and __dcopy128           *
**************************************************************/

inline void *bg_dcopy128 ( void * dest, const void *src, size_t n )
{
  const double *f1 = ((const double *) src) - 1;
  double       *f2 = ((double *) dest) - 1;
  
  size_t size = n >> 7;
  size_t remainder = n & 0x7f;
  
  register double r0;
  register double r1;
  register double r2;
  register double r3;
  register double r4;
  register double r5;
  register double r6;
  register double r7;

  if (size > 0) {
    size --;   
    r0 = *(++f1);
    r1 = *(++f1);
    r2 = *(++f1);
    r3 = *(++f1);
    r4 = *(++f1);
    
    while (size -- ) {    
      *(++ f2) = r0;
      r5       = *(++f1);
      *(++ f2) = r1;
      r6       = *(++f1);
      *(++ f2) = r2;
      r7       = *(++f1);
      *(++ f2) = r3;    
      r0       = *(++f1);
      
      *(++ f2) = r4;
      r1       = *(++f1);
      *(++ f2) = r5;
      r2       = *(++f1);
      *(++ f2) = r6;
      r3       = *(++f1);
      *(++ f2) = r7;    
      r4       = *(++f1);
      
      *(++ f2) = r0;
      r5       = *(++f1);
      *(++ f2) = r1;
      r6       = *(++f1);
      *(++ f2) = r2;
      r7       = *(++f1);
      
      *(++ f2) = r3;    
      r0       = *(++f1);
      *(++ f2) = r4;
      r1       = *(++f1);
      *(++ f2) = r5;
      r2       = *(++f1);
      *(++ f2) = r6;
      r3       = *(++f1);
      *(++ f2) = r7;
      r4       = *(++f1);
    }
    
    *(++ f2) = r0;
    r5       = *(++f1);
    *(++ f2) = r1;
    r6       = *(++f1);
    *(++ f2) = r2;
    r7       = *(++f1);
    *(++ f2) = r3;    
    r0       = *(++f1);
    
    *(++ f2) = r4;
    r1       = *(++f1);
    *(++ f2) = r5;
    r2       = *(++f1);
    *(++ f2) = r6;
    r3       = *(++f1);
    *(++ f2) = r7;    
    r4       = *(++f1);
    
    *(++ f2) = r0;
    r5       = *(++f1);
    *(++ f2) = r1;
    r6       = *(++f1);
    *(++ f2) = r2;
    r7       = *(++f1);
    
    *(++ f2) = r3;    
    *(++ f2) = r4;
    *(++ f2) = r5;
    *(++ f2) = r6;
    *(++ f2) = r7;
  }
  
  if (remainder)
    return bg_wcopy (f2+1, f1+1, remainder);

  return f2+1;
}

#endif
