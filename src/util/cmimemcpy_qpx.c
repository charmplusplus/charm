#include <string.h>
#include "cmimemcpy_qpx.h"

#define QPX_LOAD(si,sb,fp) \
  do {									\
  asm volatile("qvlfdx %0,%1,%2": "=f"(fp) : "b" (si), "r" (sb));	\
  } while(0);

#define QPX_STORE(si,sb,fp)						\
  do {									\
  asm volatile("qvstfdx %2,%0,%1": : "b" (si), "r" (sb), "f"(fp) :"memory"); \
  } while(0);

#ifndef __GNUC__
#define FP_REG(i)   asm("f"#i)
#define FP_REG1(i)  "fr"#i
#else
#define FP_REG(i)  asm("fr"#i)
#define FP_REG1(i)  "fr"#i
#endif

//Copy 512 bytes from a 32b aligned pointers
static inline size_t quad_copy_512( char* dest, char* src ) {
    register double *fpp1_1, *fpp1_2;
    register double *fpp2_1, *fpp2_2;

    register double f0 FP_REG(0);
    register double f1 FP_REG(1);
    register double f2 FP_REG(2);
    register double f3 FP_REG(3);
    register double f4 FP_REG(4);
    register double f5 FP_REG(5);
    register double f6 FP_REG(6);
    register double f7 FP_REG(7);

    int r0;
    int r1;
    int r2;
    int r3;
    int r4;
    int r5;
    int r6;
    int r7;
    r0 = 0;
    r1 = 64;
    r2 = 128;
    r3 = 192;
    r4 = 256;
    r5 = 320;
    r6 = 384;
    r7 = 448;

    fpp1_1 = (double *)src;
    fpp1_2 = (double *)src +4;

    fpp2_1 = (double *)dest;
    fpp2_2 = (double *)dest +4;

    QPX_LOAD(fpp1_1,r0,f0);
    //asm volatile("qvlfdx 0,%0,%1": : "Ob" (fpp1_1), "r"(r0) :"memory");
    QPX_LOAD(fpp1_1,r1,f1);
    QPX_LOAD(fpp1_1,r2,f2);
    QPX_LOAD(fpp1_1,r3,f3);
    QPX_LOAD(fpp1_1,r4,f4);
    QPX_LOAD(fpp1_1,r5,f5);
    QPX_LOAD(fpp1_1,r6,f6);
    QPX_LOAD(fpp1_1,r7,f7);

    QPX_STORE(fpp2_1,r0,f0);
    QPX_LOAD(fpp1_2,r0,f0);
    QPX_STORE(fpp2_1,r1,f1);
    QPX_LOAD(fpp1_2,r1,f1);
    QPX_STORE(fpp2_1,r2,f2);
    QPX_LOAD(fpp1_2,r2,f2);
    QPX_STORE(fpp2_1,r3,f3);
    QPX_LOAD(fpp1_2,r3,f3);
    QPX_STORE(fpp2_1,r4,f4);
    QPX_LOAD(fpp1_2,r4,f4);
    QPX_STORE(fpp2_1,r5,f5);
    QPX_LOAD(fpp1_2,r5,f5);
    QPX_STORE(fpp2_1,r6,f6);
    QPX_LOAD(fpp1_2,r6,f6);
    QPX_STORE(fpp2_1,r7,f7);
    QPX_LOAD(fpp1_2,r7,f7);
 
    QPX_STORE(fpp2_2,r0,f0);
    QPX_STORE(fpp2_2,r1,f1);
    QPX_STORE(fpp2_2,r2,f2);
    QPX_STORE(fpp2_2,r3,f3);
    QPX_STORE(fpp2_2,r4,f4);
    QPX_STORE(fpp2_2,r5,f5);
    QPX_STORE(fpp2_2,r6,f6);
    QPX_STORE(fpp2_2,r7,f7);

    return 0;
}

void CmiMemcpy_qpx (void *dst, const void *src, size_t n)
{
  const char *s = src;
  char *d = dst;
  int n512 = n >> 9;
  while (n512 --) {
    quad_copy_512(d, s);
    d += 512;
    s += 512;
  }
 
  if ( (n & 511UL) != 0 )
    memcpy (d, s, n & 511UL);
}
