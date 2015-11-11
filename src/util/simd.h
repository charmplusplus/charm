#ifndef __SIMDIA_H__
#define __SIMDIA_H__


#if defined(__SSE2__) && !defined(_CRAYC)
  #include "emmintrin.h"
#endif

#if CMK_CELL_SPE != 0
  #include "spu_intrinsics.h"
#else
  #include "math.h"
#endif

#if defined(__VEC__)
  #include "altivec.h"
  #ifdef pixel
  #undef pixel
  #endif
  #ifdef bool
  #undef bool
  #endif
#endif


/* Solaris does not support sqrtf (float), so just map it to sqrt (double) instead */
#if !CMK_HAS_SQRTF
  #define sqrtf(a) ((float)(sqrt((double)(a))))
#endif



/* Flags to force architecture specific SIMD instructions off */
#define SIMDIA_FORCE_NO_SSE       (0)
#define SIMDIA_FORCE_NO_ALTIVEC   (0)
#define SIMDIA_FORCE_NO_SPE_SIMD  (0)


/***** Math Constants *****/
#define SIMDIA_CONSTANT_PI      (3.141592653589793)
#define SIMDIA_CONSTANT_E       (2.718281828459045)
#define SIMDIA_CONSTANT_SQRT_2  (1.414213562373095)


/* TODO | FIXME - Find platform independent way of ensuring alignment
 * (using __attribute__((aligned(XXX))) doesn't seem to work in net-win and
 * net-sol builds).  Just to be safe since compilers should do this anyway.
 */

/* TODO | FIXME - Add a function that will test the functionality of the
 * various operations defined by these abstractions and somehow tie this test
 * into the nightly build to ensure these operations give correct results.
 */


/*******************************************************************************
 *******************************************************************************
 ***** Generic C Implementation
 *******************************************************************************
 *******************************************************************************/

/*@{*/

/* NOTE: This is declared first so any architecture specific implementations
 *   can simply use the generic functions for specific data types or operations
 *   that they do not implement.
 */
 
/***** Data Types *****/
/* NOTE (DMK): Since this is the generic implementation, arbitrarily choosing 128 byte "vector" size. */
typedef struct __simdia_vec_i  {    int v0, v1, v2, v3; }  __simdia_veci;
typedef struct __simdia_vec_f  {  float v0, v1, v2, v3; }  __simdia_vecf;
typedef struct __simdia_vec_lf { double v0, v1;         } __simdia_veclf;


/***** Insert *****/
inline  __simdia_veci  __simdia_vinserti( __simdia_veci v, const    int s, const int i) {  __simdia_veci r = v;    int* rPtr = (   int*)(&r); rPtr[i] = s; return r; }
inline  __simdia_vecf  __simdia_vinsertf( __simdia_vecf v, const  float s, const int i) {  __simdia_vecf r = v;  float* rPtr = ( float*)(&r); rPtr[i] = s; return r; }
inline __simdia_veclf __simdia_vinsertlf(__simdia_veclf v, const double s, const int i) { __simdia_veclf r = v; double* rPtr = (double*)(&r); rPtr[i] = s; return r; }

/***** Extract *****/
inline    int  __simdia_vextracti( __simdia_veci v, const int i) {    int* vPtr = (   int*)(&v); return vPtr[i]; }
inline  float  __simdia_vextractf( __simdia_vecf v, const int i) {  float* vPtr = ( float*)(&v); return vPtr[i]; }
inline double __simdia_vextractlf(__simdia_veclf v, const int i) { double* vPtr = (double*)(&v); return vPtr[i]; }

/***** Set *****/
inline  __simdia_veci  __simdia_vseti(const    int a) {  __simdia_veci r; r.v0 = r.v1 = r.v2 = r.v3 = a; return r; }
inline  __simdia_vecf  __simdia_vsetf(const  float a) {  __simdia_vecf r; r.v0 = r.v1 = r.v2 = r.v3 = a; return r; }
inline __simdia_veclf __simdia_vsetlf(const double a) { __simdia_veclf r; r.v0 = r.v1 =               a; return r; }

/* NOTE: Would it be better to generate the constants instead of read them from memory in the generic version? */

/***** Constant Zero *****/
const  __simdia_veci  __simdia_const_vzeroi = {   0 ,   0 ,   0 ,   0  };
const  __simdia_vecf  __simdia_const_vzerof = { 0.0f, 0.0f, 0.0f, 0.0f };
const __simdia_veclf __simdia_const_vzerolf = { 0.0 , 0.0              };

/***** Constant One *****/
const  __simdia_veci  __simdia_const_vonei = {   1 ,   1 ,   1 ,   1  };
const  __simdia_vecf  __simdia_const_vonef = { 1.0f, 1.0f, 1.0f, 1.0f };
const __simdia_veclf __simdia_const_vonelf = { 1.0 , 1.0              };

/***** Constant Two *****/
const  __simdia_veci  __simdia_const_vtwoi = {   2 ,   2 ,   2 ,   2  };
const  __simdia_vecf  __simdia_const_vtwof = { 2.0f, 2.0f, 2.0f, 2.0f };
const __simdia_veclf __simdia_const_vtwolf = { 2.0 , 2.0              };

/***** Constant Negative One *****/
const  __simdia_veci  __simdia_const_vnegonei = {   -1 ,   -1 ,   -1 ,   -1  };
const  __simdia_vecf  __simdia_const_vnegonef = { -1.0f, -1.0f, -1.0f, -1.0f };
const __simdia_veclf __simdia_const_vnegonelf = { -1.0 , -1.0                };

/* TODO | FIXME - Try to create constants such that it does not require a
 * memory operations to access the constants (like the SSE constants).
 */

/***** Rotate *****/
inline  __simdia_veci  __simdia_vrothi(const  __simdia_veci a, int s) {  __simdia_veci b;    int* a_ptr = (   int*)(&a);    int* b_ptr = (   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
inline  __simdia_vecf  __simdia_vrothf(const  __simdia_vecf a, int s) {  __simdia_vecf b;  float* a_ptr = ( float*)(&a);  float* b_ptr = ( float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
inline __simdia_veclf __simdia_vrothlf(const __simdia_veclf a, int s) { __simdia_veclf b; double* a_ptr = (double*)(&a); double* b_ptr = (double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0-s)&0x1]; b_ptr[1] = a_ptr[(1-s)&0x1]; return b; }
inline  __simdia_veci  __simdia_vrotli(const  __simdia_veci a, int s) {  __simdia_veci b;    int* a_ptr = (   int*)(&a);    int* b_ptr = (   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
inline  __simdia_vecf  __simdia_vrotlf(const  __simdia_vecf a, int s) {  __simdia_vecf b;  float* a_ptr = ( float*)(&a);  float* b_ptr = ( float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
inline __simdia_veclf __simdia_vrotllf(const __simdia_veclf a, int s) { __simdia_veclf b; double* a_ptr = (double*)(&a); double* b_ptr = (double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0+s)&0x1]; b_ptr[1] = a_ptr[(1+s)&0x1]; return b; }

/***** Addition *****/
inline  __simdia_veci  __simdia_vaddi(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = a.v0 + b.v0; r.v1 = a.v1 + b.v1; r.v2 = a.v2 + b.v2; r.v3 = a.v3 + b.v3; return r; }
inline  __simdia_vecf  __simdia_vaddf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; r.v0 = a.v0 + b.v0; r.v1 = a.v1 + b.v1; r.v2 = a.v2 + b.v2; r.v3 = a.v3 + b.v3; return r; }
inline __simdia_veclf __simdia_vaddlf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; r.v0 = a.v0 + b.v0; r.v1 = a.v1 + b.v1;                                         return r; }

/***** Subtraction *****/
inline  __simdia_veci  __simdia_vsubi(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = a.v0 - b.v0; r.v1 = a.v1 - b.v1; r.v2 = a.v2 - b.v2; r.v3 = a.v3 - b.v3; return r; }
inline  __simdia_vecf  __simdia_vsubf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; r.v0 = a.v0 - b.v0; r.v1 = a.v1 - b.v1; r.v2 = a.v2 - b.v2; r.v3 = a.v3 - b.v3; return r; }
inline __simdia_veclf __simdia_vsublf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; r.v0 = a.v0 - b.v0; r.v1 = a.v1 - b.v1;                                         return r; }

/***** Multiplication *****/
inline  __simdia_veci  __simdia_vmuli(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = a.v0 * b.v0; r.v1 = a.v1 * b.v1; r.v2 = a.v2 * b.v2; r.v3 = a.v3 * b.v3; return r; }
inline  __simdia_vecf  __simdia_vmulf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; r.v0 = a.v0 * b.v0; r.v1 = a.v1 * b.v1; r.v2 = a.v2 * b.v2; r.v3 = a.v3 * b.v3; return r; }
inline __simdia_veclf __simdia_vmullf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; r.v0 = a.v0 * b.v0; r.v1 = a.v1 * b.v1;                                         return r; }

/***** Division *****/
inline  __simdia_veci  __simdia_vdivi(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = a.v0 / b.v0; r.v1 = a.v1 / b.v1; r.v2 = a.v2 / b.v2; r.v3 = a.v3 / b.v3; return r; }
inline  __simdia_vecf  __simdia_vdivf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; r.v0 = a.v0 / b.v0; r.v1 = a.v1 / b.v1; r.v2 = a.v2 / b.v2; r.v3 = a.v3 / b.v3; return r; }
inline __simdia_veclf __simdia_vdivlf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; r.v0 = a.v0 / b.v0; r.v1 = a.v1 / b.v1;                                         return r; }

/***** Fused Multiply Add *****/
inline  __simdia_veci  __simdia_vmaddi(const  __simdia_veci a, const  __simdia_veci b, const  __simdia_veci c) {  __simdia_veci r; r.v0 = a.v0 * b.v0 + c.v0; r.v1 = a.v1 * b.v1 + c.v1; r.v2 = a.v2 * b.v2 + c.v2; r.v3 = a.v3 * b.v3 + c.v3; return r; }
inline  __simdia_vecf  __simdia_vmaddf(const  __simdia_vecf a, const  __simdia_vecf b, const  __simdia_vecf c) {  __simdia_vecf r; r.v0 = a.v0 * b.v0 + c.v0; r.v1 = a.v1 * b.v1 + c.v1; r.v2 = a.v2 * b.v2 + c.v2; r.v3 = a.v3 * b.v3 + c.v3; return r; }
inline __simdia_veclf __simdia_vmaddlf(const __simdia_veclf a, const __simdia_veclf b, const __simdia_veclf c) { __simdia_veclf r; r.v0 = a.v0 * b.v0 + c.v0; r.v1 = a.v1 * b.v1 + c.v1;                                                       return r; }

/***** Reciprocal *****/
/* TODO | FIXME  - See if there is a better way to do this (few cycles and avoid the memory load) */
inline  __simdia_vecf  __simdia_vrecipf(const  __simdia_vecf a) {  __simdia_vecf r; r.v0 = 1.0f / a.v0; r.v1 = 1.0f / a.v1; r.v2 = 1.0f / a.v2; r.v3 = 1.0f / a.v3; return r; }
inline __simdia_veclf __simdia_vreciplf(const __simdia_veclf a) { __simdia_veclf r; r.v0 = 1.0f / a.v0; r.v1 = 1.0f / a.v1; return r; }

/***** Square Root *****/
inline  __simdia_vecf  __simdia_vsqrtf(const  __simdia_vecf a) {  __simdia_vecf r; r.v0 = sqrtf(a.v0); r.v1 = sqrtf(a.v1); r.v2 = sqrtf(a.v2); r.v3 = sqrtf(a.v3); return r; }
inline __simdia_veclf __simdia_vsqrtlf(const __simdia_veclf a) { __simdia_veclf r; r.v0 = sqrt(a.v0); r.v1 = sqrt(a.v1); return r; }

/***** Reciprocal Square Root *****/
inline  __simdia_vecf  __simdia_vrsqrtf(const  __simdia_vecf a) {  __simdia_vecf r; r.v0 = 1.0f / sqrtf(a.v0); r.v1 = 1.0f / sqrtf(a.v1); r.v2 = 1.0f / sqrtf(a.v2); r.v3 = 1.0f / sqrtf(a.v3); return r; }
inline __simdia_veclf __simdia_vrsqrtlf(const __simdia_veclf a) { __simdia_veclf r; r.v0 = 1.0 / sqrt(a.v0); r.v1 = 1.0 / sqrt(a.v1); return r; }

/***** Not *****/
inline  __simdia_veci  __simdia_vnoti(const  __simdia_veci a) {  __simdia_veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); rPtr[0] = aPtr[0] ^ -1; rPtr[1] = aPtr[1] ^ -1; rPtr[2] = aPtr[2] ^ -1; rPtr[3] = aPtr[3] ^ -1; return r; }
inline  __simdia_vecf  __simdia_vnotf(const  __simdia_vecf a) {  __simdia_vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); rPtr[0] = aPtr[0] ^ -1; rPtr[1] = aPtr[1] ^ -1; rPtr[2] = aPtr[2] ^ -1; rPtr[3] = aPtr[3] ^ -1; return r; }
inline __simdia_veclf __simdia_vnotlf(const __simdia_veclf a) { __simdia_veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); rPtr[0] = aPtr[0] ^ -1; rPtr[1] = aPtr[1] ^ -1; rPtr[2] = aPtr[2] ^ -1; rPtr[3] = aPtr[3] ^ -1; return r; }

/***** Or *****/
inline  __simdia_veci  __simdia_vori(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] | bPtr[0]; rPtr[1] = aPtr[1] | bPtr[1]; rPtr[2] = aPtr[2] | bPtr[2]; rPtr[3] = aPtr[3] | bPtr[3]; return r; }
inline  __simdia_vecf  __simdia_vorf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] | bPtr[0]; rPtr[1] = aPtr[1] | bPtr[1]; rPtr[2] = aPtr[2] | bPtr[2]; rPtr[3] = aPtr[3] | bPtr[3]; return r; }
inline __simdia_veclf __simdia_vorlf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] | bPtr[0]; rPtr[1] = aPtr[1] | bPtr[1]; rPtr[2] = aPtr[2] | bPtr[2]; rPtr[3] = aPtr[3] | bPtr[3]; return r; }

/***** Nor *****/
inline  __simdia_veci  __simdia_vnori(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] | bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] | bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] | bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] | bPtr[3]) ^ -1; return r; }
inline  __simdia_vecf  __simdia_vnorf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] | bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] | bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] | bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] | bPtr[3]) ^ -1; return r; }
inline __simdia_veclf __simdia_vnorlf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] | bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] | bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] | bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] | bPtr[3]) ^ -1; return r; }

/***** And *****/
inline  __simdia_veci  __simdia_vandi(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] & bPtr[0]; rPtr[1] = aPtr[1] & bPtr[1]; rPtr[2] = aPtr[2] & bPtr[2]; rPtr[3] = aPtr[3] & bPtr[3]; return r; }
inline  __simdia_vecf  __simdia_vandf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] & bPtr[0]; rPtr[1] = aPtr[1] & bPtr[1]; rPtr[2] = aPtr[2] & bPtr[2]; rPtr[3] = aPtr[3] & bPtr[3]; return r; }
inline __simdia_veclf __simdia_vandlf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] & bPtr[0]; rPtr[1] = aPtr[1] & bPtr[1]; rPtr[2] = aPtr[2] & bPtr[2]; rPtr[3] = aPtr[3] & bPtr[3]; return r; }

/***** Nand *****/
inline  __simdia_veci  __simdia_vnandi(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] & bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] & bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] & bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] & bPtr[3]) ^ -1; return r; }
inline  __simdia_vecf  __simdia_vnandf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] & bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] & bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] & bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] & bPtr[3]) ^ -1; return r; }
inline __simdia_veclf __simdia_vnandlf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] & bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] & bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] & bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] & bPtr[3]) ^ -1; return r; }

/***** Xor *****/
inline  __simdia_veci  __simdia_vxori(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] ^ bPtr[0]; rPtr[1] = aPtr[1] ^ bPtr[1]; rPtr[2] = aPtr[2] ^ bPtr[2]; rPtr[3] = aPtr[3] ^ bPtr[3]; return r; }
inline  __simdia_vecf  __simdia_vxorf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] ^ bPtr[0]; rPtr[1] = aPtr[1] ^ bPtr[1]; rPtr[2] = aPtr[2] ^ bPtr[2]; rPtr[3] = aPtr[3] ^ bPtr[3]; return r; }
inline __simdia_veclf __simdia_vxorlf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] ^ bPtr[0]; rPtr[1] = aPtr[1] ^ bPtr[1]; rPtr[2] = aPtr[2] ^ bPtr[2]; rPtr[3] = aPtr[3] ^ bPtr[3]; return r; }

/***** Nxor *****/
inline  __simdia_veci  __simdia_vnxori(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] ^ bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] ^ bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] ^ bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] ^ bPtr[3]) ^ -1; return r; }
inline  __simdia_vecf  __simdia_vnxorf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] ^ bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] ^ bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] ^ bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] ^ bPtr[3]) ^ -1; return r; }
inline __simdia_veclf __simdia_vnxorlf(const __simdia_veclf a, const __simdia_veclf b) { __simdia_veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] ^ bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] ^ bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] ^ bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] ^ bPtr[3]) ^ -1; return r; }

/* TODO | FIXME - Try to do the comparisons in a branchless way */

/***** Equal To *****/
inline __simdia_veci  __simdia_vcmpeqi(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = ((a.v0 == b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 == b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 == b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 == b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci  __simdia_vcmpeqf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = ((a.v0 == b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 == b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 == b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 == b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci __simdia_vcmpeqlf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = r.v1 = ((a.v0 == b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 == b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }

/***** Greater Than *****/
inline __simdia_veci  __simdia_vcmpgti(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = ((a.v0 > b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 > b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 > b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 > b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci  __simdia_vcmpgtf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = ((a.v0 > b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 > b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 > b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 > b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci __simdia_vcmpgtlf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = r.v1 = ((a.v0 > b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 > b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }

/***** Greater Than Or Equal To *****/
inline __simdia_veci  __simdia_vcmpgei(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = ((a.v0 >= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 >= b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 >= b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 >= b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci  __simdia_vcmpgef(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = ((a.v0 >= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 >= b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 >= b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 >= b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci __simdia_vcmpgelf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = r.v1 = ((a.v0 >= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 >= b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }

/***** Less Than *****/
inline __simdia_veci  __simdia_vcmplti(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = ((a.v0 < b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 < b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 < b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 < b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci  __simdia_vcmpltf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = ((a.v0 < b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 < b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 < b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 < b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci __simdia_vcmpltlf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = r.v1 = ((a.v0 < b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 < b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }

/***** Less Than Or Equal To *****/
inline __simdia_veci  __simdia_vcmplei(const  __simdia_veci a, const  __simdia_veci b) {  __simdia_veci r; r.v0 = ((a.v0 <= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 <= b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 <= b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 <= b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci  __simdia_vcmplef(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = ((a.v0 <= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 <= b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 <= b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 <= b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __simdia_veci __simdia_vcmplelf(const  __simdia_vecf a, const  __simdia_vecf b) {  __simdia_veci r; r.v0 = r.v1 = ((a.v0 <= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 <= b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }


/*******************************************************************************
 ***** C++ Operators for Generic Implementation
 *******************************************************************************/
#if defined(__cplusplus)

  /***** Addition *****/
  inline  __simdia_veci operator+(const  __simdia_veci &a, const  __simdia_veci &b) { return  __simdia_vaddi(a, b); }
  inline  __simdia_vecf operator+(const  __simdia_vecf &a, const  __simdia_vecf &b) { return  __simdia_vaddf(a, b); }
  inline __simdia_veclf operator+(const __simdia_veclf &a, const __simdia_veclf &b) { return __simdia_vaddlf(a, b); }
  inline  __simdia_veci operator+=( __simdia_veci &a, const  __simdia_veci &b) { a =  __simdia_vaddi(a, b); return a; }
  inline  __simdia_vecf operator+=( __simdia_vecf &a, const  __simdia_vecf &b) { a =  __simdia_vaddf(a, b); return a; }
  inline __simdia_veclf operator+=(__simdia_veclf &a, const __simdia_veclf &b) { a = __simdia_vaddlf(a, b); return a; }

  inline  __simdia_veci operator+(const  __simdia_veci &a, const    int &b) { return  __simdia_vaddi(a,  __simdia_vseti(b)); }
  inline  __simdia_vecf operator+(const  __simdia_vecf &a, const  float &b) { return  __simdia_vaddf(a,  __simdia_vsetf(b)); }
  inline __simdia_veclf operator+(const __simdia_veclf &a, const double &b) { return __simdia_vaddlf(a, __simdia_vsetlf(b)); }
  inline  __simdia_veci operator+=( __simdia_veci &a, const    int &b) { a =  __simdia_vaddi(a,  __simdia_vseti(b)); return a; }
  inline  __simdia_vecf operator+=( __simdia_vecf &a, const  float &b) { a =  __simdia_vaddf(a,  __simdia_vsetf(b)); return a; }
  inline __simdia_veclf operator+=(__simdia_veclf &a, const double &b) { a = __simdia_vaddlf(a, __simdia_vsetlf(b)); return a; }

  /***** Subtraction *****/
  inline  __simdia_veci operator-(const  __simdia_veci &a, const  __simdia_veci &b) { return  __simdia_vsubi(a, b); }
  inline  __simdia_vecf operator-(const  __simdia_vecf &a, const  __simdia_vecf &b) { return  __simdia_vsubf(a, b); }
  inline __simdia_veclf operator-(const __simdia_veclf &a, const __simdia_veclf &b) { return __simdia_vsublf(a, b); }
  inline  __simdia_veci operator-=( __simdia_veci &a, const  __simdia_veci &b) { a =  __simdia_vsubi(a, b); return a; }
  inline  __simdia_vecf operator-=( __simdia_vecf &a, const  __simdia_vecf &b) { a =  __simdia_vsubf(a, b); return a; }
  inline __simdia_veclf operator-=(__simdia_veclf &a, const __simdia_veclf &b) { a = __simdia_vsublf(a, b); return a; }

  inline  __simdia_veci operator-(const  __simdia_veci &a, const    int &b) { return  __simdia_vsubi(a,  __simdia_vseti(b)); }
  inline  __simdia_vecf operator-(const  __simdia_vecf &a, const  float &b) { return  __simdia_vsubf(a,  __simdia_vsetf(b)); }
  inline __simdia_veclf operator-(const __simdia_veclf &a, const double &b) { return __simdia_vsublf(a, __simdia_vsetlf(b)); }
  inline  __simdia_veci operator-=( __simdia_veci &a, const    int &b) { a =  __simdia_vsubi(a,  __simdia_vseti(b)); return a; }
  inline  __simdia_vecf operator-=( __simdia_vecf &a, const  float &b) { a =  __simdia_vsubf(a,  __simdia_vsetf(b)); return a; }
  inline __simdia_veclf operator-=(__simdia_veclf &a, const double &b) { a = __simdia_vsublf(a, __simdia_vsetlf(b)); return a; }

  /***** Multiplication *****/
  inline  __simdia_vecf operator*(const  __simdia_vecf &a, const  __simdia_vecf &b) { return  __simdia_vmulf(a, b); }
  inline __simdia_veclf operator*(const __simdia_veclf &a, const __simdia_veclf &b) { return __simdia_vmullf(a, b); }
  inline  __simdia_vecf operator*=( __simdia_vecf &a, const  __simdia_vecf &b) { a =  __simdia_vmulf(a, b); return a; }
  inline __simdia_veclf operator*=(__simdia_veclf &a, const __simdia_veclf &b) { a = __simdia_vmullf(a, b); return a; }

  inline  __simdia_vecf operator*(const  __simdia_vecf &a, const  float &b) { return  __simdia_vmulf(a,  __simdia_vsetf(b)); }
  inline __simdia_veclf operator*(const __simdia_veclf &a, const double &b) { return __simdia_vmullf(a, __simdia_vsetlf(b)); }
  inline  __simdia_vecf operator*=( __simdia_vecf &a, const  float &b) { a =  __simdia_vmulf(a,  __simdia_vsetf(b)); return a; }
  inline __simdia_veclf operator*=(__simdia_veclf &a, const double &b) { a = __simdia_vmullf(a, __simdia_vsetlf(b)); return a; }

  /***** Division *****/
  inline  __simdia_vecf operator/(const  __simdia_vecf &a, const  __simdia_vecf &b) { return  __simdia_vdivf(a, b); }
  inline __simdia_veclf operator/(const __simdia_veclf &a, const __simdia_veclf &b) { return __simdia_vdivlf(a, b); }
  inline  __simdia_vecf operator/=( __simdia_vecf &a, const  __simdia_vecf &b) { a =  __simdia_vdivf(a, b); return a; }
  inline __simdia_veclf operator/=(__simdia_veclf &a, const __simdia_veclf &b) { a = __simdia_vdivlf(a, b); return a; }

  inline  __simdia_vecf operator/(const  __simdia_vecf &a, const  float &b) { return  __simdia_vdivf(a,  __simdia_vsetf(b)); }
  inline __simdia_veclf operator/(const __simdia_veclf &a, const double &b) { return __simdia_vdivlf(a, __simdia_vsetlf(b)); }
  inline  __simdia_vecf operator/=( __simdia_vecf &a, const  float &b) { a =  __simdia_vdivf(a,  __simdia_vsetf(b)); return a; }
  inline __simdia_veclf operator/=(__simdia_veclf &a, const double &b) { a = __simdia_vdivlf(a, __simdia_vsetlf(b)); return a; }

  /***** Or *****/
  inline  __simdia_veci operator|(const  __simdia_veci &a, const  __simdia_veci &b) { return  __simdia_vori(a, b); }
  inline  __simdia_vecf operator|(const  __simdia_vecf &a, const  __simdia_vecf &b) { return  __simdia_vorf(a, b); }
  inline __simdia_veclf operator|(const __simdia_veclf &a, const __simdia_veclf &b) { return __simdia_vorlf(a, b); }
  inline  __simdia_veci operator|=( __simdia_veci &a, const  __simdia_veci &b) { a =  __simdia_vori(a, b); return a; }
  inline  __simdia_vecf operator|=( __simdia_vecf &a, const  __simdia_vecf &b) { a =  __simdia_vorf(a, b); return a; }
  inline __simdia_veclf operator|=(__simdia_veclf &a, const __simdia_veclf &b) { a = __simdia_vorlf(a, b); return a; }

  inline  __simdia_veci operator|(const  __simdia_veci &a, const    int &b) { return  __simdia_vori(a,  __simdia_vseti(b)); }
  inline  __simdia_vecf operator|(const  __simdia_vecf &a, const  float &b) { return  __simdia_vorf(a,  __simdia_vsetf(b)); }
  inline __simdia_veclf operator|(const __simdia_veclf &a, const double &b) { return __simdia_vorlf(a, __simdia_vsetlf(b)); }
  inline  __simdia_veci operator|=( __simdia_veci &a, const    int &b) { a =  __simdia_vori(a,  __simdia_vseti(b)); return a; }
  inline  __simdia_vecf operator|=( __simdia_vecf &a, const  float &b) { a =  __simdia_vorf(a,  __simdia_vsetf(b)); return a; }
  inline __simdia_veclf operator|=(__simdia_veclf &a, const double &b) { a = __simdia_vorlf(a, __simdia_vsetlf(b)); return a; }

  /***** And *****/
  inline  __simdia_veci operator&(const  __simdia_veci &a, const  __simdia_veci &b) { return  __simdia_vandi(a, b); }
  inline  __simdia_vecf operator&(const  __simdia_vecf &a, const  __simdia_vecf &b) { return  __simdia_vandf(a, b); }
  inline __simdia_veclf operator&(const __simdia_veclf &a, const __simdia_veclf &b) { return __simdia_vandlf(a, b); }
  inline  __simdia_veci operator&=( __simdia_veci &a, const  __simdia_veci &b) { a =  __simdia_vandi(a, b); return a; }
  inline  __simdia_vecf operator&=( __simdia_vecf &a, const  __simdia_vecf &b) { a =  __simdia_vandf(a, b); return a; }
  inline __simdia_veclf operator&=(__simdia_veclf &a, const __simdia_veclf &b) { a = __simdia_vandlf(a, b); return a; }

  inline  __simdia_veci operator&(const  __simdia_veci &a, const    int &b) { return  __simdia_vandi(a,  __simdia_vseti(b)); }
  inline  __simdia_vecf operator&(const  __simdia_vecf &a, const  float &b) { return  __simdia_vandf(a,  __simdia_vsetf(b)); }
  inline __simdia_veclf operator&(const __simdia_veclf &a, const double &b) { return __simdia_vandlf(a, __simdia_vsetlf(b)); }
  inline  __simdia_veci operator&=( __simdia_veci &a, const    int &b) { a =  __simdia_vandi(a,  __simdia_vseti(b)); return a; }
  inline  __simdia_vecf operator&=( __simdia_vecf &a, const  float &b) { a =  __simdia_vandf(a,  __simdia_vsetf(b)); return a; }
  inline __simdia_veclf operator&=(__simdia_veclf &a, const double &b) { a = __simdia_vandlf(a, __simdia_vsetlf(b)); return a; }

  /***** Xor *****/
  inline  __simdia_veci operator^(const  __simdia_veci &a, const  __simdia_veci &b) { return  __simdia_vxori(a, b); }
  inline  __simdia_vecf operator^(const  __simdia_vecf &a, const  __simdia_vecf &b) { return  __simdia_vxorf(a, b); }
  inline __simdia_veclf operator^(const __simdia_veclf &a, const __simdia_veclf &b) { return __simdia_vxorlf(a, b); }
  inline  __simdia_veci operator^=( __simdia_veci &a, const  __simdia_veci &b) { a =  __simdia_vxori(a, b); return a; }
  inline  __simdia_vecf operator^=( __simdia_vecf &a, const  __simdia_vecf &b) { a =  __simdia_vxorf(a, b); return a; }
  inline __simdia_veclf operator^=(__simdia_veclf &a, const __simdia_veclf &b) { a = __simdia_vxorlf(a, b); return a; }

  inline  __simdia_veci operator^(const  __simdia_veci &a, const    int &b) { return  __simdia_vxori(a,  __simdia_vseti(b)); }
  inline  __simdia_vecf operator^(const  __simdia_vecf &a, const  float &b) { return  __simdia_vxorf(a,  __simdia_vsetf(b)); }
  inline __simdia_veclf operator^(const __simdia_veclf &a, const double &b) { return __simdia_vxorlf(a, __simdia_vsetlf(b)); }
  inline  __simdia_veci operator^=( __simdia_veci &a, const    int &b) { a =  __simdia_vxori(a,  __simdia_vseti(b)); return a; }
  inline  __simdia_vecf operator^=( __simdia_vecf &a, const  float &b) { a =  __simdia_vxorf(a,  __simdia_vsetf(b)); return a; }
  inline __simdia_veclf operator^=(__simdia_veclf &a, const double &b) { a = __simdia_vxorlf(a, __simdia_vsetlf(b)); return a; }

#endif /* defined(__cplusplus) */

/*@}*/


/*******************************************************************************
 *******************************************************************************
 ***** SSE Support
 *******************************************************************************
 *******************************************************************************/
#if defined(__SSE2__) && (!(SIMDIA_FORCE_NO_SSE)) && !defined(_CRAYC)

  /* NOTE | TODO | FIXME : Add checks for various version of SSE.  For now, only
   *   support and assume that minimum level SSE2.
   */

  /***** Data Types *****/
  typedef __m128i  simdia_veci;
  typedef  __m128  simdia_vecf;
  typedef __m128d simdia_veclf;

  /***** Insert *****/
  /* TODO | FIXME - Try to make these functions not reference memory so values stay in registers */
  inline  simdia_veci  simdia_vinserti( simdia_veci v, const    int s, const int i) {  simdia_veci r = v;    int* rPtr = (   int*)(&r); rPtr[i] = s; return r; }
  inline  simdia_vecf  simdia_vinsertf( simdia_vecf v, const  float s, const int i) {  simdia_vecf r = v;  float* rPtr = ( float*)(&r); rPtr[i] = s; return r; }
  inline simdia_veclf simdia_vinsertlf(simdia_veclf v, const double s, const int i) { simdia_veclf r = v; double* rPtr = (double*)(&r); rPtr[i] = s; return r; }

  /***** Extract *****/
  /* TODO | FIXME - Try to make these functions not reference memory so values stay in registers */
  inline    int  vextracti( simdia_veci v, const int i) { return ((   int*)(&v))[i]; }
  inline  float  vextractf( simdia_vecf v, const int i) { return (( float*)(&v))[i]; }
  inline double vextractlf(simdia_veclf v, const int i) { return ((double*)(&v))[i]; }

  /***** Set *****/
  #define  simdia_vseti(a)  (_mm_set1_epi32((int)(a)))
  #define  simdia_vsetf(a)  (_mm_set1_ps((float)(a)))
  #define simdia_vsetlf(a)  (_mm_set1_pd((double)(a)))

  /***** Constant Zero *****/
  #define  simdia_const_vzeroi  (_mm_setzero_si128())
  #define  simdia_const_vzerof  (_mm_setzero_ps())
  #define simdia_const_vzerolf  (_mm_setzero_pd())

  /***** Constant One *****/
  #define  simdia_const_vonei  (simdia_vseti(1))
  #define  simdia_const_vonef  (simdia_vsetf(1.0f))
  #define simdia_const_vonelf  (simdia_vsetlf(1.0))

  /***** Constant Two *****/
  #define  simdia_const_vtwoi  (simdia_vseti(2))
  #define  simdia_const_vtwof  (simdia_vsetf(2.0f))
  #define simdia_const_vtwolf  (simdia_vsetlf(2.0))

  /***** Constant Negative One *****/
  #define  simdia_const_vnegonei  (simdia_vseti(-1))
  #define  simdia_const_vnegonef  (simdia_vsetf(-1.0f))
  #define simdia_const_vnegonelf  (simdia_vsetlf(-1.0))

  /***** Rotate *****/
  /* TODO : FIXME - Find a better way to do Rotate in SSE */
  inline  simdia_veci  simdia_vrothi(const  simdia_veci &a, int s) {  simdia_veci b;    int* a_ptr = (   int*)(&a);    int* b_ptr = (   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
  inline  simdia_vecf  simdia_vrothf(const  simdia_vecf &a, int s) {  simdia_vecf b;  float* a_ptr = ( float*)(&a);  float* b_ptr = ( float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
  inline simdia_veclf simdia_vrothlf(const simdia_veclf &a, int s) { simdia_veclf b; double* a_ptr = (double*)(&a); double* b_ptr = (double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0-s)&0x1]; b_ptr[1] = a_ptr[(1-s)&0x1]; return b; }
  inline  simdia_veci  simdia_vrotli(const  simdia_veci &a, int s) {  simdia_veci b;    int* a_ptr = (   int*)(&a);    int* b_ptr = (   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
  inline  simdia_vecf  simdia_vrotlf(const  simdia_vecf &a, int s) {  simdia_vecf b;  float* a_ptr = ( float*)(&a);  float* b_ptr = ( float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
  inline simdia_veclf simdia_vrotllf(const simdia_veclf &a, int s) { simdia_veclf b; double* a_ptr = (double*)(&a); double* b_ptr = (double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0+s)&0x1]; b_ptr[1] = a_ptr[(1+s)&0x1]; return b; }

  /***** Addition *****/
  #define  simdia_vaddi(a, b)  (_mm_add_epi32((a), (b)))
  #define  simdia_vaddf(a, b)  (_mm_add_ps((a), (b)))
  #define simdia_vaddlf(a, b)  (_mm_add_pd((a), (b)))

  /***** Subtraction *****/
  #define  simdia_vsubi(a, b)  (_mm_sub_epi32((a), (b)))
  #define  simdia_vsubf(a, b)  (_mm_sub_ps((a), (b)))
  #define simdia_vsublf(a, b)  (_mm_sub_pd((a), (b)))

  /***** Multiplication *****/
  #define    simdia_vmulf(a, b)  (_mm_mul_ps((a), (b)))
  #define   simdia_vmullf(a, b)  (_mm_mul_pd((a), (b)))

  /***** Division *****/
  #define   simdia_vdivf(a, b)  (_mm_div_ps((a), (b)))
  #define  simdia_vdivlf(a, b)  (_mm_div_pd((a), (b)))

  /***** Fused Multiply Add *****/
  #define  simdia_vmaddf(a, b, c)  ( vaddf( vmulf((a), (b)), (c)))
  #define simdia_vmaddlf(a, b, c)  (vaddlf(vmullf((a), (b)), (c)))

  /***** Reciprocal *****/
  #define  simdia_vrecipf(a)  (_mm_rcp_ps(a))
  inline simdia_veclf simdia_vreciplf(const simdia_veclf a) { simdia_veclf r; double* a_ptr =  (double*)(&a); double* r_ptr = (double*)(&r); r_ptr[0] = 1.0f /  a_ptr[0]; r_ptr[1] = 1.0f / a_ptr[1]; return r; }

  /***** Square Root *****/
  #define  simdia_vsqrtf(a)  (_mm_sqrt_ps(a))
  #define simdia_vsqrtlf(a)  (_mm_sqrt_pd(a))

  /***** Reciprocal Square Root *****/
  #define  simdia_vrsqrtf(a)  (_mm_rsqrt_ps(a))
  #define simdia_vrsqrtlf(a)  (vreciplf(vsqrtlf(a)))

  /***** Not *****/
  #define  simdia_vnoti(a)  (_mm_xor_si128((a), simdia_const_vnegonei))
  #define  simdia_vnotf(a)  (_mm_xor_ps((a), simdia_const_vnegonei))
  #define simdia_vnotlf(a)  (_mm_xor_pd((a), simdia_const_vnegonei))

  /***** Or *****/
  #define  simdia_vori(a, b)  (_mm_or_si128((a), (b)))
  #define  simdia_vorf(a, b)  (_mm_or_ps((a), (b)))
  #define simdia_vorlf(a, b)  (_mm_or_pd((a), (b)))

  /***** Nor *****/
  #define  simdia_vnori(a, b)  ( simdia_vnoti( simdia_vori((a), (b))))
  #define  simdia_vnorf(a, b)  ( simdia_vnotf( simdia_vorf((a), (b))))
  #define simdia_vnorlf(a, b)  (simdia_vnotlf(simdia_vorlf((a), (b))))

  /***** And *****/
  #define  simdia_vandi(a, b)  (_mm_and_si128((a), (b)))
  #define  simdia_vandf(a, b)  (_mm_and_ps((a), (b)))
  #define simdia_vandlf(a, b)  (_mm_and_pd((a), (b)))

  /***** Nand *****/
  #define  simdia_vnandi(a, b)  ( simdia_vnoti( simdia_vandi((a), (b))))
  #define  simdia_vnandf(a, b)  ( simdia_vnotf( simdia_vandf((a), (b))))
  #define simdia_vnandlf(a, b)  (simdia_vnotlf(simdia_vandlf((a), (b))))

  /***** Xor *****/
  #define  simdia_vxori(a, b)  (_mm_xor_si128((a), (b)))
  #define  simdia_vxorf(a, b)  (_mm_xor_ps((a), (b)))
  #define simdia_vxorlf(a, b)  (_mm_xor_pd((a), (b)))

  /***** Nxor *****/
  #define  simdia_vnxori(a, b)  ( simdia_vnoti( simdia_vxori((a), (b))))
  #define  simdia_vnxorf(a, b)  ( simdia_vnotf( simdia_vxorf((a), (b))))
  #define simdia_vnxorlf(a, b)  (simdia_vnotlf(simdia_vxorlf((a), (b))))

  /***** Equal To *****/
  #define  simdia_vcmpeqi(a, b)  ((simdia_veci)(_mm_cmpeq_epi32((a), (b))))
  #define  simdia_vcmpeqf(a, b)  ((simdia_veci)(_mm_cmpeq_ps((a), (b))))
  #define simdia_vcmpeqlf(a, b)  ((simdia_veci)(_mm_cmpeq_pd((a), (b))))

  /***** Greater Than *****/
  #define  simdia_vcmpgti(a, b)  ((simdia_veci)(_mm_cmpgt_epi32((a), (b))))
  #define  simdia_vcmpgtf(a, b)  ((simdia_veci)(_mm_cmpgt_ps((a), (b))))
  #define simdia_vcmpgtlf(a, b)  ((simdia_veci)(_mm_cmpgt_pd((a), (b))))

  /***** Greater Than Or Equal To *****/
  #define  simdia_vcmpgei(a, b)  ((simdia_veci)(_mm_cmpge_epi32((a), (b))))
  #define  simdia_vcmpgef(a, b)  ((simdia_veci)(_mm_cmpge_ps((a), (b))))
  #define simdia_vcmpgelf(a, b)  ((simdia_veci)(_mm_cmpge_pd((a), (b))))

  /***** Less Than *****/
  #define  simdia_vcmplti(a, b)  ((simdia_veci)(_mm_cmplt_epi32((a), (b))))
  #define  simdia_vcmpltf(a, b)  ((simdia_veci)(_mm_cmplt_ps((a), (b))))
  #define simdia_vcmpltlf(a, b)  ((simdia_veci)(_mm_cmplt_pd((a), (b))))

  /***** Less Than Or Equal To *****/
  #define  simdia_vcmplei(a, b)  ((simdia_veci)(_mm_cmple_epi32((a), (b))))
  #define  simdia_vcmplef(a, b)  ((simdia_veci)(_mm_cmple_ps((a), (b))))
  #define simdia_vcmplelf(a, b)  ((simdia_veci)(_mm_cmple_pd((a), (b))))


/*******************************************************************************
 *******************************************************************************
 ***** SPE SIMD Instructions
 *******************************************************************************
 *******************************************************************************/
/* TODO | FIXME : Find a more general check for this (this is Charm++ specific) */
#elif (CMK_CELL_SPE != 0) && (!(SIMDIA_FORCE_NO_SPE_SIMD))

  /***** Data Types *****/
  typedef vector signed int  simdia_veci;
  typedef vector float       simdia_vecf;
  typedef vector double     simdia_veclf;

  /***** Insert *****/
  #define  simdia_vinserti(v, s, i)  (spu_insert((s), (v), (i)))
  #define  simdia_vinsertf(v, s, i)  (spu_insert((s), (v), (i)))
  #define simdia_vinsertlf(v, s, i)  (spu_insert((s), (v), (i)))

  /***** Extract *****/
  #define  simdia_vextracti(v, i)  (spu_extract((v), (i)))
  #define  simdia_vextractf(v, i)  (spu_extract((v), (i)))
  #define simdia_vextractlf(v, i)  (spu_extract((v), (i)))

  /***** Set *****/
  #define  simdia_vseti(a)  (spu_splats((int)(a)))
  #define  simdia_vsetf(a)  (spu_splats((float)(a)))
  #define simdia_vsetlf(a)  (spu_splats((double)(a)))

  /***** Constant Zero *****/
  #define  simdia_const_vzeroi  (vseti(0))
  #define  simdia_const_vzerof  (vsetf(0.0f))
  #define simdia_const_vzerolf  (vsetlf(0.0))

  /***** Constant One *****/
  #define  simdia_const_vonei  (vseti(1))
  #define  simdia_const_vonef  (vsetf(1.0f))
  #define simdia_const_vonelf  (vsetlf(1.0))

  /***** Constant Two *****/
  #define  simdia_const_vtwoi  (vseti(2))
  #define  simdia_const_vtwof  (vsetf(2.0f))
  #define simdia_const_vtwolf  (vsetlf(2.0))

  /***** Constant Negative One *****/
  #define  simdia_const_vnegonei  (vseti(-1))
  #define  simdia_const_vnegonef  (vsetf(-1.0f))
  #define simdia_const_vnegonelf  (vsetlf(-1.0))

  /***** Rotate *****/
  #define   simdia_vrothi(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x3)<<2)) ))
  #define   simdia_vrothf(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x3)<<2)) ))
  #define  simdia_vrothlf(a, s) (spu_rlqwbyte((a),       (((s)&0x1)<<3)  ))
  #define   simdia_vrotli(a, s) (spu_rlqwbyte((a), ((s)&0x3)<<2))
  #define   simdia_vrotlf(a, s) (spu_rlqwbyte((a), ((s)&0x3)<<2))
  #define  simdia_vrotllf(a, s) (spu_rlqwbyte((a), ((s)&0x1)<<3))

  /***** Addition *****/
  #define  simdia_vaddi(a, b)  (spu_add((a), (b)))
  #define  simdia_vaddf(a, b)  (spu_add((a), (b)))
  #define simdia_vaddlf(a, b)  (spu_add((a), (b)))

  /***** Subtraction *****/
  #define  simdia_vsubi(a, b)  (spu_sub((a), (b)))
  #define  simdia_vsubf(a, b)  (spu_sub((a), (b)))
  #define simdia_vsublf(a, b)  (spu_sub((a), (b)))

  /***** Multiplication *****/
  #define   simdia_vmulf(a, b)  (spu_mul((a), (b)))
  #define  simdia_vmullf(a, b)  (spu_mul((a), (b)))

  /***** Division *****/
  #define simdia_vdivf(a, b)  (spu_mul((a), spu_re(b)))
  inline simdia_veclf simdia_vdivlf(const simdia_veclf a, const simdia_veclf b) { simdia_veclf r = { 0.0, 0.0 }; spu_insert((spu_extract(a, 0) / spu_extract(b, 0)), r, 0); spu_insert((spu_extract(a, 1) / spu_extract(b, 1)), r, 1); return r; }

  /***** Fused Multiply Add *****/
  #define  simdia_vmaddf(a, b, c)  (spu_madd((a), (b), (c)))
  #define simdia_vmaddlf(a, b, c)  (spu_madd((a), (b), (c)))

  /***** Reciprocal *****/
  #define  simdia_vrecipf(a)  (spu_re(a))
  inline simdia_veclf simdia_vreciplf(const simdia_veclf a, const simdia_veclf b) { simdia_veclf r = { 0.0, 0.0 }; spu_insert((1.0f / spu_extract(a, 0)), r, 0); spu_insert((1.0f / spu_extract(a, 1)), r, 1); return r; }

  /***** Square Root *****/
  #define simdia_vsqrtf(a) (spu_re(spu_rsqrte(a)))
  inline simdia_veclf simdia_vsqrtlf(const simdia_veclf a, const simdia_veclf b) { simdia_veclf r = { 0.0, 0.0 }; spu_insert(sqrt(spu_extract(a, 0)), r, 0); spu_insert(sqrt(spu_extract(a, 1)), r, 1); return r; }

  /***** Reciprocal Square Root *****/
  #define simdia_vrsqrtf(a) (spu_rsqrte(a))
  inline simdia_veclf simdia_vrsqrtlf(const simdia_veclf a, const simdia_veclf b) { simdia_veclf r = { 0.0, 0.0 }; spu_insert((1.0f / sqrt(spu_extract(a, 0))), r, 0); spu_insert((1.0f / sqrt(spu_extract(a, 1))), r, 1); return r; }

  /***** Not *****/
  #define  simdia_vnoti(a)  (spu_nor((a), (a)))
  #define  simdia_vnotf(a)  (spu_nor((a), (a)))
  #define simdia_vnotlf(a)  (spu_nor((a), (a)))

  /***** Or *****/
  #define  simdia_vori(a, b)  (spu_or((a), (b)))
  #define  simdia_vorf(a, b)  (spu_or((a), (b)))
  #define simdia_vorlf(a, b)  (spu_or((a), (b)))

  /***** Nor *****/
  #define  simdia_vnori(a, b)  (spu_nor((a), (b)))
  #define  simdia_vnorf(a, b)  (spu_nor((a), (b)))
  #define simdia_vnorlf(a, b)  (spu_nor((a), (b)))

  /***** And *****/
  #define  simdia_vandi(a, b)  (spu_and((a), (b)))
  #define  simdia_vandf(a, b)  (spu_and((a), (b)))
  #define simdia_vandlf(a, b)  (spu_and((a), (b)))

  /***** Nand *****/
  #define  simdia_vnandi(a, b)  (spu_nand((a), (b)))
  #define  simdia_vnandf(a, b)  (spu_nand((a), (b)))
  #define simdia_vnandlf(a, b)  (spu_nand((a), (b)))

  /***** Xor *****/
  #define  simdia_vxori(a, b)  (spu_xor((a), (b)))
  #define  simdia_vxorf(a, b)  (spu_xor((a), (b)))
  #define simdia_vxorlf(a, b)  (spu_xor((a), (b)))

  /***** Nxor *****/
  #define  simdia_vnxori(a, b)  ( simdia_vnoti( simdia_vxori((a), (b))))
  #define  simdia_vnxorf(a, b)  ( simdia_vnotf( simdia_vxorf((a), (b))))
  #define simdia_vnxorlf(a, b)  (simdia_vnotlf(simdia_vxorlf((a), (b))))

  /***** Equal To *****/
  #define  simdia_vcmpeqi(a, b)  ((simdia_veci)(spu_cmpeq((a), (b))))
  #define  simdia_vcmpeqf(a, b)  ((simdia_veci)(spu_cmpeq((a), (b))))
  #define simdia_vcmpeqlf(a, b)  ((simdia_veci)(spu_cmpeq((a), (b))))

  /***** Greater Than *****/
  #define  simdia_vcmpgti(a, b)  ((simdia_veci)(spu_cmpgt((a), (b))))
  #define  simdia_vcmpgtf(a, b)  ((simdia_veci)(spu_cmpgt((a), (b))))
  #define simdia_vcmpgtlf(a, b)  ((simdia_veci)(spu_cmpgt((a), (b))))

  // NOTE : Try to create versions of >= and < that do not double evaluate their inputs

  /***** Greater Than or Equal To *****/
  #define  simdia_vcmpgei(a, b)  (spu_or( simdia_vcmpeqi((a), (b)),  simdia_vcmpgti((a), (b))))
  #define  simdia_vcmpgef(a, b)  (spu_or( simdia_vcmpeqf((a), (b)),  simdia_vcmpgtf((a), (b))))
  #define simdia_vcmpgelf(a, b)  (spu_or(simdia_vcmpeqlf((a), (b)), simdia_vcmpgtlf((a), (b))))

  /***** Less Than *****/
  #define  simdia_vcmplti(a, b)  (spu_nor( simdia_vcmpgti((a), (b)),  simdia_vcmpeqi((a), (b))))
  #define  simdia_vcmpltf(a, b)  (spu_nor( simdia_vcmpgtf((a), (b)),  simdia_vcmpeqf((a), (b))))
  #define simdia_vcmpltlf(a, b)  (spu_nor(simdia_vcmpgtlf((a), (b)), simdia_vcmpeqlf((a), (b))))

  /***** Less Than or Equal To *****/
  #define  simdia_vcmplei(a, b)  (spu_nor( simdia_vcmpgti((a), (b)),  simdia_const_vzeroi))
  #define  simdia_vcmplef(a, b)  (spu_nor( simdia_vcmpgtf((a), (b)),  simdia_const_vzerof))
  #define simdia_vcmplelf(a, b)  (spu_nor(simdia_vcmpgtlf((a), (b)), simdia_const_vzerolf))


/*******************************************************************************
 *******************************************************************************
 ***** AltiVec
 *******************************************************************************
 *******************************************************************************/
#elif defined(__VEC__) && (!(SIMDIA_FORCE_NO_ALTIVEC))

  /***** Data Types *****/
  typedef vector signed int simdia_veci;
  typedef vector float      simdia_vecf;
  #ifdef _ARCH_PWR7
    /** power 7 VSX supports 64 bit operands, it also includes VMX support
     * which means that things like vec_div, vec_insert, etcetera work for
     * ints floats and doubles.  These intrinsics also require a suitably
     * new version of the compiler on Power 7.  If you are somehow using a
     * Power 7 with an old compiler, please do not hesitate to open a can
     * of whoopass on whoever installed the tool chain, because that kind
     * of stupidity should not be tolerated.
     */ 
    typedef vector double  simdia_veclf;
  #else
    typedef __simdia_veclf simdia_veclf;
  #endif

  /***** Insert *****/
  /* TODO | FIXME - Try to make these functions not reference memory
     so values stay in registers */
  #ifdef _ARCH_PWR7 
    // swap argument order
    #define  simdia_vinserti(a, b, c)  (vec_insert((b)), ((a)), ((c)))
    #define  simdia_vinsertf(a, b, c)  (vec_insert((b)), ((a)), ((c)))
    #define  simdia_vinsertlf(a, b, c)  (vec_insert((b)), ((a)), ((c)))
  #else
    inline  simdia_veci  simdia_vinserti( simdia_veci v, const    int s, const int i) {  simdia_veci r = v;    int* rPtr = (   int*)(&r); rPtr[i] = s; return r; }
    inline  simdia_vecf  simdia_vinsertf( simdia_vecf v, const  float s, const int i) {  simdia_vecf r = v;  float* rPtr = ( float*)(&r); rPtr[i] = s; return r; }
    #define simdia_vinsertlf __simdia_vinsertlf
  #endif

  /***** Extract *****/
  #ifdef _ARCH_PWR7 
    #define  simdia_vextracti(a, b)  (vec_extract((a), (b)))
    #define  simdia_vextractf(a, b)  (vec_extract((a), (b)))
    #define  simdia_vextractlf(a, b)  (vec_extract((a), (b)))
  #else
    /* TODO | FIXME - Try to make these functions not reference memory so values stay in registers */
    inline    int  simdia_vextracti( simdia_veci v, const int i) {    int* vPtr = (   int*)(&v); return vPtr[i]; }
    inline  float  simdia_vextractf( simdia_vecf v, const int i) {  float* vPtr = ( float*)(&v); return vPtr[i]; }
    #define simdia_vextractlf __simdia_vextractlf
  #endif

  /***** Set *****/
  #ifdef _ARCH_PWR7 
    #define  simdia_vseti(a)  (vec_promote((a), 0))
    #define  simdia_vsetf(a)  (vec_promote((a), 0))
    #define  simdia_vsetlf(a)  (vec_promote((a), 0))
  #else
    /* TODO : FIXME - There must be a better way to do this, but it
    seems the only way to convert scalar to vector is to go through
    memory instructions.  

    EJB: converting between scalar and vector is the sort of thing you
    want to avoid doing on altivec.  Better to rethink and find a way to
    stay in the vector engine if at all possible.
    */

    inline simdia_veci simdia_vseti(const   int a) { __simdia_veci r; r.v0 = a; return vec_splat(*((simdia_veci*)(&r)), 0); }
    inline simdia_vecf simdia_vsetf(const float a) { __simdia_vecf r; r.v0 = a; return vec_splat(*((simdia_vecf*)(&r)), 0); }
    #define simdia_vsetlf __simdia_vsetlf
  #endif
  /* NOTE: Declare one for unsigned char vector also (required by rotate functions) */
  inline vector unsigned char simdia_vset16uc(const unsigned char c) { vector unsigned char r __attribute__((aligned(16))); ((unsigned char*)(&r))[0] = c; return vec_splat(r, 0); }

  /***** Constant Zero *****/
  #define  simdia_const_vzeroi  (vec_splat_s32(0))
  #define  simdia_const_vzerof  (vec_ctf(vec_splat_s32(0), 0))
  #ifdef _ARCH_PWR7 
    #define simdia_const_vzerolf  (vec_splats(0))
  #else
    #define simdia_const_vzerolf  (__simdia_const_vzerolf)
  #endif

  /***** Constant One *****/
  #define  simdia_const_vonei  (vec_splat_s32(1))
  #define  simdia_const_vonef  (vec_ctf(vec_splat_s32(1), 0))
  #ifdef _ARCH_PWR7 
    #define simdia_const_vonelf  (vec_splats(1))
  #else
    #define simdia_const_vonelf  (__simdia_const_vonelf)
  #endif

  /***** Constant Two *****/
  #define  simdia_const_vtwoi  (vec_splat_s32(2))
  #define  simdia_const_vtwof  (vec_ctf(vec_splat_s32(2), 0))
  #ifdef _ARCH_PWR7 
    #define simdia_const_vtwolf  (vec_splats(2))
  #else
    #define simdia_const_vtwolf  (__simdia_const_vtwolf)
  #endif

  /***** Constant Negative One *****/
  #define  simdia_const_vnegonei  (vec_splat_s32(-1))
  #define  simdia_const_vnegonef  (vec_ctf(vec_splat_s32(-1), 0))
  #ifdef _ARCH_PWR7 
    #define simdia_const_vnegonelf  (vec_splats(-1))
  #else
    #define simdia_const_vnegonelf  (__const_veclf)
  #endif

  /***** Rotate *****/
  #define __simdia_vrotlbytes(a, s)  (vec_or(vec_slo((a), simdia_vset16uc(((s) & 0xf) << 3)), vec_sro((a), simdia_set16uc((16 - ((s) & 0xf)) << 3))))
  #define __simdia_vrotrbytes(a, s)  (vec_or(vec_sro((a), simdia_vset16uc(((s) & 0xf) << 3)), vec_slo((a), simdia_set16uc((16 - ((s) & 0xf)) << 3))))
  #define  simdia_vrotli(a, s)  __simdia_vrotlbytes((a), ((s) << 2))
  #define  simdia_vrotlf(a, s)  __simdia_vrotlbytes((a), ((s) << 2))
  #define simdia_vrotllf(a, s)  __simdia_vrotlbytes((a), ((s) << 3))
  #define  simdia_vrothi(a, s)  __simdia_vrotrbytes((a), ((s) << 2))
  #define  simdia_vrothf(a, s)  __simdia_vrotrbytes((a), ((s) << 2))
  #define simdia_vrothlf(a, s)  __simdia_vrotrbytes((a), ((s) << 3))

  /***** Addition *****/
  #define  simdia_vaddi(a, b)  (vec_add((a), (b)))
  #define  simdia_vaddf(a, b)  (vec_add((a), (b)))
  #ifdef _ARCH_PWR7 
    #define  simdia_vaddlf(a, b)  (vec_add((a), (b)))
  #else
    #define simdia_vaddlf __simdia_vaddlf
  #endif

  /***** Subtraction *****/
  #define  simdia_vsubi(a, b)  (vec_sub((a), (b)))
  #define  simdia_vsubf(a, b)  (vec_sub((a), (b)))
  #ifdef _ARCH_PWR7 
    #define  simdia_vsublf(a, b)  (vec_sub((a), (b)))
  #else
    #define simdia_vsublf __simdia_vsublf
  #endif

  /***** Multiplication *****/
  // NOTE: Try to find a way to do this without double evaluating a
  #ifdef _ARCH_PWR7 
    #define  simdia_vmulf(a, b)  (vec_mul((a), (b)))
    #define  simdia_vmullf(a, b)  (vec_mul((a), (b)))
  #else
    #define  simdia_vmulf(a, b)  (vec_madd((a), (b), vec_xor((a), (a))))
    #define  simdia_vmullf __simdia_vmullf
  #endif

  /***** Division *****/
  #ifdef _ARCH_PWR7 
    #define simdia_vdivf(a, b)  (vec_div((a)), ((b)))
    #define simdia_vdivlf(a, b)  (vec_div((a)), ((b)))
  #else
    #define simdia_vdivf(a, b)  (simdia_vmulf((a), vec_re(b)))
    #define simdia_vdivlf __simdia_vdivlf
  #endif

  /***** Fused Multiply Add *****/
  #define simdia_vmaddf(a, b, c)  (vec_madd((a), (b), (c)))
  #ifdef _ARCH_PWR7 
    #define simdia_vmaddlf(a, b, c)  (vec_madd((a), (b), (c)))
  #else
    #define simdia_vmaddlf __simdia_vmaddlf
  #endif

  /***** Reciprocal *****/
  #define simdia_vrecipf(a)  (vec_re(a))
  #ifdef _ARCH_PWR7 
    #define simdia_vreciplf(a)  (vec_re(a))
  #else
    #define simdia_vreciplf __simdia_vreciplf
  #endif

  /***** Square Root *****/
  #define simdia_vsqrtf(a)  (vec_re(vec_rsqrte(a)))
  #ifdef _ARCH_PWR7 
    #define simdia_vsqrtlf(a)  (vec_sqrt(a))
  #else
    #define simdia_vsqrtlf __simdia_vsqrtlf
  #endif

  /***** Reciprocal Square Root *****/
  #define simdia_vrsqrtf(a)  (vec_rsqrte(a))
  #ifdef _ARCH_PWR7 
    #define simdia_vrsqrtlf(a)  (vec_rsqrte(a))
  #else
    #define simdia_vrsqrtlf __simdia_vrsqrtlf
  #endif

  /***** Not *****/
  #ifdef _ARCH_PWR7 
    #define simdia_vnoti(a)  (vec_neg(a))
    #define simdia_vnotf(a)  (vec_neg(a))
    #define simdia_vnotlf(a)  (vec_neg(a))
  #else
    #define simdia_vnoti(a)  (vec_xor((a), simdia_const_vnegonei))
    #define simdia_vnotf(a)  (vec_xor((a), simdia_const_vnegonei))
    #define simdia_vnotlf __simdia_vnotlf
  #endif

  /***** Or *****/
  #define simdia_vori(a, b)  (vec_or((a), (b)))
  #define simdia_vorf(a, b)  (vec_or((a), (b)))
  #ifdef _ARCH_PWR7 
    #define simdia_vorlf(a, b)  (vec_or((a), (b)))
  #else
    #define simdia_vorlf __simdia_vorlf
  #endif

  /***** Nor *****/
  #define simdia_vnori(a, b)  (vec_nor((a), (b)))
  #define simdia_vnorf(a, b)  (vec_nor((a), (b)))
  #ifdef _ARCH_PWR7 
    #define simdia_vnorlf(a, b)  (vec_nor((a), (b)))
  #else
    #define simdia_vnorlf __simdia_vnorlf
  #endif

  /***** And *****/
  #define simdia_vandi(a, b)  (vec_and((a), (b)))
  #define simdia_vandf(a, b)  (vec_and((a), (b)))
  #ifdef _ARCH_PWR7 
    #define simdia_vandlf(a, b)  (vec_and((a), (b)))
  #else
    #define simdia_vandlf __simdia_vandlf
  #endif

  /***** Nand *****/
  #define simdia_vnandi(a, b)  (simdia_vnoti(simdia_vandi((a), (b))))
  #define simdia_vnandf(a, b)  (simdia_vnotf(simdia_vandf((a), (b))))
  #ifdef _ARCH_PWR7 
    #define simdia_vnandlf(a, b)  (simdia_vnotf(simdia_vandf((a), (b))))
  #else
    #define simdia_vnandlf __simdia_vnandlf
  #endif

  /***** Xor *****/
  #define simdia_vxori(a, b)  (vec_xor((a), (b)))
  #define simdia_vxorf(a, b)  (vec_xor((a), (b)))
  #ifdef _ARCH_PWR7 
    #define simdia_vxorlf(a, b)  (vec_xor((a), (b)))
  #else
    #define simdia_vxorlf __simdia_vxorlf
  #endif

  /***** Nxor *****/
  #define simdia_vnxori(a, b)  (simdia_vnoti(simdia_vxori((a), (b))))
  #define simdia_vnxorf(a, b)  (simdia_vnotf(simdia_vxorf((a), (b))))
  #ifdef _ARCH_PWR7 
    #define simdia_vnxorlf(a, b)  (simdia_vnotlf(simdia_vxorf((a), (b))))
  #else
    #define simdia_vnxorlf __simdia_vnxorlf
  #endif

  /***** Equal To *****/
  #define  simdia_vcmpeqi(a, b)  ((simdia_veci)(vec_cmpeq((a), (b))))
  #define  simdia_vcmpeqf(a, b)  ((simdia_veci)(vec_cmpeq((a), (b))))
  #ifdef _ARCH_PWR7 
    #define  simdia_vcmpeqlf(a, b)  ((simdia_veci)(vec_cmpeq((a), (b))))
  #else
    #define simdia_vcmpeqlf __simdia_vcmpeqlf
  #endif

  /***** Greater Than *****/
  #define  simdia_vcmpgti(a, b)  ((simdia_veci)(vec_cmpgt((a), (b))))
  #define  simdia_vcmpgtf(a, b)  ((simdia_veci)(vec_cmpgt((a), (b))))
  #ifdef _ARCH_PWR7 
    #define  simdia_vcmpgtlf(a, b)  ((simdia_veci)(vec_cmpgt((a), (b))))
  #else
    #define simdia_vcmpgtlf __simdia_vcmpgtlf
  #endif

  /***** Greater Than Or Equal To *****/
  #define  simdia_vcmpgei(a, b)  ((simdia_veci)(vec_cmpge((a), (b))))
  #define  simdia_vcmpgef(a, b)  ((simdia_veci)(vec_cmpge((a), (b))))
  #ifdef _ARCH_PWR7 
    #define  simdia_vcmpgelf(a, b)  ((simdia_veci)(vec_cmpge((a), (b))))
  #else
    #define simdia_vcmpgelf __simdia_vcmpgelf
  #endif

  /***** Less Than *****/
  #define  simdia_vcmplti(a, b)  ((simdia_veci)(vec_cmplt((a), (b))))
  #define  simdia_vcmpltf(a, b)  ((simdia_veci)(vec_cmplt((a), (b))))
  #ifdef _ARCH_PWR7 
    #define  simdia_vcmpltlf(a, b)  ((simdia_veci)(vec_cmplt((a), (b))))
  #else
    #define simdia_vcmpltlf __simdia_vcmpltlf
  #endif

  /***** Less Than Or Equal To *****/
  #define  simdia_vcmplei(a, b)  ((simdia_veci)(vec_cmple((a), (b))))
  #define  simdia_vcmplef(a, b)  ((simdia_veci)(vec_cmple((a), (b))))
  #ifdef _ARCH_PWR7 
    #define  simdia_vcmplelf(a, b)  ((simdia_veci)(vec_cmple((a), (b))))
    // NOTE: vec_cmple not listed in Calin's wiki page of builtins for
    // PWR7, but has a header definition in the compiler
  #else
    #define simdia_vcmplelf __simdia_vcmplelf
  #endif

/*******************************************************************************
 *******************************************************************************
 ***** Mapping to Generic C Implementation
 *******************************************************************************
 *******************************************************************************/
#else

  /***** Data Types *****/
  typedef   __simdia_veci   simdia_veci;
  typedef   __simdia_vecf   simdia_vecf;
  typedef  __simdia_veclf  simdia_veclf;

  /***** Insert *****/
  #define  simdia_vinserti  __simdia_vinserti
  #define  simdia_vinsertf  __simdia_vinsertf
  #define simdia_vinsertlf __simdia_vinsertlf

  /***** Extract *****/
  #define  simdia_vextracti  __simdia_vextracti
  #define  simdia_vextractf  __simdia_vextractf
  #define simdia_vextractlf __simdia_vextractlf

  /***** Set *****/
  #define  simdia_vseti  __simdia_vseti
  #define  simdia_vsetf  __simdia_vsetf
  #define simdia_vsetlf __simdia_vsetlf

  /***** Constant Zero *****/
  #define  simdia_const_vzeroi  __simdia_const_vzeroi
  #define  simdia_const_vzerof  __simdia_const_vzerof
  #define simdia_const_vzerolf __simdia_const_vzerolf

  /***** Constant One *****/
  #define  simdia_const_vonei  __simdia_const_vonei
  #define  simdia_const_vonef  __simdia_const_vonef
  #define simdia_const_vonelf __simdia_const_vonelf

  /***** Constant Two *****/
  #define  simdia_const_vtwoi  __simdia_const_vtwoi
  #define  simdia_const_vtwof  __simdia_const_vtwof
  #define simdia_const_vtwolf __simdia_const_vtwolf

  /***** Constant Negative One *****/
  #define  simdia_const_vnegonei  __simdia_const_vnegonei
  #define  simdia_const_vnegonef  __simdia_const_vnegonef
  #define simdia_const_vnegonelf __simdia_const_vnegonelf

  /***** Rotate *****/
  #define  simdia_vrothi  __simdia_vrothi
  #define  simdia_vrothf  __simdia_vrothf
  #define simdia_vrothlf __simdia_vrothlf
  #define  simdia_vrotli  __simdia_vrotli
  #define  simdia_vrotlf  __simdia_vrotlf
  #define simdia_vrotllf __simdia_vrotllf
  
  /***** Addition *****/
  #define  simdia_vaddi  __simdia_vaddi
  #define  simdia_vaddf  __simdia_vaddf
  #define simdia_vaddlf __simdia_vaddlf

  /***** Subtraction *****/
  #define  simdia_vsubi  __simdia_vsubi
  #define  simdia_vsubf  __simdia_vsubf
  #define simdia_vsublf __simdia_vsublf

  /***** Multiplication *****/
  #define  simdia_vmulf   __simdia_vmulf
  #define simdia_vmullf  __simdia_vmullf

  /***** Division *****/
  #define  simdia_vdivf   __simdia_vdivf
  #define simdia_vdivlf  __simdia_vdivlf

  /***** Fused Multiply Add *****/
  #define  simdia_vmaddf  __simdia_vmaddf
  #define simdia_vmaddlf __simdia_vmaddlf

  /***** Reciprocal *****/
  #define  simdia_vrecipf  __simdia_vrecipf
  #define simdia_vreciplf __simdia_vreciplf

  /***** Square Root *****/
  #define  simdia_vsqrtf  __simdia_vsqrtf
  #define simdia_vsqrtlf __simdia_vsqrtlf

  /***** Reciprocal Square Root *****/
  #define  simdia_vrsqrtf  __simdia_vrsqrtf
  #define simdia_vrsqrtlf __simdia_vrsqrtlf

  /***** Not *****/
  #define  simdia_vnoti  __simdia_vnoti
  #define  simdia_vnotf  __simdia_vnotf
  #define simdia_vnotlf __simdia_vnotlf

  /***** Or *****/
  #define  simdia_vori  __simdia_vori
  #define  simdia_vorf  __simdia_vorf
  #define simdia_vorlf __simdia_vorlf

  /***** Nor *****/
  #define  simdia_vnori  __simdia_vnori
  #define  simdia_vnorf  __simdia_vnorf
  #define simdia_vnorlf __simdia_vnorlf

  /***** And *****/
  #define  simdia_vandi  __simdia_vandi
  #define  simdia_vandf  __simdia_vandf
  #define simdia_vandlf __simdia_vandlf

  /***** Nand *****/
  #define  simdia_vnandi  __simdia_vnandi
  #define  simdia_vnandf  __simdia_vnandf
  #define simdia_vnandlf __simdia_vnandlf

  /***** Xor *****/
  #define  simdia_vxori  __simdia_vxori
  #define  simdia_vxorf  __simdia_vxorf
  #define simdia_vxorlf __simdia_vxorlf

  /***** Nxor *****/
  #define  simdia_vnxori  __simdia_vnxori
  #define  simdia_vnxorf  __simdia_vnxorf
  #define simdia_vnxorlf __simdia_vnxorlf

  /***** Equal To *****/
  #define  simdia_vcmpeqi  __simdia_vcmpeqi
  #define  simdia_vcmpeqf  __simdia_vcmpeqf
  #define simdia_vcmpeqlf __simdia_vcmpeqlf

  /***** Greater Than *****/
  #define  simdia_vcmpgti  __simdia_vcmpgti
  #define  simdia_vcmpgtf  __simdia_vcmpgtf
  #define simdia_vcmpgtlf __simdia_vcmpgtlf

  /***** Greater Than Or Equal To *****/
  #define  simdia_vcmpgei  __simdia_vcmpgei
  #define  simdia_vcmpgef  __simdia_vcmpgef
  #define simdia_vcmpgelf __simdia_vcmpgelf

  /***** Less Than *****/
  #define  simdia_vcmplti  __simdia_vcmplti
  #define  simdia_vcmpltf  __simdia_vcmpltf
  #define simdia_vcmpltlf __simdia_vcmpltlf

  /***** Less Than Or Equal To *****/
  #define  simdia_vcmplei  __simdia_vcmplei
  #define  simdia_vcmplef  __simdia_vcmplef
  #define simdia_vcmplelf __simdia_vcmplelf


#endif


/*******************************************************************************
 *******************************************************************************
 ***** Shared Combinations
 *******************************************************************************
 *******************************************************************************/

/* NOTE: If any architecture specific implementation can do any of these
 *   operations faster, then move them up to the architecture specific areas and
 *   make individual definitions.  This area is just meant to declare commonly
 *   use combinations so that they don't have to be repeated many times over.
 */

/***** Number of Elements per Vector Type *****/
#define  simdia_veci_numElems  (sizeof( simdia_veci)/sizeof(   int))
#define  simdia_vecf_numElems  (sizeof( simdia_vecf)/sizeof( float))
#define simdia_veclf_numElems  (sizeof(simdia_veclf)/sizeof(double))

/***** Spread (Duplicate functionality of 'Set' by another another name) *****/
#define  simdia_vspreadi(a)  ( simdia_vseti(a))
#define  simdia_vspreadf(a)  ( simdia_vsetf(a))
#define simdia_vspreadlf(a)  (simdia_vsetlf(a))

#define  simdia_visfinitef(a) (isfinite(simdia_vextractf((a),0)) && isfinite(simdia_vextractf((a),1)) && isfinite(simdia_vextractf((a),2)) && isfinite(simdia_vextractf((a),3)))
#define simdia_visfinitelf(a) (isfinite(simdia_vextractlf((a),0)) && isfinite(simdia_vextractlf((a),1)))

/***** Add to Scalar *****/
#define   simdia_vaddis(a, b)  ( simdia_vaddi((a),  simdia_vseti(b)))
#define   simdia_vaddfs(a, b)  ( simdia_vaddf((a),  simdia_vsetf(b)))
#define  simdia_vaddlfs(a, b)  (simdia_vaddlf((a), simdia_vsetlf(b)))

/***** Subtract a Scalar *****/
#define   simdia_vsubis(a, b)  ( simdia_vsubi((a),  simdia_vseti(b)))
#define   simdia_vsubfs(a, b)  ( simdia_vsubf((a),  simdia_vsetf(b)))
#define  simdia_vsublfs(a, b)  (simdia_vsublf((a), simdia_vsetlf(b)))

/***** Multiply by Scalar *****/
#define   simdia_vmulfs(a, b)  ( simdia_vmulf((a),  simdia_vsetf(b)))
#define  simdia_vmullfs(a, b)  (simdia_vmullf((a), simdia_vsetlf(b)))

/***** Divide by Scalar *****/
#define  simdia_vdivfs(a, b)  ( simdia_vdivf((a),  simdia_vsetf(b)))
#define simdia_vdivlfs(a, b)  (simdia_vdivlf((a), simdia_vsetlf(b)))

/***** Fused Multiply(Vector) Add(Scalar) *****/
#define  simdia_vmaddfs(a, b, c)  ( simdia_vmaddf((a), (b),  simdia_vsetf(c)))
#define simdia_vmaddlfs(a, b, c)  (simdia_vmaddlf((a), (b), simdia_vsetlf(c)))

/***** Fused Multiply(Scalar) Add(Scalar) *****/
#define  simdia_vmaddfss(a, b, c)  ( simdia_vmaddf((a),  simdia_vsetf(b),  simdia_vsetf(c)))
#define simdia_vmaddlfss(a, b, c)  (simdia_vmaddlf((a), simdia_vsetlf(b), simdia_vsetlf(c)))

#if defined(__VEC__)
  #ifdef vector
  #undef vector
  #endif
#endif

#endif //__SIMDIA_H__
