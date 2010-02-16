#ifndef __SIMD_H__
#define __SIMD_H__


#if defined(__SSE2__)
  #include "emmintrin.h"
#endif

#if CMK_CELL_SPE != 0
  #include "spu_intrinsics.h"
#else
  #include "math.h"
#endif

#if defined(__VEC__)
  #include "altivec.h"
#endif


/* Solaris does not support sqrtf (float), so just map it to sqrt (double) instead */
#if !CMK_HAS_SQRTF
  #define sqrtf(a) ((float)(sqrt((double)(a))))
#endif


/* Flags to force architecture specific SIMD instructions off */
#define FORCE_NO_SSE       (0)
#define FORCE_NO_ALTIVEC   (0)
#define FORCE_NO_SPE_SIMD  (0)


/***** Math Constants *****/
#define __SIMD__CONSTANT_PI      (3.141592653589793)
#define __SIMD__CONSTANT_E       (2.718281828459045)
#define __SIMD__CONSTANT_SQRT_2  (1.414213562373095)
/* TODO | FIXME - Added intrinsics below for loading/creating vectors with these values */


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
/*
 * typedef struct __vec_16_c  {           char v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15; } __vec16c;
 * typedef struct __vec_16_uc {  unsigned char v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15; } __vec16uc;
 * typedef struct __vec_8_s  {          short v0, v1, v2, v3, v4, v5, v6, v7; } __vec8s;
 * typedef struct __vec_8_us { unsigned short v0, v1, v2, v3, v4, v5, v6, v7; } __vec8us;
 * typedef struct __vec_4_ui {   unsigned int v0, v1, v2, v3; } __vec4ui;
 */


typedef struct __vec_i  {    int v0, v1, v2, v3; }  __veci;
typedef struct __vec_f  {  float v0, v1, v2, v3; }  __vecf;
typedef struct __vec_lf { double v0, v1;         } __veclf;


/***** Insert *****/
inline  __veci  __vinserti( __veci v, const    int s, const int i) {  __veci r = v;    int* rPtr = (   int*)(&r); rPtr[i] = s; return r; }
inline  __vecf  __vinsertf( __vecf v, const  float s, const int i) {  __vecf r = v;  float* rPtr = ( float*)(&r); rPtr[i] = s; return r; }
inline __veclf __vinsertlf(__veclf v, const double s, const int i) { __veclf r = v; double* rPtr = (double*)(&r); rPtr[i] = s; return r; }

/***** Extract *****/
inline    int  __vextracti( __veci v, const int i) {    int* vPtr = (   int*)(&v); return vPtr[i]; }
inline  float  __vextractf( __vecf v, const int i) {  float* vPtr = ( float*)(&v); return vPtr[i]; }
inline double __vextractlf(__veclf v, const int i) { double* vPtr = (double*)(&v); return vPtr[i]; }

/***** Set *****/
inline  __veci  __vseti(const    int a) {  __veci r; r.v0 = r.v1 = r.v2 = r.v3 = a; return r; }
inline  __vecf  __vsetf(const  float a) {  __vecf r; r.v0 = r.v1 = r.v2 = r.v3 = a; return r; }
inline __veclf __vsetlf(const double a) { __veclf r; r.v0 = r.v1 =               a; return r; }

/* NOTE: Would it be better to generate the constants instead of read them from memory in the generic version? */

/***** Constant Zero *****/
const  __veci  __const_vzeroi = {   0 ,   0 ,   0 ,   0  };
const  __vecf  __const_vzerof = { 0.0f, 0.0f, 0.0f, 0.0f };
const __veclf __const_vzerolf = { 0.0 , 0.0              };

/***** Constant One *****/
const  __veci  __const_vonei = {   1 ,   1 ,   1 ,   1  };
const  __vecf  __const_vonef = { 1.0f, 1.0f, 1.0f, 1.0f };
const __veclf __const_vonelf = { 1.0 , 1.0              };

/***** Constant Two *****/
const  __veci  __const_vtwoi = {   2 ,   2 ,   2 ,   2  };
const  __vecf  __const_vtwof = { 2.0f, 2.0f, 2.0f, 2.0f };
const __veclf __const_vtwolf = { 2.0 , 2.0              };

/***** Constant Negative One *****/
const  __veci  __const_vnegonei = {   -1 ,   -1 ,   -1 ,   -1  };
const  __vecf  __const_vnegonef = { -1.0f, -1.0f, -1.0f, -1.0f };
const __veclf __const_vnegonelf = { -1.0 , -1.0                };

/* TODO | FIXME - Try to create constants such that it does not require a
 * memory operations to access the constants (like the SSE constants).
 */

/***** Rotate *****/
inline  __veci  __vrothi(const  __veci a, int s) {  __veci b;    int* a_ptr = (   int*)(&a);    int* b_ptr = (   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
inline  __vecf  __vrothf(const  __vecf a, int s) {  __vecf b;  float* a_ptr = ( float*)(&a);  float* b_ptr = ( float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
inline __veclf __vrothlf(const __veclf a, int s) { __veclf b; double* a_ptr = (double*)(&a); double* b_ptr = (double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0-s)&0x1]; b_ptr[1] = a_ptr[(1-s)&0x1]; return b; }
inline  __veci  __vrotli(const  __veci a, int s) {  __veci b;    int* a_ptr = (   int*)(&a);    int* b_ptr = (   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
inline  __vecf  __vrotlf(const  __vecf a, int s) {  __vecf b;  float* a_ptr = ( float*)(&a);  float* b_ptr = ( float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
inline __veclf __vrotllf(const __veclf a, int s) { __veclf b; double* a_ptr = (double*)(&a); double* b_ptr = (double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0+s)&0x1]; b_ptr[1] = a_ptr[(1+s)&0x1]; return b; }

/***** Addition *****/
inline  __veci  __vaddi(const  __veci a, const  __veci b) {  __veci r; r.v0 = a.v0 + b.v0; r.v1 = a.v1 + b.v1; r.v2 = a.v2 + b.v2; r.v3 = a.v3 + b.v3; return r; }
inline  __vecf  __vaddf(const  __vecf a, const  __vecf b) {  __vecf r; r.v0 = a.v0 + b.v0; r.v1 = a.v1 + b.v1; r.v2 = a.v2 + b.v2; r.v3 = a.v3 + b.v3; return r; }
inline __veclf __vaddlf(const __veclf a, const __veclf b) { __veclf r; r.v0 = a.v0 + b.v0; r.v1 = a.v1 + b.v1;                                         return r; }

/***** Subtraction *****/
inline  __veci  __vsubi(const  __veci a, const  __veci b) {  __veci r; r.v0 = a.v0 - b.v0; r.v1 = a.v1 - b.v1; r.v2 = a.v2 - b.v2; r.v3 = a.v3 - b.v3; return r; }
inline  __vecf  __vsubf(const  __vecf a, const  __vecf b) {  __vecf r; r.v0 = a.v0 - b.v0; r.v1 = a.v1 - b.v1; r.v2 = a.v2 - b.v2; r.v3 = a.v3 - b.v3; return r; }
inline __veclf __vsublf(const __veclf a, const __veclf b) { __veclf r; r.v0 = a.v0 - b.v0; r.v1 = a.v1 - b.v1;                                         return r; }

/***** Multiplication *****/
inline  __veci  __vmuli(const  __veci a, const  __veci b) {  __veci r; r.v0 = a.v0 * b.v0; r.v1 = a.v1 * b.v1; r.v2 = a.v2 * b.v2; r.v3 = a.v3 * b.v3; return r; }
inline  __vecf  __vmulf(const  __vecf a, const  __vecf b) {  __vecf r; r.v0 = a.v0 * b.v0; r.v1 = a.v1 * b.v1; r.v2 = a.v2 * b.v2; r.v3 = a.v3 * b.v3; return r; }
inline __veclf __vmullf(const __veclf a, const __veclf b) { __veclf r; r.v0 = a.v0 * b.v0; r.v1 = a.v1 * b.v1;                                         return r; }

/***** Division *****/
inline  __veci  __vdivi(const  __veci a, const  __veci b) {  __veci r; r.v0 = a.v0 / b.v0; r.v1 = a.v1 / b.v1; r.v2 = a.v2 / b.v2; r.v3 = a.v3 / b.v3; return r; }
inline  __vecf  __vdivf(const  __vecf a, const  __vecf b) {  __vecf r; r.v0 = a.v0 / b.v0; r.v1 = a.v1 / b.v1; r.v2 = a.v2 / b.v2; r.v3 = a.v3 / b.v3; return r; }
inline __veclf __vdivlf(const __veclf a, const __veclf b) { __veclf r; r.v0 = a.v0 / b.v0; r.v1 = a.v1 / b.v1;                                         return r; }

/***** Fused Multiply Add *****/
inline  __veci  __vmaddi(const  __veci a, const  __veci b, const  __veci c) {  __veci r; r.v0 = a.v0 * b.v0 + c.v0; r.v1 = a.v1 * b.v1 + c.v1; r.v2 = a.v2 * b.v2 + c.v2; r.v3 = a.v3 * b.v3 + c.v3; return r; }
inline  __vecf  __vmaddf(const  __vecf a, const  __vecf b, const  __vecf c) {  __vecf r; r.v0 = a.v0 * b.v0 + c.v0; r.v1 = a.v1 * b.v1 + c.v1; r.v2 = a.v2 * b.v2 + c.v2; r.v3 = a.v3 * b.v3 + c.v3; return r; }
inline __veclf __vmaddlf(const __veclf a, const __veclf b, const __veclf c) { __veclf r; r.v0 = a.v0 * b.v0 + c.v0; r.v1 = a.v1 * b.v1 + c.v1;                                                       return r; }

/***** Reciprocal *****/
/* TODO | FIXME  - See if there is a better way to do this (few cycles and avoid the memory load) */
inline  __vecf  __vrecipf(const  __vecf a) {  __vecf r; r.v0 = 1.0f / a.v0; r.v1 = 1.0f / a.v1; r.v2 = 1.0f / a.v2; r.v3 = 1.0f / a.v3; return r; }
inline __veclf __vreciplf(const __veclf a) { __veclf r; r.v0 = 1.0f / a.v0; r.v1 = 1.0f / a.v1; return r; }

/***** Square Root *****/
inline  __vecf  __vsqrtf(const  __vecf a) {  __vecf r; r.v0 = sqrtf(a.v0); r.v1 = sqrtf(a.v1); r.v2 = sqrtf(a.v2); r.v3 = sqrtf(a.v3); return r; }
inline __veclf __vsqrtlf(const __veclf a) { __veclf r; r.v0 = sqrt(a.v0); r.v1 = sqrt(a.v1); return r; }

/***** Reciprocal Square Root *****/
inline  __vecf  __vrsqrtf(const  __vecf a) {  __vecf r; r.v0 = 1.0f / sqrtf(a.v0); r.v1 = 1.0f / sqrtf(a.v1); r.v2 = 1.0f / sqrtf(a.v2); r.v3 = 1.0f / sqrtf(a.v3); return r; }
inline __veclf __vrsqrtlf(const __veclf a) { __veclf r; r.v0 = 1.0 / sqrt(a.v0); r.v1 = 1.0 / sqrt(a.v1); return r; }

/***** Not *****/
inline  __veci  __vnoti(const  __veci a) {  __veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); rPtr[0] = aPtr[0] ^ -1; rPtr[1] = aPtr[1] ^ -1; rPtr[2] = aPtr[2] ^ -1; rPtr[3] = aPtr[3] ^ -1; return r; }
inline  __vecf  __vnotf(const  __vecf a) {  __vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); rPtr[0] = aPtr[0] ^ -1; rPtr[1] = aPtr[1] ^ -1; rPtr[2] = aPtr[2] ^ -1; rPtr[3] = aPtr[3] ^ -1; return r; }
inline __veclf __vnotlf(const __veclf a) { __veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); rPtr[0] = aPtr[0] ^ -1; rPtr[1] = aPtr[1] ^ -1; rPtr[2] = aPtr[2] ^ -1; rPtr[3] = aPtr[3] ^ -1; return r; }

/***** Or *****/
inline  __veci  __vori(const  __veci a, const  __veci b) {  __veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] | bPtr[0]; rPtr[1] = aPtr[1] | bPtr[1]; rPtr[2] = aPtr[2] | bPtr[2]; rPtr[3] = aPtr[3] | bPtr[3]; return r; }
inline  __vecf  __vorf(const  __vecf a, const  __vecf b) {  __vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] | bPtr[0]; rPtr[1] = aPtr[1] | bPtr[1]; rPtr[2] = aPtr[2] | bPtr[2]; rPtr[3] = aPtr[3] | bPtr[3]; return r; }
inline __veclf __vorlf(const __veclf a, const __veclf b) { __veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] | bPtr[0]; rPtr[1] = aPtr[1] | bPtr[1]; rPtr[2] = aPtr[2] | bPtr[2]; rPtr[3] = aPtr[3] | bPtr[3]; return r; }

/***** Nor *****/
inline  __veci  __vnori(const  __veci a, const  __veci b) {  __veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] | bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] | bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] | bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] | bPtr[3]) ^ -1; return r; }
inline  __vecf  __vnorf(const  __vecf a, const  __vecf b) {  __vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] | bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] | bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] | bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] | bPtr[3]) ^ -1; return r; }
inline __veclf __vnorlf(const __veclf a, const __veclf b) { __veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] | bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] | bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] | bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] | bPtr[3]) ^ -1; return r; }

/***** And *****/
inline  __veci  __vandi(const  __veci a, const  __veci b) {  __veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] & bPtr[0]; rPtr[1] = aPtr[1] & bPtr[1]; rPtr[2] = aPtr[2] & bPtr[2]; rPtr[3] = aPtr[3] & bPtr[3]; return r; }
inline  __vecf  __vandf(const  __vecf a, const  __vecf b) {  __vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] & bPtr[0]; rPtr[1] = aPtr[1] & bPtr[1]; rPtr[2] = aPtr[2] & bPtr[2]; rPtr[3] = aPtr[3] & bPtr[3]; return r; }
inline __veclf __vandlf(const __veclf a, const __veclf b) { __veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] & bPtr[0]; rPtr[1] = aPtr[1] & bPtr[1]; rPtr[2] = aPtr[2] & bPtr[2]; rPtr[3] = aPtr[3] & bPtr[3]; return r; }

/***** Nand *****/
inline  __veci  __vnandi(const  __veci a, const  __veci b) {  __veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] & bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] & bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] & bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] & bPtr[3]) ^ -1; return r; }
inline  __vecf  __vnandf(const  __vecf a, const  __vecf b) {  __vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] & bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] & bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] & bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] & bPtr[3]) ^ -1; return r; }
inline __veclf __vnandlf(const __veclf a, const __veclf b) { __veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] & bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] & bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] & bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] & bPtr[3]) ^ -1; return r; }

/***** Xor *****/
inline  __veci  __vxori(const  __veci a, const  __veci b) {  __veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] ^ bPtr[0]; rPtr[1] = aPtr[1] ^ bPtr[1]; rPtr[2] = aPtr[2] ^ bPtr[2]; rPtr[3] = aPtr[3] ^ bPtr[3]; return r; }
inline  __vecf  __vxorf(const  __vecf a, const  __vecf b) {  __vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] ^ bPtr[0]; rPtr[1] = aPtr[1] ^ bPtr[1]; rPtr[2] = aPtr[2] ^ bPtr[2]; rPtr[3] = aPtr[3] ^ bPtr[3]; return r; }
inline __veclf __vxorlf(const __veclf a, const __veclf b) { __veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = aPtr[0] ^ bPtr[0]; rPtr[1] = aPtr[1] ^ bPtr[1]; rPtr[2] = aPtr[2] ^ bPtr[2]; rPtr[3] = aPtr[3] ^ bPtr[3]; return r; }

/***** Nxor *****/
inline  __veci  __vnxori(const  __veci a, const  __veci b) {  __veci r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] ^ bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] ^ bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] ^ bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] ^ bPtr[3]) ^ -1; return r; }
inline  __vecf  __vnxorf(const  __vecf a, const  __vecf b) {  __vecf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] ^ bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] ^ bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] ^ bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] ^ bPtr[3]) ^ -1; return r; }
inline __veclf __vnxorlf(const __veclf a, const __veclf b) { __veclf r; int* rPtr = (int*)(&r); int* aPtr = (int*)(&a); int* bPtr = (int*)(&b); rPtr[0] = (aPtr[0] ^ bPtr[0]) ^ -1; rPtr[1] = (aPtr[1] ^ bPtr[1]) ^ -1; rPtr[2] = (aPtr[2] ^ bPtr[2]) ^ -1; rPtr[3] = (aPtr[3] ^ bPtr[3]) ^ -1; return r; }

/* TODO | FIXME - Try to do the comparisons in a branchless way */

/***** Equal To *****/
inline __veci  __vcmpeqi(const  __veci a, const  __veci b) {  __veci r; r.v0 = ((a.v0 == b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 == b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 == b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 == b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci  __vcmpeqf(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = ((a.v0 == b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 == b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 == b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 == b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci __vcmpeqlf(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = r.v1 = ((a.v0 == b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 == b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }

/***** Greater Than *****/
inline __veci  __vcmpgti(const  __veci a, const  __veci b) {  __veci r; r.v0 = ((a.v0 > b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 > b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 > b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 > b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci  __vcmpgtf(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = ((a.v0 > b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 > b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 > b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 > b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci __vcmpgtlf(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = r.v1 = ((a.v0 > b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 > b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }

/***** Greater Than Or Equal To *****/
inline __veci  __vcmpgei(const  __veci a, const  __veci b) {  __veci r; r.v0 = ((a.v0 >= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 >= b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 >= b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 >= b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci  __vcmpgef(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = ((a.v0 >= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 >= b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 >= b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 >= b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci __vcmpgelf(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = r.v1 = ((a.v0 >= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 >= b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }

/***** Less Than *****/
inline __veci  __vcmplti(const  __veci a, const  __veci b) {  __veci r; r.v0 = ((a.v0 < b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 < b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 < b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 < b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci  __vcmpltf(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = ((a.v0 < b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 < b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 < b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 < b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci __vcmpltlf(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = r.v1 = ((a.v0 < b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 < b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }

/***** Less Than Or Equal To *****/
inline __veci  __vcmplei(const  __veci a, const  __veci b) {  __veci r; r.v0 = ((a.v0 <= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 <= b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 <= b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 <= b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci  __vcmplef(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = ((a.v0 <= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v1 = ((a.v1 <= b.v1) ? (0xFFFFFFFF) : (0x0)); r.v2 = ((a.v2 <= b.v2) ? (0xFFFFFFFF) : (0x0)); r.v3 = ((a.v3 <= b.v3) ? (0xFFFFFFFF) : (0x0)); return r; }
inline __veci __vcmplelf(const  __vecf a, const  __vecf b) {  __veci r; r.v0 = r.v1 = ((a.v0 <= b.v0) ? (0xFFFFFFFF) : (0x0)); r.v2 = r.v3 = ((a.v1 <= b.v1) ? (0xFFFFFFFF) : (0x0)); return r; }


/*******************************************************************************
 ***** C++ Operators for Generic Implementation
 *******************************************************************************/
#if defined(__cplusplus)

  /***** Addition *****/
  inline  __veci operator+(const  __veci &a, const  __veci &b) { return  __vaddi(a, b); }
  inline  __vecf operator+(const  __vecf &a, const  __vecf &b) { return  __vaddf(a, b); }
  inline __veclf operator+(const __veclf &a, const __veclf &b) { return __vaddlf(a, b); }
  inline  __veci operator+=( __veci &a, const  __veci &b) { a =  __vaddi(a, b); return a; }
  inline  __vecf operator+=( __vecf &a, const  __vecf &b) { a =  __vaddf(a, b); return a; }
  inline __veclf operator+=(__veclf &a, const __veclf &b) { a = __vaddlf(a, b); return a; }

  inline  __veci operator+(const  __veci &a, const    int &b) { return  __vaddi(a,  __vseti(b)); }
  inline  __vecf operator+(const  __vecf &a, const  float &b) { return  __vaddf(a,  __vsetf(b)); }
  inline __veclf operator+(const __veclf &a, const double &b) { return __vaddlf(a, __vsetlf(b)); }
  inline  __veci operator+=( __veci &a, const    int &b) { a =  __vaddi(a,  __vseti(b)); return a; }
  inline  __vecf operator+=( __vecf &a, const  float &b) { a =  __vaddf(a,  __vsetf(b)); return a; }
  inline __veclf operator+=(__veclf &a, const double &b) { a = __vaddlf(a, __vsetlf(b)); return a; }

  /***** Subtraction *****/
  inline  __veci operator-(const  __veci &a, const  __veci &b) { return  __vsubi(a, b); }
  inline  __vecf operator-(const  __vecf &a, const  __vecf &b) { return  __vsubf(a, b); }
  inline __veclf operator-(const __veclf &a, const __veclf &b) { return __vsublf(a, b); }
  inline  __veci operator-=( __veci &a, const  __veci &b) { a =  __vsubi(a, b); return a; }
  inline  __vecf operator-=( __vecf &a, const  __vecf &b) { a =  __vsubf(a, b); return a; }
  inline __veclf operator-=(__veclf &a, const __veclf &b) { a = __vsublf(a, b); return a; }

  inline  __veci operator-(const  __veci &a, const    int &b) { return  __vsubi(a,  __vseti(b)); }
  inline  __vecf operator-(const  __vecf &a, const  float &b) { return  __vsubf(a,  __vsetf(b)); }
  inline __veclf operator-(const __veclf &a, const double &b) { return __vsublf(a, __vsetlf(b)); }
  inline  __veci operator-=( __veci &a, const    int &b) { a =  __vsubi(a,  __vseti(b)); return a; }
  inline  __vecf operator-=( __vecf &a, const  float &b) { a =  __vsubf(a,  __vsetf(b)); return a; }
  inline __veclf operator-=(__veclf &a, const double &b) { a = __vsublf(a, __vsetlf(b)); return a; }

  /***** Multiplication *****/
  inline  __vecf operator*(const  __vecf &a, const  __vecf &b) { return  __vmulf(a, b); }
  inline __veclf operator*(const __veclf &a, const __veclf &b) { return __vmullf(a, b); }
  inline  __vecf operator*=( __vecf &a, const  __vecf &b) { a =  __vmulf(a, b); return a; }
  inline __veclf operator*=(__veclf &a, const __veclf &b) { a = __vmullf(a, b); return a; }

  inline  __vecf operator*(const  __vecf &a, const  float &b) { return  __vmulf(a,  __vsetf(b)); }
  inline __veclf operator*(const __veclf &a, const double &b) { return __vmullf(a, __vsetlf(b)); }
  inline  __vecf operator*=( __vecf &a, const  float &b) { a =  __vmulf(a,  __vsetf(b)); return a; }
  inline __veclf operator*=(__veclf &a, const double &b) { a = __vmullf(a, __vsetlf(b)); return a; }

  /***** Division *****/
  inline  __vecf operator/(const  __vecf &a, const  __vecf &b) { return  __vdivf(a, b); }
  inline __veclf operator/(const __veclf &a, const __veclf &b) { return __vdivlf(a, b); }
  inline  __vecf operator/=( __vecf &a, const  __vecf &b) { a =  __vdivf(a, b); return a; }
  inline __veclf operator/=(__veclf &a, const __veclf &b) { a = __vdivlf(a, b); return a; }

  inline  __vecf operator/(const  __vecf &a, const  float &b) { return  __vdivf(a,  __vsetf(b)); }
  inline __veclf operator/(const __veclf &a, const double &b) { return __vdivlf(a, __vsetlf(b)); }
  inline  __vecf operator/=( __vecf &a, const  float &b) { a =  __vdivf(a,  __vsetf(b)); return a; }
  inline __veclf operator/=(__veclf &a, const double &b) { a = __vdivlf(a, __vsetlf(b)); return a; }

  /***** Or *****/
  inline  __veci operator|(const  __veci &a, const  __veci &b) { return  __vori(a, b); }
  inline  __vecf operator|(const  __vecf &a, const  __vecf &b) { return  __vorf(a, b); }
  inline __veclf operator|(const __veclf &a, const __veclf &b) { return __vorlf(a, b); }
  inline  __veci operator|=( __veci &a, const  __veci &b) { a =  __vori(a, b); return a; }
  inline  __vecf operator|=( __vecf &a, const  __vecf &b) { a =  __vorf(a, b); return a; }
  inline __veclf operator|=(__veclf &a, const __veclf &b) { a = __vorlf(a, b); return a; }

  inline  __veci operator|(const  __veci &a, const    int &b) { return  __vori(a,  __vseti(b)); }
  inline  __vecf operator|(const  __vecf &a, const  float &b) { return  __vorf(a,  __vsetf(b)); }
  inline __veclf operator|(const __veclf &a, const double &b) { return __vorlf(a, __vsetlf(b)); }
  inline  __veci operator|=( __veci &a, const    int &b) { a =  __vori(a,  __vseti(b)); return a; }
  inline  __vecf operator|=( __vecf &a, const  float &b) { a =  __vorf(a,  __vsetf(b)); return a; }
  inline __veclf operator|=(__veclf &a, const double &b) { a = __vorlf(a, __vsetlf(b)); return a; }

  /***** And *****/
  inline  __veci operator&(const  __veci &a, const  __veci &b) { return  __vandi(a, b); }
  inline  __vecf operator&(const  __vecf &a, const  __vecf &b) { return  __vandf(a, b); }
  inline __veclf operator&(const __veclf &a, const __veclf &b) { return __vandlf(a, b); }
  inline  __veci operator&=( __veci &a, const  __veci &b) { a =  __vandi(a, b); return a; }
  inline  __vecf operator&=( __vecf &a, const  __vecf &b) { a =  __vandf(a, b); return a; }
  inline __veclf operator&=(__veclf &a, const __veclf &b) { a = __vandlf(a, b); return a; }

  inline  __veci operator&(const  __veci &a, const    int &b) { return  __vandi(a,  __vseti(b)); }
  inline  __vecf operator&(const  __vecf &a, const  float &b) { return  __vandf(a,  __vsetf(b)); }
  inline __veclf operator&(const __veclf &a, const double &b) { return __vandlf(a, __vsetlf(b)); }
  inline  __veci operator&=( __veci &a, const    int &b) { a =  __vandi(a,  __vseti(b)); return a; }
  inline  __vecf operator&=( __vecf &a, const  float &b) { a =  __vandf(a,  __vsetf(b)); return a; }
  inline __veclf operator&=(__veclf &a, const double &b) { a = __vandlf(a, __vsetlf(b)); return a; }

  /***** Xor *****/
  inline  __veci operator^(const  __veci &a, const  __veci &b) { return  __vxori(a, b); }
  inline  __vecf operator^(const  __vecf &a, const  __vecf &b) { return  __vxorf(a, b); }
  inline __veclf operator^(const __veclf &a, const __veclf &b) { return __vxorlf(a, b); }
  inline  __veci operator^=( __veci &a, const  __veci &b) { a =  __vxori(a, b); return a; }
  inline  __vecf operator^=( __vecf &a, const  __vecf &b) { a =  __vxorf(a, b); return a; }
  inline __veclf operator^=(__veclf &a, const __veclf &b) { a = __vxorlf(a, b); return a; }

  inline  __veci operator^(const  __veci &a, const    int &b) { return  __vxori(a,  __vseti(b)); }
  inline  __vecf operator^(const  __vecf &a, const  float &b) { return  __vxorf(a,  __vsetf(b)); }
  inline __veclf operator^(const __veclf &a, const double &b) { return __vxorlf(a, __vsetlf(b)); }
  inline  __veci operator^=( __veci &a, const    int &b) { a =  __vxori(a,  __vseti(b)); return a; }
  inline  __vecf operator^=( __vecf &a, const  float &b) { a =  __vxorf(a,  __vsetf(b)); return a; }
  inline __veclf operator^=(__veclf &a, const double &b) { a = __vxorlf(a, __vsetlf(b)); return a; }

#endif /* defined(__cplusplus) */

/*@}*/


/*******************************************************************************
 *******************************************************************************
 ***** SSE Support
 *******************************************************************************
 *******************************************************************************/
#if defined(__SSE2__) && (!(FORCE_NO_SSE))

  /* NOTE | TODO | FIXME : Add checks for various version of SSE.  For now, only
   *   support and assume that minimum level SSE2.
   */

  /***** Data Types *****/
  typedef __m128i  veci;
  typedef  __m128  vecf;
  typedef __m128d veclf;

  /***** Insert *****/
  /* TODO | FIXME - Try to make these functions not reference memory so values stay in registers */
  inline  veci  vinserti( veci v, const    int s, const int i) {  veci r = v;    int* rPtr = (   int*)(&r); rPtr[i] = s; return r; }
  inline  vecf  vinsertf( vecf v, const  float s, const int i) {  vecf r = v;  float* rPtr = ( float*)(&r); rPtr[i] = s; return r; }
  inline veclf vinsertlf(veclf v, const double s, const int i) { veclf r = v; double* rPtr = (double*)(&r); rPtr[i] = s; return r; }

  /***** Extract *****/
  /* TODO | FIXME - Try to make these functions not reference memory so values stay in registers */
  inline    int  vextracti( veci v, const int i) { return ((   int*)(&v))[i]; }
  inline  float  vextractf( vecf v, const int i) { return (( float*)(&v))[i]; }
  inline double vextractlf(veclf v, const int i) { return ((double*)(&v))[i]; }


  /***** Set *****/
  #define  vseti(a)  (_mm_set1_epi32((int)(a)))
  #define  vsetf(a)  (_mm_set1_ps((float)(a)))
  #define vsetlf(a)  (_mm_set1_pd((double)(a)))

  /***** Constant Zero *****/
  #define  const_vzeroi  (_mm_setzero_si128())
  #define  const_vzerof  (_mm_setzero_ps())
  #define const_vzerolf  (_mm_setzero_pd())

  /***** Constant One *****/
  #define  const_vonei  (vseti(1))
  #define  const_vonef  (vsetf(1.0f))
  #define const_vonelf  (vsetlf(1.0))

  /***** Constant Two *****/
  #define  const_vtwoi  (vseti(2))
  #define  const_vtwof  (vsetf(2.0f))
  #define const_vtwolf  (vsetlf(2.0))

  /***** Constant Negative One *****/
  #define  const_vnegonei  (vseti(-1))
  #define  const_vnegonef  (vsetf(-1.0f))
  #define const_vnegonelf  (vsetlf(-1.0))

  /***** Rotate *****/
  /* TODO : FIXME - Find a better way to do Rotate in SSE */
  inline  veci  vrothi(const  veci &a, int s) {  veci b;    int* a_ptr = (   int*)(&a);    int* b_ptr = (   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
  inline  vecf  vrothf(const  vecf &a, int s) {  vecf b;  float* a_ptr = ( float*)(&a);  float* b_ptr = ( float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
  inline veclf vrothlf(const veclf &a, int s) { veclf b; double* a_ptr = (double*)(&a); double* b_ptr = (double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0-s)&0x1]; b_ptr[1] = a_ptr[(1-s)&0x1]; return b; }
  inline  veci  vrotli(const  veci &a, int s) {  veci b;    int* a_ptr = (   int*)(&a);    int* b_ptr = (   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
  inline  vecf  vrotlf(const  vecf &a, int s) {  vecf b;  float* a_ptr = ( float*)(&a);  float* b_ptr = ( float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
  inline veclf vrotllf(const veclf &a, int s) { veclf b; double* a_ptr = (double*)(&a); double* b_ptr = (double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0+s)&0x1]; b_ptr[1] = a_ptr[(1+s)&0x1]; return b; }

  /***** Addition *****/
  #define  vaddi(a, b)  (_mm_add_epi32((a), (b)))
  #define  vaddf(a, b)  (_mm_add_ps((a), (b)))
  #define vaddlf(a, b)  (_mm_add_pd((a), (b)))

  /***** Subtraction *****/
  #define  vsubi(a, b)  (_mm_sub_epi32((a), (b)))
  #define  vsubf(a, b)  (_mm_sub_ps((a), (b)))
  #define vsublf(a, b)  (_mm_sub_pd((a), (b)))

  /***** Multiplication *****/
  #define    vmulf(a, b)  (_mm_mul_ps((a), (b)))
  #define   vmullf(a, b)  (_mm_mul_pd((a), (b)))

  /***** Division *****/
  #define   vdivf(a, b)  (_mm_div_ps((a), (b)))
  #define  vdivlf(a, b)  (_mm_div_pd((a), (b)))

  /***** Fused Multiply Add *****/
  #define  vmaddf(a, b, c)  ( vaddf( vmulf((a), (b)), (c)))
  #define vmaddlf(a, b, c)  (vaddlf(vmullf((a), (b)), (c)))

  /***** Reciprocal *****/
  #define  vrecipf(a)  (_mm_rcp_ps(a))

// why, oh why hath the SSE2 developers forsaken us?
//  #define vreciplf(a)  (_mm_rcp_pd(a)) <-- no worky

// vrecip goes to a not very vector implementation
inline veclf vreciplf(const veclf a) { veclf r; double* a_ptr =  (double*)(&a); double* r_ptr = (double*)(&r); r_ptr[0] = 1.0f /  a_ptr[0]; r_ptr[1] = 1.0f / a_ptr[1]; return r; }

  /***** Square Root *****/
  #define  vsqrtf(a)  (_mm_sqrt_ps(a))
  #define vsqrtlf(a)  (_mm_sqrt_pd(a))

  /***** Reciprocal Square Root *****/
  #define  vrsqrtf(a)  (_mm_rsqrt_ps(a))
  #define vrsqrtlf(a)  (vreciplf(vsqrtlf(a)))

  /***** Not *****/
  #define  vnoti(a)  (_mm_xor_si128((a), const_vnegonei))
  #define  vnotf(a)  (_mm_xor_ps((a), const_vnegonei))
  #define vnotlf(a)  (_mm_xor_pd((a), const_vnegonei))

  /***** Or *****/
  #define  vori(a, b)  (_mm_or_si128((a), (b)))
  #define  vorf(a, b)  (_mm_or_ps((a), (b)))
  #define vorlf(a, b)  (_mm_or_pd((a), (b)))

  /***** Nor *****/
  #define  vnori(a, b)  ( vnoti( vori((a), (b))))
  #define  vnorf(a, b)  ( vnotf( vorf((a), (b))))
  #define vnorlf(a, b)  (vnotlf(vorlf((a), (b))))

  /***** And *****/
  #define  vandi(a, b)  (_mm_and_si128((a), (b)))
  #define  vandf(a, b)  (_mm_and_ps((a), (b)))
  #define vandlf(a, b)  (_mm_and_pd((a), (b)))

  /***** Nand *****/
  #define  vnandi(a, b)  ( vnoti( vandi((a), (b))))
  #define  vnandf(a, b)  ( vnotf( vandf((a), (b))))
  #define vnandlf(a, b)  (vnotlf(vandlf((a), (b))))

  /***** Xor *****/
  #define  vxori(a, b)  (_mm_xor_si128((a), (b)))
  #define  vxorf(a, b)  (_mm_xor_ps((a), (b)))
  #define vxorlf(a, b)  (_mm_xor_pd((a), (b)))

  /***** Nxor *****/
  #define  vnxori(a, b)  ( vnoti( vxori((a), (b))))
  #define  vnxorf(a, b)  ( vnotf( vxorf((a), (b))))
  #define vnxorlf(a, b)  (vnotlf(vxorlf((a), (b))))

  /***** Equal To *****/
  #define  vcmpeqi(a, b)  ((_m128i)(_mm_cmpeq_epi32((a), (b))))
  #define  vcmpeqf(a, b)  ((_m128i)(_mm_cmpeq_ps((a), (b))))
  #define vcmpeqlf(a, b)  ((_m128i)(_mm_cmpeq_pd((a), (b))))

  /***** Greater Than *****/
  #define  vcmpgti(a, b)  ((_m128i)(_mm_cmpgt_epi32((a), (b))))
  #define  vcmpgtf(a, b)  ((_m128i)(_mm_cmpgt_ps((a), (b))))
  #define vcmpgtlf(a, b)  ((_m128i)(_mm_cmpgt_pd((a), (b))))

  /***** Greater Than Or Equal To *****/
  #define  vcmpgei(a, b)  ((_m128i)(_mm_cmpge_epi32((a), (b))))
  #define  vcmpgef(a, b)  ((_m128i)(_mm_cmpge_ps((a), (b))))
  #define vcmpgelf(a, b)  ((_m128i)(_mm_cmpge_pd((a), (b))))

  /***** Less Than *****/
  #define  vcmplti(a, b)  ((_m128i)(_mm_cmplt_epi32((a), (b))))
  #define  vcmpltf(a, b)  ((_m128i)(_mm_cmplt_ps((a), (b))))
  #define vcmpltlf(a, b)  ((_m128i)(_mm_cmplt_pd((a), (b))))

  /***** Less Than Or Equal To *****/
  #define  vcmplei(a, b)  ((_m128i)(_mm_cmple_epi32((a), (b))))
  #define  vcmplef(a, b)  ((_m128i)(_mm_cmple_ps((a), (b))))
  #define vcmplelf(a, b)  ((_m128i)(_mm_cmple_pd((a), (b))))


/*******************************************************************************
 *******************************************************************************
 ***** SPE SIMD Instructions
 *******************************************************************************
 *******************************************************************************/
/* TODO | FIXME : Find a more general check for this (this is Charm++ specific) */
#elif (CMK_CELL_SPE != 0) && (!(FORCE_NO_SPE_SIMD))

  /***** Data Types *****/
  typedef vector signed int veci;
  typedef vector float vecf;
  typedef vector double veclf;

  /***** Insert *****/
  #define  vinserti(v, s, i)  (spu_insert((s), (v), (i)))
  #define  vinsertf(v, s, i)  (spu_insert((s), (v), (i)))
  #define vinsertlf(v, s, i)  (spu_insert((s), (v), (i)))

  /***** Extract *****/
  #define  vextracti(v, i)  (spu_extract((v), (i)))
  #define  vextractf(v, i)  (spu_extract((v), (i)))
  #define vextractlf(v, i)  (spu_extract((v), (i)))

  /***** Set *****/
  #define  vseti(a)  (spu_splats((int)(a)))
  #define  vsetf(a)  (spu_splats((float)(a)))
  #define vsetlf(a)  (spu_splats((double)(a)))

  /***** Constant Zero *****/
  #define  const_vzeroi  (vseti(0))
  #define  const_vzerof  (vsetf(0.0f))
  #define const_vzerolf  (vsetlf(0.0))

  /***** Constant One *****/
  #define  const_vonei  (vseti(1))
  #define  const_vonef  (vsetf(1.0f))
  #define const_vonelf  (vsetlf(1.0))

  /***** Constant Two *****/
  #define  const_vtwoi  (vseti(2))
  #define  const_vtwof  (vsetf(2.0f))
  #define const_vtwolf  (vsetlf(2.0))

  /***** Constant Negative One *****/
  #define  const_vnegonei  (vseti(-1))
  #define  const_vnegonef  (vsetf(-1.0f))
  #define const_vnegonelf  (vsetlf(-1.0))

  /***** Rotate *****/
  #define   vrothi(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x3)<<2)) ))
  #define   vrothf(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x3)<<2)) ))
  #define  vrothlf(a, s) (spu_rlqwbyte((a),       (((s)&0x1)<<3)  ))
  #define   vrotli(a, s) (spu_rlqwbyte((a), ((s)&0x3)<<2))
  #define   vrotlf(a, s) (spu_rlqwbyte((a), ((s)&0x3)<<2))
  #define  vrotllf(a, s) (spu_rlqwbyte((a), ((s)&0x1)<<3))

  /***** Addition *****/
  #define  vaddi(a, b)  (spu_add((a), (b)))
  #define  vaddf(a, b)  (spu_add((a), (b)))
  #define vaddlf(a, b)  (spu_add((a), (b)))

  /***** Subtraction *****/
  #define  vsubi(a, b)  (spu_sub((a), (b)))
  #define  vsubf(a, b)  (spu_sub((a), (b)))
  #define vsublf(a, b)  (spu_sub((a), (b)))

  /***** Multiplication *****/
  #define   vmulf(a, b)  (spu_mul((a), (b)))
  #define  vmullf(a, b)  (spu_mul((a), (b)))

  /***** Division *****/
  #define vdivf(a, b)  (spu_mul((a), spu_re(b)))
  inline veclf vdivlf(const veclf a, const veclf b) { veclf r = { 0.0, 0.0 }; spu_insert((spu_extract(a, 0) / spu_extract(b, 0)), r, 0); spu_insert((spu_extract(a, 1) / spu_extract(b, 1)), r, 1); return r; }

  /***** Fused Multiply Add *****/
  #define  vmaddf(a, b, c)  (spu_madd((a), (b), (c)))
  #define vmaddlf(a, b, c)  (spu_madd((a), (b), (c)))

  /***** Reciprocal *****/
  #define  vrecipf(a)  (spu_re(a))
  inline veclf vreciplf(const veclf a, const veclf b) { veclf r = { 0.0, 0.0 }; spu_insert((1.0f / spu_extract(a, 0)), r, 0); spu_insert((1.0f / spu_extract(a, 1)), r, 1); return r; }

  /***** Square Root *****/
  #define vsqrtf(a) (spu_re(spu_rsqrte(a)))
  inline veclf vsqrtlf(const veclf a, const veclf b) { veclf r = { 0.0, 0.0 }; spu_insert(sqrt(spu_extract(a, 0)), r, 0); spu_insert(sqrt(spu_extract(a, 1)), r, 1); return r; }

  /***** Reciprocal Square Root *****/
  #define vrsqrtf(a) (spu_rsqrte(a))
  inline veclf vrsqrtlf(const veclf a, const veclf b) { veclf r = { 0.0, 0.0 }; spu_insert((1.0f / sqrt(spu_extract(a, 0))), r, 0); spu_insert((1.0f / sqrt(spu_extract(a, 1))), r, 1); return r; }

  /***** Not *****/
  #define  vnoti(a)  (spu_nor((a), const_vzeroi))
  #define  vnotf(a)  (spu_nor((a), const_vzerof))
  #define vnotlf(a)  (spu_nor((a), const_vzerolf))

  /***** Or *****/
  #define  vori(a, b)  (spu_or((a), (b)))
  #define  vorf(a, b)  (spu_or((a), (b)))
  #define vorlf(a, b)  (spu_or((a), (b)))

  /***** Nor *****/
  #define  vnori(a, b)  (spu_nor((a), (b)))
  #define  vnorf(a, b)  (spu_nor((a), (b)))
  #define vnorlf(a, b)  (spu_nor((a), (b)))

  /***** And *****/
  #define  vandi(a, b)  (spu_and((a), (b)))
  #define  vandf(a, b)  (spu_and((a), (b)))
  #define vandlf(a, b)  (spu_and((a), (b)))

  /***** Nand *****/
  #define  vnandi(a, b)  (spu_nand((a), (b)))
  #define  vnandf(a, b)  (spu_nand((a), (b)))
  #define vnandlf(a, b)  (spu_nand((a), (b)))

  /***** Xor *****/
  #define  vxori(a, b)  (spu_xor((a), (b)))
  #define  vxorf(a, b)  (spu_xor((a), (b)))
  #define vxorlf(a, b)  (spu_xor((a), (b)))

  /***** Nxor *****/
  #define  vnxori(a, b)  ( vnoti( vxori((a), (b))))
  #define  vnxorf(a, b)  ( vnotf( vxorf((a), (b))))
  #define vnxorlf(a, b)  (vnotlf(vxorlf((a), (b))))

  /***** Equal To *****/
  #define  vcmpeqi(a, b)  ((veci)(spu_cmpeq((a), (b))))
  #define  vcmpeqf(a, b)  ((veci)(spu_cmpeq((a), (b))))
  #define vcmpeqlf(a, b)  ((veci)(spu_cmpeq((a), (b))))

  /***** Greater Than *****/
  #define  vcmpgti(a, b)  ((veci)(spu_cmpgt((a), (b))))
  #define  vcmpgtf(a, b)  ((veci)(spu_cmpgt((a), (b))))
  #define vcmpgtlf(a, b)  ((veci)(spu_cmpgt((a), (b))))

  // NOTE : Try to create versions of >= and < that do not double evaluate their inputs

  /***** Greater Than or Equal To *****/
  #define  vcmpgei(a, b)  (spu_or( vcmpeqi((a), (b)),  vcmpgti((a), (b))))
  #define  vcmpgef(a, b)  (spu_or( vcmpeqf((a), (b)),  vcmpgtf((a), (b))))
  #define vcmpgelf(a, b)  (spu_or(vcmpeqlf((a), (b)), vcmpgtlf((a), (b))))

  /***** Less Than *****/
  #define  vcmplti(a, b)  (spu_nor( vcmpgti((a), (b)),  vcmpeqi((a), (b))))
  #define  vcmpltf(a, b)  (spu_nor( vcmpgtf((a), (b)),  vcmpeqf((a), (b))))
  #define vcmpltlf(a, b)  (spu_nor(vcmpgtlf((a), (b)), vcmpeqlf((a), (b))))

  /***** Less Than or Equal To *****/
  #define  vcmplei(a, b)  (spu_nor( vcmpgti((a), (b)),  const_vzeroi))
  #define  vcmplef(a, b)  (spu_nor( vcmpgtf((a), (b)),  const_vzerof))
  #define vcmplelf(a, b)  (spu_nor(vcmpgtlf((a), (b)), const_vzerolf))


/*******************************************************************************
 *******************************************************************************
 ***** AltiVec
 *******************************************************************************
 *******************************************************************************/
#elif defined(__VEC__) && (!(FORCE_NO_ALTIVEC))

  /***** Data Types *****/
  typedef vector signed int veci;
  typedef vector float vecf;
#ifdef _ARCH_PWR7
/** power 7 VSX supports 64 bit operands, it also includes VMX support
 * which means that things like vec_div, vec_insert, etcetera work for
 * ints floats and doubles.  These intrinsics also require a suitably
 * new version of the compiler on Power 7.  If you are somehow using a
 * Power 7 with an old compiler, please do not hesitate to open a can
 * of whoopass on whoever installed the tool chain, because that kind
 * of stupidity should not be tolerated.
 */
 
  typedef vector double  veclf;
#else
  typedef __veclf veclf;
#endif

  /***** Insert *****/
  /* TODO | FIXME - Try to make these functions not reference memory
     so values stay in registers */


#ifdef _ARCH_PWR7 
// swap argument order
 #define  vinserti(a, b, c)  (vec_insert((b)), ((a)), ((c)))
 #define  vinsertf(a, b, c)  (vec_insert((b)), ((a)), ((c)))
 #define  vinsertlf(a, b, c)  (vec_insert((b)), ((a)), ((c)))
#else
  inline  veci  vec_inserti( veci v, const    int s, const int i) {  veci r = v;    int* rPtr = (   int*)(&r); rPtr[i] = s; return r; }
  inline  vecf  vinsertf( vecf v, const  float s, const int i) {  vecf r = v;  float* rPtr = ( float*)(&r); rPtr[i] = s; return r; }
  inline veclf vinsertlf(veclf v, const double s, const int i) { return vec_insert(s,v,i); }

#endif
  /***** Extract *****/

#ifdef _ARCH_PWR7 
#define  vextracti(a, b)  (vec_extract((a), (b)))
#define  vextractf(a, b)  (vec_extract((a), (b)))
#define  vextractlf(a, b)  (vec_extract((a), (b)))
#else
  /* TODO | FIXME - Try to make these functions not reference memory so values stay in registers */
  inline    int  vextracti( veci v, const int i) {    int* vPtr = (   int*)(&v); return vPtr[i]; }
  inline  float  vextractf( vecf v, const int i) {  float* vPtr = ( float*)(&v); return vPtr[i]; }

  inline double vextractlf(veclf v, const int i) { double* vPtr = (double*)(&v); return vPtr[i]; }
#endif
  /***** Set *****/
#ifdef _ARCH_PWR7 
  /***** Set *****/
#define  vseti(a)  (vec_promote((a), 0))
#define  vsetf(a)  (vec_promote((a), 0))
#define  vsetlf(a)  (vec_promote((a), 0))
#else
  /* TODO : FIXME - There must be a better way to do this, but it
  seems the only way to convert scalar to vector is to go through
  memory instructions.  

  EJB: converting between scalar and vector is the sort of thing you
  want to avoid doing on altivec.  Better to rethink and find a way to
  stay in the vector engine if at all possible.

   */
  inline veci vseti(const   int a) { __veci r; r.v0 = a; return vec_splat(*((veci*)(&r)), 0); }
  inline vecf vsetf(const float a) { __vecf r; r.v0 = a; return vec_splat(*((vecf*)(&r)), 0); }
  #define vsetlf __vsetlf
#endif
  /* NOTE: Declare one for unsigned char vector also (required by rotate functions) */
  inline vector unsigned char vset16uc(const unsigned char c) { vector unsigned char r __attribute__((aligned(16))); ((unsigned char*)(&r))[0] = c; return vec_splat(r, 0); }

  /***** Constant Zero *****/
  #define  const_vzeroi  (vec_splat_s32(0))
  #define  const_vzerof  (vec_ctf(vec_splat_s32(0), 0))
#ifdef _ARCH_PWR7 
  #define const_vzerolf  (vec_splats(0))
#else
  #define const_vzerolf  (__const_vzerolf)
#endif
  /***** Constant One *****/
  #define  const_vonei  (vec_splat_s32(1))
  #define  const_vonef  (vec_ctf(vec_splat_s32(1), 0))
#ifdef _ARCH_PWR7 
  #define const_vonelf  (vec_splats(1))
#else
  #define const_vonelf  (__const_vonelf)
#endif

  /***** Constant Two *****/
  #define  const_vtwoi  (vec_splat_s32(2))
  #define  const_vtwof  (vec_ctf(vec_splat_s32(2), 0))
#ifdef _ARCH_PWR7 
  #define const_vtwolf  (vec_splats(2))
#else
  #define const_vtwolf  (__const_vtwolf)
#endif
  /***** Constant Negative One *****/
  #define  const_vnegonei  (vec_splat_s32(-1))
  #define  const_vnegonef  (vec_ctf(vec_splat_s32(-1), 0))
#ifdef _ARCH_PWR7 
  #define const_vnegonelf  (vec_splats(-1))
#else
  #define const_vnegonelf  (__const_veclf)
#endif
  /***** Rotate *****/
  #define __vrotlbytes(a, s)  (vec_or(vec_slo((a), vset16uc(((s) & 0xf) << 3)), vec_sro((a), set16uc((16 - ((s) & 0xf)) << 3))))
  #define __vrotrbytes(a, s)  (vec_or(vec_sro((a), vset16uc(((s) & 0xf) << 3)), vec_slo((a), set16uc((16 - ((s) & 0xf)) << 3))))
  #define  vrotli(a, s)  __vrotlbytes((a), ((s) << 2))
  #define  vrotlf(a, s)  __vrotlbytes((a), ((s) << 2))
  #define vrotllf(a, s)  __vrotlbytes((a), ((s) << 3))
  #define  vrothi(a, s)  __vrotrbytes((a), ((s) << 2))
  #define  vrothf(a, s)  __vrotrbytes((a), ((s) << 2))
  #define vrothlf(a, s)  __vrotrbytes((a), ((s) << 3))

  /***** Addition *****/
  #define  vaddi(a, b)  (vec_add((a), (b)))
  #define  vaddf(a, b)  (vec_add((a), (b)))
#ifdef _ARCH_PWR7 
  #define  vaddlf(a, b)  (vec_add((a), (b)))
#else
  #define vaddlf __vaddlf
#endif
  /***** Subtraction *****/
  #define  vsubi(a, b)  (vec_sub((a), (b)))
  #define  vsubf(a, b)  (vec_sub((a), (b)))
#ifdef _ARCH_PWR7 
  #define  vsublf(a, b)  (vec_sub((a), (b)))
#else
  #define vsublf __vsublf
#endif
  /***** Multiplication *****/

// NOTE: Try to find a way to do this without double evaluating a

#ifdef _ARCH_PWR7 
#define  vmulf(a, b)  (vec_mul((a), (b)))
  #define  vmullf(a, b)  (vec_mul((a), (b)))
#else
  #define  vmulf(a, b)  (vec_madd((a), (b), vec_xor((a), (a))))
  #define vmullf __vmullf
#endif
  /***** Division *****/
#ifdef _ARCH_PWR7 
  #define vdivf(a, b)  (vec_div((a)), ((b)))
  #define vdivlf(a, b)  (vec_div((a)), ((b)))
#else
  #define vdivf(a, b)  (vmulf((a), vec_re(b)))
  #define vdivlf __vdivlf
#endif

  /***** Fused Multiply Add *****/
  #define vmaddf(a, b, c)  (vec_madd((a), (b), (c)))
#ifdef _ARCH_PWR7 
  #define vmaddlf(a, b, c)  (vec_madd((a), (b), (c)))
#else
  #define vmaddlf __vmaddlf
#endif

  /***** Reciprocal *****/
  #define vrecipf(a)  (vec_re(a))
#ifdef _ARCH_PWR7 
  #define vreciplf(a)  (vec_re(a))
#else
  #define vreciplf __vreciplf
#endif
  /***** Square Root *****/
  #define vsqrtf(a)  (vec_re(vec_rsqrte(a)))
#ifdef _ARCH_PWR7 
#define vsqrtlf(a)  (vec_sqrt(a))
#else
  #define vsqrtlf __vsqrtlf
#endif
  /***** Reciprocal Square Root *****/
  #define vrsqrtf(a)  (vec_rsqrte(a))
#ifdef _ARCH_PWR7 
  #define vrsqrtlf(a)  (vec_rsqrte(a))
#else
  #define vrsqrtlf __vrsqrtlf
#endif

  /***** Not *****/


#ifdef _ARCH_PWR7 
  #define vnoti(a)  (vec_neg(a))
  #define vnotf(a)  (vec_neg(a))
  #define vnotlf(a)  (vec_neg(a))
#else
  #define vnoti(a)  (vec_xor((a), const_vnegonei))
  #define vnotf(a)  (vec_xor((a), const_vnegonei))
  #define vnotlf __vnotlf
#endif

  /***** Or *****/
  #define vori(a, b)  (vec_or((a), (b)))
  #define vorf(a, b)  (vec_or((a), (b)))
#ifdef _ARCH_PWR7 
  #define vorlf(a, b)  (vec_or((a), (b)))
#else
  #define vorlf __vorlf
#endif

  /***** Nor *****/
  #define vnori(a, b)  (vec_nor((a), (b)))
  #define vnorf(a, b)  (vec_nor((a), (b)))
#ifdef _ARCH_PWR7 
  #define vnorlf(a, b)  (vec_nor((a), (b)))
#else
  #define vnorlf __vnorlf
#endif
  /***** And *****/
  #define vandi(a, b)  (vec_and((a), (b)))
  #define vandf(a, b)  (vec_and((a), (b)))
#ifdef _ARCH_PWR7 
  #define vandlf(a, b)  (vec_and((a), (b)))
#else
  #define vandlf __vandlf
#endif
  /***** Nand *****/
  #define vnandi(a, b)  (vnoti(vandi((a), (b))))
  #define vnandf(a, b)  (vnotf(vandf((a), (b))))
#ifdef _ARCH_PWR7 
  #define vnandlf(a, b)  (vnotf(vandf((a), (b))))
#else
  #define vnandlf __vnandlf
#endif
  /***** Xor *****/
  #define vxori(a, b)  (vec_xor((a), (b)))
  #define vxorf(a, b)  (vec_xor((a), (b)))
#ifdef _ARCH_PWR7 
  #define vxorlf(a, b)  (vec_xor((a), (b)))
#else
  #define vxorlf __vxorlf
#endif

  /***** Nxor *****/
  #define vnxori(a, b)  (vnoti(vxori((a), (b))))
  #define vnxorf(a, b)  (vnotf(vxorf((a), (b))))
#ifdef _ARCH_PWR7 
  #define vnxorlf(a, b)  (vnotlf(vxorf((a), (b))))
#else
  #define vnxorlf __vnxorlf
#endif
  /***** Equal To *****/
  #define  vcmpeqi(a, b)  ((veci)(vec_cmpeq((a), (b))))
  #define  vcmpeqf(a, b)  ((veci)(vec_cmpeq((a), (b))))
#ifdef _ARCH_PWR7 
  #define  vcmpeqlf(a, b)  ((veci)(vec_cmpeq((a), (b))))
#else
  #define vcmpeqlf __vcmpeqlf
#endif
  /***** Greater Than *****/
  #define  vcmpgti(a, b)  ((veci)(vec_cmpgt((a), (b))))
  #define  vcmpgtf(a, b)  ((veci)(vec_cmpgt((a), (b))))
#ifdef _ARCH_PWR7 
  #define  vcmpgtlf(a, b)  ((veci)(vec_cmpgt((a), (b))))
#else
  #define vcmpgtlf __vcmpgtlf
#endif

  /***** Greater Than Or Equal To *****/
  #define  vcmpgei(a, b)  ((veci)(vec_cmpge((a), (b))))
  #define  vcmpgef(a, b)  ((veci)(vec_cmpge((a), (b))))
#ifdef _ARCH_PWR7 
  #define  vcmpgelf(a, b)  ((veci)(vec_cmpge((a), (b))))
#else
  #define vcmpgelf __vcmpgelf
#endif

  /***** Less Than *****/
  #define  vcmplti(a, b)  ((veci)(vec_cmplt((a), (b))))
  #define  vcmpltf(a, b)  ((veci)(vec_cmplt((a), (b))))
#ifdef _ARCH_PWR7 
  #define  vcmpltlf(a, b)  ((veci)(vec_cmplt((a), (b))))
#else
  #define vcmpltlf __vcmpltlf
#endif

  /***** Less Than Or Equal To *****/
  #define  vcmplei(a, b)  ((veci)(vec_cmple((a), (b))))
  #define  vcmplef(a, b)  ((veci)(vec_cmple((a), (b))))
#ifdef _ARCH_PWR7 
  #define  vcmplelf(a, b)  ((veci)(vec_cmple((a), (b))))
// NOTE: vec_cmple not listed in Calin's wiki page of builtins for
// PWR7, but has a header definition in the compiler
#else
  #define vcmplelf __vcmplelf
#endif
/*******************************************************************************
 *******************************************************************************
 ***** Mapping to Generic C Implementation
 *******************************************************************************
 *******************************************************************************/
#else

  /***** Data Types *****/
  typedef   __veci   veci;
  typedef   __vecf   vecf;
  typedef  __veclf  veclf;

  /***** Insert *****/
  #define  vinserti  __vinserti
  #define  vinsertf  __vinsertf
  #define vinsertlf __vinsertlf

  /***** Extract *****/
  #define  vextracti  __vextracti
  #define  vextractf  __vextractf
  #define vextractlf __vextractlf

  /***** Set *****/
  #define  vseti  __vseti
  #define  vsetf  __vsetf
  #define vsetlf __vsetlf

  /***** Constant Zero *****/
  #define  const_vzeroi  __const_vzeroi
  #define  const_vzerof  __const_vzerof
  #define const_vzerolf __const_vzerolf

  /***** Constant One *****/
  #define  const_vonei  __const_vonei
  #define  const_vonef  __const_vonef
  #define const_vonelf __const_vonelf

  /***** Constant Two *****/
  #define  const_vtwoi  __const_vtwoi
  #define  const_vtwof  __const_vtwof
  #define const_vtwolf __const_vtwolf

  /***** Constant Negative One *****/
  #define  const_vnegonei  __const_vnegonei
  #define  const_vnegonef  __const_vnegonef
  #define const_vnegonelf __const_vnegonelf

  /***** Rotate *****/
  #define  vrothi  __vrothi
  #define  vrothf  __vrothf
  #define vrothlf __vrothlf
  #define  vrotli  __vrotli
  #define  vrotlf  __vrotlf
  #define vrotllf __vrotllf
  
  /***** Addition *****/
  #define  vaddi  __vaddi
  #define  vaddf  __vaddf
  #define vaddlf __vaddlf

  /***** Subtraction *****/
  #define  vsubi  __vsubi
  #define  vsubf  __vsubf
  #define vsublf __vsublf

  /***** Multiplication *****/
  #define  vmulf   __vmulf
  #define vmullf  __vmullf

  /***** Division *****/
  #define  vdivf   __vdivf
  #define vdivlf  __vdivlf

  /***** Fused Multiply Add *****/
  #define  vmaddf  __vmaddf
  #define vmaddlf __vmaddlf

  /***** Reciprocal *****/
  #define  vrecipf  __vrecipf
  #define vreciplf __vreciplf

  /***** Square Root *****/
  #define  vsqrtf  __vsqrtf
  #define vsqrtlf __vsqrtlf

  /***** Reciprocal Square Root *****/
  #define  vrsqrtf  __vrsqrtf
  #define vrsqrtlf __vrsqrtlf

  /***** Not *****/
  #define  vnoti  __vnoti
  #define  vnotf  __vnotf
  #define vnotlf __vnotlf

  /***** Or *****/
  #define  vori  __vori
  #define  vorf  __vorf
  #define vorlf __vorlf

  /***** Nor *****/
  #define  vnori  __vnori
  #define  vnorf  __vnorf
  #define vnorlf __vnorlf

  /***** And *****/
  #define  vandi  __vandi
  #define  vandf  __vandf
  #define vandlf __vandlf

  /***** Nand *****/
  #define  vnandi  __vnandi
  #define  vnandf  __vnandf
  #define vnandlf __vnandlf

  /***** Xor *****/
  #define  vxori  __vxori
  #define  vxorf  __vxorf
  #define vxorlf __vxorlf

  /***** Nxor *****/
  #define  vnxori  __vnxori
  #define  vnxorf  __vnxorf
  #define vnxorlf __vnxorlf

  /***** Equal To *****/
  #define  vcmpeqi  __vcmpeqi
  #define  vcmpeqf  __vcmpeqf
  #define vcmpeqlf __vcmpeqlf

  /***** Greater Than *****/
  #define  vcmpgti  __vcmpgti
  #define  vcmpgtf  __vcmpgtf
  #define vcmpgtlf __vcmpgtlf

  /***** Greater Than Or Equal To *****/
  #define  vcmpgei  __vcmpgei
  #define  vcmpgef  __vcmpgef
  #define vcmpgelf __vcmpgelf

  /***** Less Than *****/
  #define  vcmplti  __vcmplti
  #define  vcmpltf  __vcmpltf
  #define vcmpltlf __vcmpltlf

  /***** Less Than Or Equal To *****/
  #define  vcmplei  __vcmplei
  #define  vcmplef  __vcmplef
  #define vcmplelf __vcmplelf


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
#define  veci_numElems  (sizeof( veci)/sizeof(   int))
#define  vecf_numElems  (sizeof( vecf)/sizeof( float))
#define veclf_numElems  (sizeof(veclf)/sizeof(double))

/***** Spread (Duplicate functionality of 'Set' by another another name) *****/
#define  vspreadi(a)  ( vseti(a))
#define  vspreadf(a)  ( vsetf(a))
#define vspreadlf(a)  (vsetlf(a))

#define visfinitef(a) ( isfinite(vextractf((a),0)) && isfinite(vextractf((a),1)) && isfinite(vextractf((a),2)) && isfinite(vextractf((a),3)))
#define visfinitelf(a) (isfinite(vextractlf((a),0)) && isfinite(vextractlf((a),1)))


/***** Add to Scalar *****/
#define   vaddis(a, b)  ( vaddi((a),  vseti(b)))
#define   vaddfs(a, b)  ( vaddf((a),  vsetf(b)))
#define  vaddlfs(a, b)  (vaddlf((a), vsetlf(b)))

/***** Subtract a Scalar *****/
#define   vsubis(a, b)  ( vsubi((a),  vseti(b)))
#define   vsubfs(a, b)  ( vsubf((a),  vsetf(b)))
#define  vsublfs(a, b)  (vsublf((a), vsetlf(b)))

/***** Multiply by Scalar *****/
#define   vmulfs(a, b)  ( vmulf((a),  vsetf(b)))
#define  vmullfs(a, b)  (vmullf((a), vsetlf(b)))

/***** Divide by Scalar *****/
#define  vdivfs(a, b)  ( vdivf((a),  vsetf(b)))
#define vdivlfs(a, b)  (vdivlf((a), vsetlf(b)))

/***** Fused Multiply(Vector) Add(Scalar) *****/
#define  vmaddfs(a, b, c)  ( vmaddf((a), (b),  vsetf(c)))
#define vmaddlfs(a, b, c)  (vmaddlf((a), (b), vsetlf(c)))

/***** Fused Multiply(Scalar) Add(Scalar) *****/
#define  vmaddfss(a, b, c)  ( vmaddf((a),  vsetf(b),  vsetf(c)))
#define vmaddlfss(a, b, c)  (vmaddlf((a), vsetlf(b), vsetlf(c)))


#endif //__SIMD_H__
