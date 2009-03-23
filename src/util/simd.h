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


// Solaris does not support sqrtf (float), so just map it to sqrt (double) instead
#if !CMK_HAS_SQRTF
  #define sqrtf(a) ((float)(sqrt((double)(a))))
#endif


// DMK - DEBUG - Flag to force SSE not to be used
#define FORCE_NO_SSE   (0)


////////////////////////////////////////////////////////////////////////////////
// Vector Types

typedef struct __vec_8_c  {           char v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15; } __vec16c;
typedef struct __vec_8_uc {  unsigned char v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15; } __vec16uc;
typedef struct __vec_8_s  {          short v0, v1, v2, v3, v4, v5, v6, v7; } __vec8s;
typedef struct __vec_8_us { unsigned short v0, v1, v2, v3, v4, v5, v6, v7; } __vec8us;
typedef struct __vec_4_i  {            int v0, v1, v2, v3; } __vec4i;
typedef struct __vec_4_ui {   unsigned int v0, v1, v2, v3; } __vec4ui;
typedef struct __vec_4_f  {          float v0, v1, v2, v3; } __vec4f;
typedef struct __vec_2_lf {         double v0, v1; } __vec2lf;

#if defined(__SSE2__) && !(FORCE_NO_SSE)   // SSE2

  typedef  __vec16c  vec16c;
  typedef __vec16uc vec16uc;
  typedef   __vec8s   vec8s;
  typedef  __vec8us  vec8us;
  typedef  __vec4ui  vec4ui;

  typedef __m128i  vec4i;
  typedef  __m128  vec4f;
  typedef __m128d vec2lf;

#elif CMK_CELL_SPE != 0    // Cell - SPE

  typedef vector signed char vec16c;
  typedef vector unsigned char vec16uc;
  typedef vector signed short vec8s;
  typedef vector unsigned short vec8us;
  typedef vector signed int vec4i;
  typedef vector unsigned int vec4ui;
  typedef vector float vec4f;
  typedef vector double vec2lf;

#else                    // General C

  typedef  __vec16c  vec16c;
  typedef __vec16uc vec16uc;
  typedef   __vec8s   vec8s;
  typedef  __vec8us  vec8us;
  typedef   __vec4i   vec4i;
  typedef  __vec4ui  vec4ui;
  typedef   __vec4f   vec4f;
  typedef  __vec2lf  vec2lf;

#endif


////////////////////////////////////////////////////////////////////////////////
// Functions for modifying elements

///// Extract /////
// Desc: Returns element 'i' from vector 'a'
#if defined(__SSE2__) && !(FORCE_NO_SSE)     // SSE2

  inline float vextract4f(const vec4f &a, const int i) { return ((float*)(&a))[i]; }

#elif CMK_CELL_SPE != 0    // Cell - SPE

  #define  vextract16c(a, i) (spu_extract((a), (i)))
  #define vextract16uc(a, i) (spu_extract((a), (i)))
  #define   vextract8s(a, i) (spu_extract((a), (i)))
  #define  vextract8us(a, i) (spu_extract((a), (i)))
  #define   vextract4i(a, i) (spu_extract((a), (i)))
  #define  vextract4ui(a, i) (spu_extract((a), (i)))
  #define   vextract4f(a, i) (spu_extract((a), (i)))
  #define  vextract2lf(a, i) (spu_extract((a), (i)))

#else                    // General C

  inline           char  vextract16c(const  vec16c &a, const int i) { return *(((          char*)(&a))+i); }
  inline  unsigned char vextract16uc(const vec16uc &a, const int i) { return *((( unsigned char*)(&a))+i); }
  inline          short   vextract8s(const   vec8s &a, const int i) { return *(((         short*)(&a))+i); }
  inline unsigned short  vextract8us(const  vec8us &a, const int i) { return *(((unsigned short*)(&a))+i); }
  inline            int   vextract4i(const   vec4i &a, const int i) { return *(((           int*)(&a))+i); }
  inline   unsigned int  vextract4ui(const  vec4ui &a, const int i) { return *(((  unsigned int*)(&a))+i); }
  inline          float   vextract4f(const   vec4f &a, const int i) { return *(((         float*)(&a))+i); }
  inline         double  vextract2lf(const  vec2lf &a, const int i) { return *(((        double*)(&a))+i); }

#endif


///// Insert /////
// Desc: Returns a vector that has scalar 's' inserted into vector 'v' at index 'i'
#if defined(__SSE2__)      // SSE2


#elif CMK_CELL_SPE != 0    // Cell - SPE

  #define  vinsert16c(v, s, i) (spu_insert((s), (v), (i)))
  #define vinsert16uc(v, s, i) (spu_insert((s), (v), (i)))
  #define   vinsert8s(v, s, i) (spu_insert((s), (v), (i)))
  #define  vinsert8us(v, s, i) (spu_insert((s), (v), (i)))
  #define   vinsert4i(v, s, i) (spu_insert((s), (v), (i)))
  #define  vinsert4ui(v, s, i) (spu_insert((s), (v), (i)))
  #define   vinsert4f(v, s, i) (spu_insert((s), (v), (i)))
  #define  vinsert2lf(v, s, i) (spu_insert((s), (v), (i)))

#else                    // General C

  inline  vec16c  vinsert16c( vec16c v, const           char s, const int i) { ((          char*)&v)[i] = s; return v; }
  inline vec16uc vinsert16uc(vec16uc v, const unsigned  char s, const int i) { ((unsigned  char*)&v)[i] = s; return v; }
  inline   vec8s   vinsert8s(  vec8s v, const          short s, const int i) { ((         short*)&v)[i] = s; return v; }
  inline  vec8us  vinsert8us( vec8us v, const unsigned short s, const int i) { ((unsigned short*)&v)[i] = s; return v; }
  inline   vec4i   vinsert4i(  vec4i v, const            int s, const int i) { ((           int*)&v)[i] = s; return v; }
  inline  vec4ui  vinsert4ui( vec4ui v, const unsigned   int s, const int i) { ((unsigned   int*)&v)[i] = s; return v; }
  inline   vec4f   vinsert4f(  vec4f v, const          float s, const int i) { ((         float*)&v)[i] = s; return v; }
  inline  vec2lf  vinsert2lf( vec2lf v, const         double s, const int i) { ((        double*)&v)[i] = s; return v; }

#endif


///// Spread /////
// Desc: Returns a vector that has the scalar value 's' in all elements
#if defined(__SSE2__) && !(FORCE_NO_SSE)     // SSE2

  inline vec4f vspread4f(const float s) { vec4f a; __vec_4_f* aPtr = (__vec_4_f*)(&a); aPtr->v0 = aPtr->v1 = aPtr->v2 = aPtr->v3 = s; return a; }

#elif CMK_CELL_SPE != 0    // Cell - SPE

  #define  vspread16c(s) (spu_splats(s))
  #define vspread16uc(s) (spu_splats(s))
  #define   vspread8s(s) (spu_splats(s))
  #define  vspread8us(s) (spu_splats(s))
  #define   vspread4i(s) (spu_splats(s))
  #define  vspread4ui(s) (spu_splats(s))
  #define   vspread4f(s) (spu_splats(s))
  #define  vspread2lf(s) (spu_splats(s))

#else

  inline  vec16c  vspread16c(const           char s) {  vec16c a; a.v0 = a.v1 = a.v2 = a.v3 = a.v4 = a.v5 = a.v6 = a.v7 = a.v8 = a.v9 = a.v10 = a.v11 = a.v12 = a.v13 = a.v14 = a.v15 = s; return a; }
  inline vec16uc vspread16uc(const unsigned  char s) { vec16uc a; a.v0 = a.v1 = a.v2 = a.v3 = a.v4 = a.v5 = a.v6 = a.v7 = a.v8 = a.v9 = a.v10 = a.v11 = a.v12 = a.v13 = a.v14 = a.v15 = s; return a; }
  inline   vec8s   vspread8s(const          short s) {   vec8s a; a.v0 = a.v1 = a.v2 = a.v3 = a.v4 = a.v5 = a.v6 = a.v7 = s; return a; }
  inline  vec8us  vspread8us(const unsigned short s) {  vec8us a; a.v0 = a.v1 = a.v2 = a.v3 = a.v4 = a.v5 = a.v6 = a.v7 = s; return a; }
  inline   vec4i   vspread4i(const            int s) {   vec4i a; a.v0 = a.v1 = a.v2 = a.v3 = s; return a; }
  inline  vec4ui  vspread4ui(const unsigned   int s) {  vec4ui a; a.v0 = a.v1 = a.v2 = a.v3 = s; return a; }
  inline   vec4f   vspread4f(const          float s) {   vec4f a; a.v0 = a.v1 = a.v2 = a.v3 = s; return a; }
  inline  vec2lf  vspread2lf(const         double s) {  vec2lf a; a.v0 = a.v1 = s; return a; }

#endif


////////////////////////////////////////////////////////////////////////////////
// Shift Operations

///// Rotate /////
// Desc: Returns the vector 'a' rotated (towards high or low) by 's' elements
// NOTE: 'roth' => rotate towards higher element indexes
//       'rotl' => rotate towards lower element indexes
#if defined(__SSE2__)      // SSE2


#elif CMK_CELL_SPE != 0

  #define  vroth16c(a, s) (spu_rlqwbyte((a), (0x10- ((s)&0xf)    ) ))
  #define vroth16uc(a, s) (spu_rlqwbyte((a), (0x10- ((s)&0xf)    ) ))
  #define   vroth8s(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x7)<<1)) ))
  #define  vroth8us(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x7)<<1)) ))
  #define   vroth4i(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x3)<<2)) ))
  #define  vroth4ui(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x3)<<2)) ))
  #define   vroth4f(a, s) (spu_rlqwbyte((a), (0x10-(((s)&0x3)<<2)) ))
  #define  vroth2lf(a, s) (spu_rlqwbyte((a),       (((s)&0x1)<<3)  ))

  #define  vrotl16c(a, s) (spu_rlqwbyte((a),  (s)&0xf    ))
  #define vrotl16uc(a, s) (spu_rlqwbyte((a),  (s)&0xf    ))
  #define   vrotl8s(a, s) (spu_rlqwbyte((a), ((s)&0x7)<<1))
  #define  vrotl8us(a, s) (spu_rlqwbyte((a), ((s)&0x7)<<1))
  #define   vrotl4i(a, s) (spu_rlqwbyte((a), ((s)&0x3)<<2))
  #define  vrotl4ui(a, s) (spu_rlqwbyte((a), ((s)&0x3)<<2))
  #define   vrotl4f(a, s) (spu_rlqwbyte((a), ((s)&0x3)<<2))
  #define  vrotl2lf(a, s) (spu_rlqwbyte((a), ((s)&0x1)<<3))

#else

  inline  vec16c  vroth16c(const  vec16c &a, int s) {  vec16c b;           char* a_ptr = (          char*)(&a);           char* b_ptr = (          char*)(&b); s &= 0xf; b_ptr[0] = a_ptr[(0-s)&0xf]; b_ptr[1] = a_ptr[(1-s)&0xf]; b_ptr[2] = a_ptr[(2-s)&0xf]; b_ptr[3] = a_ptr[(3-s)&0xf]; b_ptr[4] = a_ptr[(4-s)&0xf]; b_ptr[5] = a_ptr[(5-s)&0xf]; b_ptr[6] = a_ptr[(6-s)&0xf]; b_ptr[7] = a_ptr[(7-s)&0xf]; b_ptr[8] = a_ptr[(8-s)&0xf]; b_ptr[9] = a_ptr[(9-s)&0xf]; b_ptr[10] = a_ptr[(10-s)&0xf]; b_ptr[11] = a_ptr[(11-s)&0xf]; b_ptr[12] = a_ptr[(12-s)&0xf]; b_ptr[13] = a_ptr[(13-s)&0xf]; b_ptr[14] = a_ptr[(14-s)&0xf]; b_ptr[15] = a_ptr[(15-s)&0xf]; return b; }
  inline vec16uc vroth16uc(const vec16uc &a, int s) { vec16uc b; unsigned  char* a_ptr = (unsigned  char*)(&a); unsigned  char* b_ptr = (unsigned  char*)(&b); s &= 0xf; b_ptr[0] = a_ptr[(0-s)&0xf]; b_ptr[1] = a_ptr[(1-s)&0xf]; b_ptr[2] = a_ptr[(2-s)&0xf]; b_ptr[3] = a_ptr[(3-s)&0xf]; b_ptr[4] = a_ptr[(4-s)&0xf]; b_ptr[5] = a_ptr[(5-s)&0xf]; b_ptr[6] = a_ptr[(6-s)&0xf]; b_ptr[7] = a_ptr[(7-s)&0xf]; b_ptr[8] = a_ptr[(8-s)&0xf]; b_ptr[9] = a_ptr[(9-s)&0xf]; b_ptr[10] = a_ptr[(10-s)&0xf]; b_ptr[11] = a_ptr[(11-s)&0xf]; b_ptr[12] = a_ptr[(12-s)&0xf]; b_ptr[13] = a_ptr[(13-s)&0xf]; b_ptr[14] = a_ptr[(14-s)&0xf]; b_ptr[15] = a_ptr[(15-s)&0xf]; return b; }
  inline   vec8s   vroth8s(const   vec8s &a, int s) {   vec8s b;          short* a_ptr = (         short*)(&a);          short* b_ptr = (         short*)(&b); s &= 0x7; b_ptr[0] = a_ptr[(0-s)&0x7]; b_ptr[1] = a_ptr[(1-s)&0x7]; b_ptr[2] = a_ptr[(2-s)&0x7]; b_ptr[3] = a_ptr[(3-s)&0x7]; b_ptr[4] = a_ptr[(4-s)&0x7]; b_ptr[5] = a_ptr[(5-s)&0x7]; b_ptr[6] = a_ptr[(6-s)&0x7]; b_ptr[7] = a_ptr[(7-s)&0x7]; return b; }
  inline  vec8us  vroth8us(const  vec8us &a, int s) {  vec8us b; unsigned short* a_ptr = (unsigned short*)(&a); unsigned short* b_ptr = (unsigned short*)(&b); s &= 0x7; b_ptr[0] = a_ptr[(0-s)&0x7]; b_ptr[1] = a_ptr[(1-s)&0x7]; b_ptr[2] = a_ptr[(2-s)&0x7]; b_ptr[3] = a_ptr[(3-s)&0x7]; b_ptr[4] = a_ptr[(4-s)&0x7]; b_ptr[5] = a_ptr[(5-s)&0x7]; b_ptr[6] = a_ptr[(6-s)&0x7]; b_ptr[7] = a_ptr[(7-s)&0x7]; return b; }
  inline   vec4i   vroth4i(const   vec4i &a, int s) {   vec4i b;            int* a_ptr = (           int*)(&a);            int* b_ptr = (           int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
  inline  vec4ui  vroth4ui(const  vec4ui &a, int s) {  vec4ui b; unsigned   int* a_ptr = (unsigned   int*)(&a); unsigned   int* b_ptr = (unsigned   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
  inline   vec4f   vroth4f(const   vec4f &a, int s) {   vec4f b;          float* a_ptr = (         float*)(&a);          float* b_ptr = (         float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0-s)&0x3]; b_ptr[1] = a_ptr[(1-s)&0x3]; b_ptr[2] = a_ptr[(2-s)&0x3]; b_ptr[3] = a_ptr[(3-s)&0x3]; return b; }
  inline  vec2lf  vroth2lf(const  vec2lf &a, int s) {  vec2lf b;         double* a_ptr = (        double*)(&a);         double* b_ptr = (        double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0-s)&0x1]; b_ptr[1] = a_ptr[(1-s)&0x1]; return b; }

  inline  vec16c  vrotl16c(const  vec16c &a, int s) {  vec16c b;           char* a_ptr = (          char*)(&a);           char* b_ptr = (          char*)(&b); s &= 0xf; b_ptr[0] = a_ptr[(0+s)&0xf]; b_ptr[1] = a_ptr[(1+s)&0xf]; b_ptr[2] = a_ptr[(2+s)&0xf]; b_ptr[3] = a_ptr[(3+s)&0xf]; b_ptr[4] = a_ptr[(4+s)&0xf]; b_ptr[5] = a_ptr[(5+s)&0xf]; b_ptr[6] = a_ptr[(6+s)&0xf]; b_ptr[7] = a_ptr[(7+s)&0xf]; b_ptr[8] = a_ptr[(8+s)&0xf]; b_ptr[9] = a_ptr[(9+s)&0xf]; b_ptr[10] = a_ptr[(10+s)&0xf]; b_ptr[11] = a_ptr[(11+s)&0xf]; b_ptr[12] = a_ptr[(12+s)&0xf]; b_ptr[13] = a_ptr[(13+s)&0xf]; b_ptr[14] = a_ptr[(14+s)&0xf]; b_ptr[15] = a_ptr[(15+s)&0xf]; return b; }
  inline vec16uc vrotl16uc(const vec16uc &a, int s) { vec16uc b; unsigned  char* a_ptr = (unsigned  char*)(&a); unsigned  char* b_ptr = (unsigned  char*)(&b); s &= 0xf; b_ptr[0] = a_ptr[(0+s)&0xf]; b_ptr[1] = a_ptr[(1+s)&0xf]; b_ptr[2] = a_ptr[(2+s)&0xf]; b_ptr[3] = a_ptr[(3+s)&0xf]; b_ptr[4] = a_ptr[(4+s)&0xf]; b_ptr[5] = a_ptr[(5+s)&0xf]; b_ptr[6] = a_ptr[(6+s)&0xf]; b_ptr[7] = a_ptr[(7+s)&0xf]; b_ptr[8] = a_ptr[(8+s)&0xf]; b_ptr[9] = a_ptr[(9+s)&0xf]; b_ptr[10] = a_ptr[(10+s)&0xf]; b_ptr[11] = a_ptr[(11+s)&0xf]; b_ptr[12] = a_ptr[(12+s)&0xf]; b_ptr[13] = a_ptr[(13+s)&0xf]; b_ptr[14] = a_ptr[(14+s)&0xf]; b_ptr[15] = a_ptr[(15+s)&0xf]; return b; }
  inline   vec8s   vrotl8s(const   vec8s &a, int s) {   vec8s b;          short* a_ptr = (         short*)(&a);          short* b_ptr = (         short*)(&b); s &= 0x7; b_ptr[0] = a_ptr[(0+s)&0x7]; b_ptr[1] = a_ptr[(1+s)&0x7]; b_ptr[2] = a_ptr[(2+s)&0x7]; b_ptr[3] = a_ptr[(3+s)&0x7]; b_ptr[4] = a_ptr[(4+s)&0x7]; b_ptr[5] = a_ptr[(5+s)&0x7]; b_ptr[6] = a_ptr[(6+s)&0x7]; b_ptr[7] = a_ptr[(7+s)&0x7]; return b; }
  inline  vec8us  vrotl8us(const  vec8us &a, int s) {  vec8us b; unsigned short* a_ptr = (unsigned short*)(&a); unsigned short* b_ptr = (unsigned short*)(&b); s &= 0x7; b_ptr[0] = a_ptr[(0+s)&0x7]; b_ptr[1] = a_ptr[(1+s)&0x7]; b_ptr[2] = a_ptr[(2+s)&0x7]; b_ptr[3] = a_ptr[(3+s)&0x7]; b_ptr[4] = a_ptr[(4+s)&0x7]; b_ptr[5] = a_ptr[(5+s)&0x7]; b_ptr[6] = a_ptr[(6+s)&0x7]; b_ptr[7] = a_ptr[(7+s)&0x7]; return b; }
  inline   vec4i   vrotl4i(const   vec4i &a, int s) {   vec4i b;            int* a_ptr = (           int*)(&a);            int* b_ptr = (           int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
  inline  vec4ui  vrotl4ui(const  vec4ui &a, int s) {  vec4ui b; unsigned   int* a_ptr = (unsigned   int*)(&a); unsigned   int* b_ptr = (unsigned   int*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
  inline   vec4f   vrotl4f(const   vec4f &a, int s) {   vec4f b;          float* a_ptr = (         float*)(&a);          float* b_ptr = (         float*)(&b); s &= 0x3; b_ptr[0] = a_ptr[(0+s)&0x3]; b_ptr[1] = a_ptr[(1+s)&0x3]; b_ptr[2] = a_ptr[(2+s)&0x3]; b_ptr[3] = a_ptr[(3+s)&0x3]; return b; }
  inline  vec2lf  vrotl2lf(const  vec2lf &a, int s) {  vec2lf b;         double* a_ptr = (        double*)(&a);         double* b_ptr = (        double*)(&b); s &= 0x1; b_ptr[0] = a_ptr[(0+s)&0x1]; b_ptr[1] = a_ptr[(1+s)&0x1]; return b; }

#endif


////////////////////////////////////////////////////////////////////////////////
// Arithmetic Functions


///// Addition /////
// Desc: Returns a vector that has the corresponding elements of 'a' and 'b' added together
#if defined(__SSE2__) && !(FORCE_NO_SSE)     // SSE2

  #define vadd4f(a, b)  (_mm_add_ps((a), (b)))

#elif CMK_CELL_SPE != 0    // Cell - SPE

  #define  vadd16c(a, b) (spu_add((a), (b)))
  #define vadd16uc(a, b) (spu_add((a), (b)))
  #define   vadd8s(a, b) (spu_add((a), (b)))
  #define  vadd8us(a, b) (spu_add((a), (b)))
  #define   vadd4i(a, b) (spu_add((a), (b)))
  #define  vadd4ui(a, b) (spu_add((a), (b)))
  #define   vadd4f(a, b) (spu_add((a), (b)))
  #define  vadd2lf(a, b) (spu_add((a), (b)))

  #define  vadd16cs(a, s) (spu_add((a),  vspread16c(s)))
  #define vadd16ucs(a, s) (spu_add((a), vspread16uc(s)))
  #define   vadd8ss(a, s) (spu_add((a),   vspread8s(s)))
  #define  vadd8uss(a, s) (spu_add((a),  vspread8us(s)))
  #define   vadd4is(a, s) (spu_add((a),   vspread4i(s)))
  #define  vadd4uis(a, s) (spu_add((a),  vspread4ui(s)))
  #define   vadd4fs(a, s) (spu_add((a),   vspread4f(s)))
  #define  vadd2lfs(a, s) (spu_add((a),  vspread2lf(s)))

#else                    // General C

  inline  vec16c  vadd16c(const  vec16c &a, const  vec16c &b) {  vec16c c; c.v0 = a.v0 + b.v0; c.v1 = a.v1 + b.v1; c.v2 = a.v2 + b.v2; c.v3 = a.v3 + b.v3; c.v4 = a.v4 + b.v4; c.v5 = a.v5 + b.v5; c.v6 = a.v6 + b.v6; c.v7 = a.v7 + b.v7; c.v8 = a.v8 + b.v8; c.v9 = a.v9 + b.v9; c.v10 = a.v10 + b.v10; c.v11 = a.v11 + b.v11; c.v12 = a.v12 + b.v12; c.v13 = a.v13 + b.v13; c.v14 = a.v14 + b.v14; c.v15 = a.v15 + b.v15; return c; }
  inline vec16uc vadd16uc(const vec16uc &a, const vec16uc &b) { vec16uc c; c.v0 = a.v0 + b.v0; c.v1 = a.v1 + b.v1; c.v2 = a.v2 + b.v2; c.v3 = a.v3 + b.v3; c.v4 = a.v4 + b.v4; c.v5 = a.v5 + b.v5; c.v6 = a.v6 + b.v6; c.v7 = a.v7 + b.v7; c.v8 = a.v8 + b.v8; c.v9 = a.v9 + b.v9; c.v10 = a.v10 + b.v10; c.v11 = a.v11 + b.v11; c.v12 = a.v12 + b.v12; c.v13 = a.v13 + b.v13; c.v14 = a.v14 + b.v14; c.v15 = a.v15 + b.v15; return c; }
  inline   vec8s   vadd8s(const   vec8s &a, const   vec8s &b) {   vec8s c; c.v0 = a.v0 + b.v0; c.v1 = a.v1 + b.v1; c.v2 = a.v2 + b.v2; c.v3 = a.v3 + b.v3; c.v4 = a.v4 + b.v4; c.v5 = a.v5 + b.v5; c.v6 = a.v6 + b.v6; c.v7 = a.v7 + b.v7; return c; }
  inline  vec8us  vadd8us(const  vec8us &a, const  vec8us &b) {  vec8us c; c.v0 = a.v0 + b.v0; c.v1 = a.v1 + b.v1; c.v2 = a.v2 + b.v2; c.v3 = a.v3 + b.v3; c.v4 = a.v4 + b.v4; c.v5 = a.v5 + b.v5; c.v6 = a.v6 + b.v6; c.v7 = a.v7 + b.v7; return c; }
  inline   vec4i   vadd4i(const   vec4i &a, const   vec4i &b) {   vec4i c; c.v0 = a.v0 + b.v0; c.v1 = a.v1 + b.v1; c.v2 = a.v2 + b.v2; c.v3 = a.v3 + b.v3; return c; }
  inline  vec4ui  vadd4ui(const  vec4ui &a, const  vec4ui &b) {  vec4ui c; c.v0 = a.v0 + b.v0; c.v1 = a.v1 + b.v1; c.v2 = a.v2 + b.v2; c.v3 = a.v3 + b.v3; return c; }
  inline   vec4f   vadd4f(const   vec4f &a, const   vec4f &b) {   vec4f c; c.v0 = a.v0 + b.v0; c.v1 = a.v1 + b.v1; c.v2 = a.v2 + b.v2; c.v3 = a.v3 + b.v3; return c; }
  inline  vec2lf  vadd2lf(const  vec2lf &a, const  vec2lf &b) {  vec2lf c; c.v0 = a.v0 + b.v0; c.v1 = a.v1 + b.v1; return c; }

  inline  vec16c  vadd16cs(const  vec16c &a, const           char &s) {  vec16c c; c.v0 = a.v0 + s; c.v1 = a.v1 + s; c.v2 = a.v2 + s; c.v3 = a.v3 + s; c.v4 = a.v4 + s; c.v5 = a.v5 + s; c.v6 = a.v6 + s; c.v7 = a.v7 + s; c.v8 = a.v8 + s; c.v9 = a.v9 + s; c.v10 = a.v10 + s; c.v11 = a.v11 + s; c.v12 = a.v12 + s; c.v13 = a.v13 + s; c.v14 = a.v14 + s; c.v15 = a.v15 + s; return c; }
  inline vec16uc vadd16ucs(const vec16uc &a, const unsigned  char &s) { vec16uc c; c.v0 = a.v0 + s; c.v1 = a.v1 + s; c.v2 = a.v2 + s; c.v3 = a.v3 + s; c.v4 = a.v4 + s; c.v5 = a.v5 + s; c.v6 = a.v6 + s; c.v7 = a.v7 + s; c.v8 = a.v8 + s; c.v9 = a.v9 + s; c.v10 = a.v10 + s; c.v11 = a.v11 + s; c.v12 = a.v12 + s; c.v13 = a.v13 + s; c.v14 = a.v14 + s; c.v15 = a.v15 + s; return c; }
  inline   vec8s   vadd8ss(const   vec8s &a, const          short &s) {   vec8s c; c.v0 = a.v0 + s; c.v1 = a.v1 + s; c.v2 = a.v2 + s; c.v3 = a.v3 + s; c.v4 = a.v4 + s; c.v5 = a.v5 + s; c.v6 = a.v6 + s; c.v7 = a.v7 + s; return c; }
  inline  vec8us  vadd8uss(const  vec8us &a, const unsigned short &s) {  vec8us c; c.v0 = a.v0 + s; c.v1 = a.v1 + s; c.v2 = a.v2 + s; c.v3 = a.v3 + s; c.v4 = a.v4 + s; c.v5 = a.v5 + s; c.v6 = a.v6 + s; c.v7 = a.v7 + s; return c; }
  inline   vec4i   vadd4is(const   vec4i &a, const            int &s) {   vec4i c; c.v0 = a.v0 + s; c.v1 = a.v1 + s; c.v2 = a.v2 + s; c.v3 = a.v3 + s; return c; }
  inline  vec4ui  vadd4uis(const  vec4ui &a, const unsigned   int &s) {  vec4ui c; c.v0 = a.v0 + s; c.v1 = a.v1 + s; c.v2 = a.v2 + s; c.v3 = a.v3 + s; return c; }
  inline   vec4f   vadd4fs(const   vec4f &a, const          float &s) {   vec4f c; c.v0 = a.v0 + s; c.v1 = a.v1 + s; c.v2 = a.v2 + s; c.v3 = a.v3 + s; return c; }
  inline  vec2lf  vadd2lfs(const  vec2lf &a, const         double &s) {  vec2lf c; c.v0 = a.v0 + s; c.v1 = a.v1 + s; return c; }

  // Overide C++ operators
  #if defined(__cplusplus)

    inline  vec4i operator+(const  vec4i& a, const  vec4i& b) { return  vadd4i(a,b); }
    inline  vec4f operator+(const  vec4f& a, const  vec4f& b) { return  vadd4f(a,b); }
    inline vec2lf operator+(const vec2lf& a, const vec2lf& b) { return vadd2lf(a,b); }

    inline  vec4i operator+=( vec4i& a, const  vec4i& b) { return (a =  vadd4i(a,b)); }
    inline  vec4f operator+=( vec4f& a, const  vec4f& b) { return (a =  vadd4f(a,b)); }
    inline vec2lf operator+=(vec2lf& a, const vec2lf& b) { return (a = vadd2lf(a,b)); }

  #endif

#endif


///// Subtraction /////
// Desc: Returns a vector that where the correspinding elements of vector 'b' have been
//   subtracted from vector 'a'
#if defined(__SSE2__) && !(FORCE_NO_SSE)     // SSE2


#elif CMK_CELL_SPE != 0    // Cell - SPE

  #define  vsub16c(a, b) (spu_sub((a), (b)))
  #define vsub16uc(a, b) (spu_sub((a), (b)))
  #define   vsub8s(a, b) (spu_sub((a), (b)))
  #define  vsub8us(a, b) (spu_sub((a), (b)))
  #define   vsub4i(a, b) (spu_sub((a), (b)))
  #define  vsub4ui(a, b) (spu_sub((a), (b)))
  #define   vsub4f(a, b) (spu_sub((a), (b)))
  #define  vsub2lf(a, b) (spu_sub((a), (b)))

  #define  vsub16cs(a, s) (spu_sub((a),  vspread16c(s)))
  #define vsub16ucs(a, s) (spu_sub((a), vspread16uc(s)))
  #define   vsub8ss(a, s) (spu_sub((a),   vspread8s(s)))
  #define  vsub8uss(a, s) (spu_sub((a),  vspread8us(s)))
  #define   vsub4is(a, s) (spu_sub((a),   vspread4i(s)))
  #define  vsub4uis(a, s) (spu_sub((a),  vspread4ui(s)))
  #define   vsub4fs(a, s) (spu_sub((a),   vspread4f(s)))
  #define  vsub2lfs(a, s) (spu_sub((a),  vspread2lf(s)))

#else                    // General C

  inline  vec16c  vsub16c(const  vec16c &a, const  vec16c &b) {  vec16c c; c.v0 = a.v0 - b.v0; c.v1 = a.v1 - b.v1; c.v2 = a.v2 - b.v2; c.v3 = a.v3 - b.v3; c.v4 = a.v4 - b.v4; c.v5 = a.v5 - b.v5; c.v6 = a.v6 - b.v6; c.v7 = a.v7 - b.v7; c.v8 = a.v8 - b.v8; c.v9 = a.v9 - b.v9; c.v10 = a.v10 - b.v10; c.v11 = a.v11 - b.v11; c.v12 = a.v12 - b.v12; c.v13 = a.v13 - b.v13; c.v14 = a.v14 - b.v14; c.v15 = a.v15 - b.v15; return c; }
  inline vec16uc vsub16uc(const vec16uc &a, const vec16uc &b) { vec16uc c; c.v0 = a.v0 - b.v0; c.v1 = a.v1 - b.v1; c.v2 = a.v2 - b.v2; c.v3 = a.v3 - b.v3; c.v4 = a.v4 - b.v4; c.v5 = a.v5 - b.v5; c.v6 = a.v6 - b.v6; c.v7 = a.v7 - b.v7; c.v8 = a.v8 - b.v8; c.v9 = a.v9 - b.v9; c.v10 = a.v10 - b.v10; c.v11 = a.v11 - b.v11; c.v12 = a.v12 - b.v12; c.v13 = a.v13 - b.v13; c.v14 = a.v14 - b.v14; c.v15 = a.v15 - b.v15; return c; }
  inline   vec8s   vsub8s(const   vec8s &a, const   vec8s &b) {   vec8s c; c.v0 = a.v0 - b.v0; c.v1 = a.v1 - b.v1; c.v2 = a.v2 - b.v2; c.v3 = a.v3 - b.v3; c.v4 = a.v4 - b.v4; c.v5 = a.v5 - b.v5; c.v6 = a.v6 - b.v6; c.v7 = a.v7 - b.v7; return c; }
  inline  vec8us  vsub8us(const  vec8us &a, const  vec8us &b) {  vec8us c; c.v0 = a.v0 - b.v0; c.v1 = a.v1 - b.v1; c.v2 = a.v2 - b.v2; c.v3 = a.v3 - b.v3; c.v4 = a.v4 - b.v4; c.v5 = a.v5 - b.v5; c.v6 = a.v6 - b.v6; c.v7 = a.v7 - b.v7; return c; }
  inline   vec4i   vsub4i(const   vec4i &a, const   vec4i &b) {   vec4i c; c.v0 = a.v0 - b.v0; c.v1 = a.v1 - b.v1; c.v2 = a.v2 - b.v2; c.v3 = a.v3 - b.v3; return c; }
  inline  vec4ui  vsub4ui(const  vec4ui &a, const  vec4ui &b) {  vec4ui c; c.v0 = a.v0 - b.v0; c.v1 = a.v1 - b.v1; c.v2 = a.v2 - b.v2; c.v3 = a.v3 - b.v3; return c; }
  inline   vec4f   vsub4f(const   vec4f &a, const   vec4f &b) {   vec4f c; c.v0 = a.v0 - b.v0; c.v1 = a.v1 - b.v1; c.v2 = a.v2 - b.v2; c.v3 = a.v3 - b.v3; return c; }
  inline  vec2lf  vsub2lf(const  vec2lf &a, const  vec2lf &b) {  vec2lf c; c.v0 = a.v0 - b.v0; c.v1 = a.v1 - b.v1; return c; }

  inline  vec16c  vsub16c(const  vec16c &a, const           char &s) {  vec16c c; c.v0 = a.v0 - s; c.v1 = a.v1 - s; c.v2 = a.v2 - s; c.v3 = a.v3 - s; c.v4 = a.v4 - s; c.v5 = a.v5 - s; c.v6 = a.v6 - s; c.v7 = a.v7 - s; c.v8 = a.v8 - s; c.v9 = a.v9 - s; c.v10 = a.v10 - s; c.v11 = a.v11 - s; c.v12 = a.v12 - s; c.v13 = a.v13 - s; c.v14 = a.v14 - s; c.v15 = a.v15 - s; return c; }
  inline vec16uc vsub16uc(const vec16uc &a, const unsigned  char &s) { vec16uc c; c.v0 = a.v0 - s; c.v1 = a.v1 - s; c.v2 = a.v2 - s; c.v3 = a.v3 - s; c.v4 = a.v4 - s; c.v5 = a.v5 - s; c.v6 = a.v6 - s; c.v7 = a.v7 - s; c.v8 = a.v8 - s; c.v9 = a.v9 - s; c.v10 = a.v10 - s; c.v11 = a.v11 - s; c.v12 = a.v12 - s; c.v13 = a.v13 - s; c.v14 = a.v14 - s; c.v15 = a.v15 - s; return c; }
  inline   vec8s   vsub8s(const   vec8s &a, const          short &s) {   vec8s c; c.v0 = a.v0 - s; c.v1 = a.v1 - s; c.v2 = a.v2 - s; c.v3 = a.v3 - s; c.v4 = a.v4 - s; c.v5 = a.v5 - s; c.v6 = a.v6 - s; c.v7 = a.v7 - s; return c; }
  inline  vec8us  vsub8us(const  vec8us &a, const unsigned short &s) {  vec8us c; c.v0 = a.v0 - s; c.v1 = a.v1 - s; c.v2 = a.v2 - s; c.v3 = a.v3 - s; c.v4 = a.v4 - s; c.v5 = a.v5 - s; c.v6 = a.v6 - s; c.v7 = a.v7 - s; return c; }
  inline   vec4i   vsub4i(const   vec4i &a, const            int &s) {   vec4i c; c.v0 = a.v0 - s; c.v1 = a.v1 - s; c.v2 = a.v2 - s; c.v3 = a.v3 - s; return c; }
  inline  vec4ui  vsub4ui(const  vec4ui &a, const unsigned   int &s) {  vec4ui c; c.v0 = a.v0 - s; c.v1 = a.v1 - s; c.v2 = a.v2 - s; c.v3 = a.v3 - s; return c; }
  inline   vec4f   vsub4f(const   vec4f &a, const          float &s) {   vec4f c; c.v0 = a.v0 - s; c.v1 = a.v1 - s; c.v2 = a.v2 - s; c.v3 = a.v3 - s; return c; }
  inline  vec2lf  vsub2lf(const  vec2lf &a, const         double &s) {  vec2lf c; c.v0 = a.v0 - s; c.v1 = a.v1 - s; return c; }

  // Overide C++ operators
  #if defined(__cplusplus)

    inline  vec4i operator-(const  vec4i& a, const  vec4i& b) { return  vsub4i(a,b); }
    inline  vec4f operator-(const  vec4f& a, const  vec4f& b) { return  vsub4f(a,b); }
    inline vec2lf operator-(const vec2lf& a, const vec2lf& b) { return vsub2lf(a,b); }

    inline  vec4i operator-=( vec4i& a, const  vec4i& b) { return (a =  vsub4i(a,b)); }
    inline  vec4f operator-=( vec4f& a, const  vec4f& b) { return (a =  vsub4f(a,b)); }
    inline vec2lf operator-=(vec2lf& a, const vec2lf& b) { return (a = vsub2lf(a,b)); }

  #endif

#endif


///// Multiply & Multiply-Add /////
#if defined(__SSE2__) && !(FORCE_NO_SSE)     // SSE2


#elif CMK_CELL_SPE != 0    // Cell - SPE

  #define  vmul4s(a, b) (spu_mul((a), (b)))
  #define  vmul4f(a, b) (spu_mul((a), (b)))
  #define vmul2lf(a, b) (spu_mul((a), (b)))

  #define  vmul4ss(a, s) (spu_mul((a),  vspread8s(s)))
  #define  vmul4fs(a, s) (spu_mul((a),  vspread4f(s)))
  #define vmul2lfs(a, s) (spu_mul((a), vspread2lf(s)))

  #define  vmadd4s(a, b, c) (spu_madd((a), (b), (c)))
  #define  vmadd4f(a, b, c) (spu_madd((a), (b), (c)))
  #define vmadd2lf(a, b, c) (spu_madd((a), (b), (c)))

  #define  vmadd4ss(a, b, s) (spu_madd((a), (b),  vspread4i(s)))
  #define  vmadd4fs(a, b, s) (spu_madd((a), (b),  vspread4f(s)))
  #define vmadd2lfs(a, b, s) (spu_madd((a), (b), vspread2lf(s)))

#else

  inline  vec4i  vmul4s(const  vec8s &a, const  vec8s &b) {  vec4i c; c.v0 = ((int)a.v1) * ((int)b.v1); c.v1 = ((int)a.v3) * ((int)b.v3); c.v2 = ((int)a.v5) * ((int)b.v5); c.v3 = ((int)a.v7) * ((int)b.v7); return c; }
  inline  vec4f  vmul4f(const  vec4f &a, const  vec4f &b) {  vec4f c; c.v0 = a.v0 * b.v0; c.v1 = a.v1 * b.v1; c.v2 = a.v2 * b.v2; c.v3 = a.v3 * b.v3; return c; }
  inline vec2lf vmul2lf(const vec2lf &a, const vec2lf &b) { vec2lf c; c.v0 = a.v0 * b.v0; c.v1 = a.v1 * b.v1; return c; }

  inline  vec4i  vmul4ss(const  vec8s &a, const  short &s) {  vec4i c; c.v0 = ((int)a.v1) * ((int)s); c.v1 = ((int)a.v3) * ((int)s); c.v2 = ((int)a.v5) * ((int)s); c.v3 = ((int)a.v7) * ((int)s); return c; }
  inline  vec4f  vmul4fs(const  vec4f &a, const  float &s) {  vec4f c; c.v0 = a.v0 * s; c.v1 = a.v1 * s; c.v2 = a.v2 * s; c.v3 = a.v3 * s; return c; }
  inline vec2lf vmul2lfs(const vec2lf &a, const double &s) { vec2lf c; c.v0 = a.v0 * s; c.v1 = a.v1 * s; return c; }

  inline  vec4i  vmadd4s(const  vec8s &a, const  vec8s &b, const  vec4i &d) {  vec4i c; c.v0 = ((int)a.v1) * ((int)b.v1) + d.v0; c.v1 = ((int)a.v3) * ((int)b.v3) + d.v1; c.v2 = ((int)a.v5) * ((int)b.v5) + d.v2; c.v3 = ((int)a.v7) * ((int)b.v7) + d.v3; return c; }
  inline  vec4f  vmadd4f(const  vec4f &a, const  vec4f &b, const  vec4f &d) {  vec4f c; c.v0 = a.v0 * b.v0 + d.v0; c.v1 = a.v1 * b.v1 + d.v1; c.v2 = a.v2 * b.v2 + d.v2; c.v3 = a.v3 * b.v3 + d.v3; return c; }
  inline vec2lf vmadd2lf(const vec2lf &a, const vec2lf &b, const vec2lf &d) { vec2lf c; c.v0 = a.v0 * b.v0 + d.v0; c.v1 = a.v1 * b.v1 + d.v1; return c; }

  inline  vec4i  vmadd4ss(const  vec8s &a, const  vec8s &b, const  short &s) {  vec4i c; c.v0 = ((int)a.v1) * ((int)b.v1) + s; c.v1 = ((int)a.v3) * ((int)b.v3) + s; c.v2 = ((int)a.v5) * ((int)b.v5) + s; c.v3 = ((int)a.v7) * ((int)b.v7) + s; return c; }
  inline  vec4f  vmadd4fs(const  vec4f &a, const  vec4f &b, const  float &s) {  vec4f c; c.v0 = a.v0 * b.v0 + s; c.v1 = a.v1 * b.v1 + s; c.v2 = a.v2 * b.v2 + s; c.v3 = a.v3 * b.v3 + s; return c; }
  inline vec2lf vmadd2lfs(const vec2lf &a, const vec2lf &b, const double &s) { vec2lf c; c.v0 = a.v0 * b.v0 + s; c.v1 = a.v1 * b.v1 + s; return c; }

  // Overide C++ operators
  #if defined(__cplusplus)

    //inline  vec4i operator*(const  vec4i& a, const  vec4i& b) { return  vmul4i(a,b); }
    inline  vec4f operator*(const  vec4f& a, const  vec4f& b) { return  vmul4f(a,b); }
    inline vec2lf operator*(const vec2lf& a, const vec2lf& b) { return vmul2lf(a,b); }

  #endif

#endif


///// Divide /////

// DMK - TODO : FIXME - Figure out the SPE version of these functions (for now, just use general C++)
//   SPE version of the functions to use the "frest" and "fi" 
#if defined(__SSE2__) && !(FORCE_NO_SSE)     // SSE2


#elif CMK_CELL_SPE != 0

  inline vec4i vdiv4i(const vec4i a, const vec4i b) { vec4i c; __vec4i* __c = (__vec4i*)(&c); __vec4i* __a = (__vec4i*)(&a); __vec4i* __b = (__vec4i*)(&b); __c->v0 = __a->v0 / __b->v0; __c->v1 = __a->v1 / __b->v1; return c; }
  #define vdiv4f(a, b) (spu_mul((a), spu_re(b)))
  inline vec2lf vdiv2lf(const vec2lf a, const vec2lf b) { vec2lf c; __vec2lf* __c = (__vec2lf*)(&c); __vec2lf* __a = (__vec2lf*)(&a); __vec2lf* __b = (__vec2lf*)(&b); __c->v0 = __a->v0 / __b->v0; __c->v1 = __a->v1 / __b->v1; return c; }

#else

  inline vec4i vdiv4i(const vec4i &a, const vec4i &b) { vec4i c; c.v0 = a.v0 / b.v0; c.v1 = a.v1 / b.v1; c.v2 = a.v2 / b.v2; c.v3 = a.v3 / b.v3; return c; }
  inline vec4f vdiv4f(const vec4f &a, const vec4f &b) { vec4f c; c.v0 = a.v0 / b.v0; c.v1 = a.v1 / b.v1; c.v2 = a.v2 / b.v2; c.v3 = a.v3 / b.v3; return c; }
  inline vec2lf vdiv2lf(const vec2lf &a, const vec2lf &b) { vec2lf c; c.v0 = a.v0 / b.v0; c.v1 = a.v1 / b.v1; return c; }

  // Overide C++ operators
  #if defined(__cplusplus)

    //inline  vec4i operator*(const  vec4i& a, const  vec4i& b) { return  vmul4i(a,b); }
    inline  vec4f operator/(const  vec4f& a, const  vec4f& b) { return  vdiv4f(a,b); }
    inline vec2lf operator/(const vec2lf& a, const vec2lf& b) { return vdiv2lf(a,b); }

  #endif

#endif


///// Misc : TODO : Organize later /////
#if defined(__SSE2__) && !(FORCE_NO_SSE)     // SSE2

  // DMK - DEBUG - Disable this single sqrt function to see how much of an impact it has
  #if 1
    #define vsqrt4f(a)  (_mm_sqrt_ps(a))
  #else
    inline vec4f vsqrt4f(const vec4f &a) {
      vec4f b;
      float *a_f = (float*)(&a), *b_f = (float*)(&b);
      b_f[0] = sqrtf(a_f[0]);
      b_f[1] = sqrtf(a_f[1]);
      b_f[2] = sqrtf(a_f[2]);
      b_f[3] = sqrtf(a_f[3]);
      return b;
    }
  #endif

#elif CMK_CELL_SPE != 0

  #define vrecip4f(a) (spu_re(a))
  #define vsqrt4f(a) (spu_re(spu_rsqrte(a)))
  #define vrsqrt4f(a) (spu_rsqrte(a))

#else

  inline vec4f vrecip4f(const vec4f &a) { vec4f b; b.v0 = 1.0f / a.v0; b.v1 = 1.0f / a.v1; b.v2 = 1.0f / a.v2; b.v3 = 1.0f / a.v3; return b; }
  inline vec4f vsqrt4f(const vec4f &a) { vec4f b; b.v0 = sqrtf(a.v0); b.v1 = sqrtf(a.v1); b.v2 = sqrtf(a.v2); b.v3 = sqrtf(a.v3); return b; }
  inline vec4f vrsqrt4f(const vec4f &a) { vec4f b; b.v0 = 1.0f / sqrtf(a.v0); b.v1 = 1.0f / sqrtf(a.v1); b.v2 = 1.0f / sqrtf(a.v2); b.v3 = 1.0f / sqrtf(a.v3); return b; }

#endif


#endif //__SIMD_H__
