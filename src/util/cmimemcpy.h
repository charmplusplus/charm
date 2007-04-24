
#ifndef __CMI_MEMCPY_H__
#define __CMI_MEMCPY_H__

/**************************************************************
*  Optimized version of memory copy designed for Blue Gene/L  *
*  It inlines short memory copy operations and optimizes      *
*  8 byte aligned copies larger than 128 bytes                *
*                  - Sameer (04/07)                           *
*  It has two functions :                                     *
*  inlined opt_memcopy and __dcopy128 (dcopy.h)               *
***************************************************************/

#include <assert.h>
#include <stdio.h>

static inline void *_bcopy( void *dest, const void *src, size_t bytes )
{
  const unsigned char *r1 = (const unsigned char *)src;
  unsigned char *r2 = (unsigned char *)dest;
  size_t b4 = bytes >> 2;
  size_t remainder = bytes & 3;
  
  while ( b4-- ) {
    unsigned char u1 = *(r1+0);
    unsigned char u2 = *(r1+1);
    unsigned char u3 = *(r1+2);
    unsigned char u4 = *(r1+3);
    *(r2+0) = u1;
    *(r2+1) = u2;
    *(r2+2) = u3;
    *(r2+3) = u4;
    r1 += 4;
    r2 += 4;
  }
  
  while( remainder -- )
    *r2++ = *r1++;
  
  return( dest );
}

static inline void *_wcopy ( void *dest, const void *src , size_t bytes )
{
  const unsigned *r1 = (const unsigned *)src;
  unsigned *r2 = (unsigned *)dest;
  size_t nw = bytes >> 3;
  size_t remainder  = bytes & 0x7;
  
  while ( nw -- ) {
    unsigned u1 = *(r1+0);
    unsigned u2 = *(r1+1);
    *(r2+0) = u1;
    *(r2+1) = u2;
    r1 += 2;
    r2 += 2;
  }
  
  if ( remainder )
    _bcopy ( r2, r1, remainder );
  
  return( dest );
}

void *_dcopy128 ( void * dest, const void *src, size_t n );

static inline void *opt_memcopy ( void * dest, const void *src, size_t n ) {
  unsigned long daddr = (unsigned long) dest;
  unsigned long saddr = (unsigned long) src;
  
  if ( (n >= 128) &&  (  ((daddr & 0x07) == 0) && ((saddr & 0x07) == 0)  ))
    return _dcopy128 (dest, src, n);
  else if ( ((daddr & 0x03) == 0) && ((saddr & 0x03) == 0) ) {
    return _wcopy (dest, src, n);
  }
  else {
    return _bcopy (dest, src, n);
  }
  
  return 0;
}

#endif
