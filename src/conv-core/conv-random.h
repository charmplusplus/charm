#ifndef _CONV_RANDOM_H
#define _CONV_RANDOM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rngen_
{
  unsigned int prime;
  double state[3], multiplier[3];/* simulate 64 bit arithmetic */
} CrnStream;

/*Type must be 0, 1, or 2.*/
void   CrnInitStream(CrnStream *dest, unsigned int seed, int type);
int    CrnInt(CrnStream *);
double CrnDouble(CrnStream *);
float  CrnFloat(CrnStream *);
void   CrnSrand(unsigned int);
int    CrnRand(void);
double CrnDrand(void);

#ifdef __cplusplus
}
#endif

#endif
