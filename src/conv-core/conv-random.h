#ifdef _CONV_RANDOM_H
#define _CONV_RANDOM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rngen_
{
  unsigned int prime;
  double state[3], multiplier[3];/* simulate 64 bit arithmetic */
} CrnStream;

void   CrnInitStream(CrnStream *, int, int);
int    CrnInt(CrnStream *);
double CrnDouble(CrnStream *);
float  CrnFloat(CrnStream *);
void   CrnSrand(int);
int    CrnRand(void);
double CrnDrand(void);

#ifdef __cplusplus
}
#endif

#endif
