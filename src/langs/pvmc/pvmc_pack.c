#include <stddef.h>
#include "converse.h"
#include "pvmc.h"

#ifdef PVM_DEBUG
#define PARG(x)	PRINTF("Pe(%d) %s:%d %s(,%d,%d) called.\n",MYPE(),__FILE__,__LINE__,x,cnt,std);
#endif

int pvm_upkbyte(char *cp, int cnt, int std)
{
  int i, n_bytes;
  char *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkbyte");
#endif

  n_bytes = cnt * sizeof(char);

  buf = (char *)pvmc_getitem(n_bytes,PVM_BYTE);

  if (buf==(char *)NULL) {
    PRINTF("%s:%d pvm_upkbyte() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(cp, buf, n_bytes);
  } else {
    /* For characters, word alignment doesn't matter, so do C copies */
    for(i=0; i<cnt; i++)
      cp[i*std] = buf[i];
  }
  return 0;
}

int pvm_upkcplx(float *xp, int cnt, int std)
{
  int i, n_bytes;
  float *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkcplx");
#endif

  n_bytes = cnt * 2 * sizeof(float);

  buf = (float *)pvmc_getitem(n_bytes,PVM_CPLX);

  if (buf==(float *)NULL) {
    PRINTF("%s:%d pvm_upkcplx() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(xp, buf, n_bytes);
  } else {
    /* do a separate memcopy for each pair of elements.  Very ugly,
     * but otherwise, word alignment problems
     */
    for(i=0; i<cnt; i++)
      memcpy(xp+2*i*std,buf+2*i,2*sizeof(float));
  }
  return 0;
}

int pvm_upkdcplx(double *zp, int cnt, int std)
{
  int i, n_bytes;
  double *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkdcplx");
#endif

  n_bytes = cnt * 2 * sizeof(double);

  buf = (double *)pvmc_getitem(n_bytes,PVM_DCPLX);

  if (buf==(double *)NULL) {
    PRINTF("%s:%d pvm_upkdcplx() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(zp, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(zp+2*i*std,buf+2*i,2*sizeof(double));
  }
  return 0;
}

int pvm_upkdouble(double *dp, int cnt, int std)
{
  int i, n_bytes;
  double *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkdouble");
#endif

  n_bytes = cnt * sizeof(double);

  buf = (double *)pvmc_getitem(n_bytes,PVM_DOUBLE);

  if (buf==(double *)NULL) {
    PRINTF("%s:%d pvm_upkdouble() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(dp, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(dp+i*std,buf+i,sizeof(double));
  }

  return 0;
}

int pvm_upkfloat(float *fp, int cnt, int std)
{
  int i, n_bytes;
  float *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkfloat");
#endif

  n_bytes = cnt * sizeof(float);

  buf = (float *)pvmc_getitem(n_bytes,PVM_FLOAT);

  if (buf==(float *)NULL) {
    PRINTF("%s:%d pvm_upkfloat() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(fp, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(fp+i*std,buf+i,sizeof(float));
  }
  return 0;
}

int pvm_upkint(int *np, int cnt, int std)
{
  int i, n_bytes;
  int *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkint");
#endif

  n_bytes = cnt * sizeof(int);

  buf = (int *)pvmc_getitem(n_bytes,PVM_INT);

  if (buf==(int *)NULL) {
    PRINTF("%s:%d pvm_upkint() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(np, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(np+i*std,buf+i,sizeof(int));
  }
  return 0;
}

int pvm_upkuint(unsigned int *np, int cnt, int std)
{
  int i, n_bytes;
  unsigned int *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkuint");
#endif

  n_bytes = cnt * sizeof(unsigned int);

  buf = (unsigned int *)pvmc_getitem(n_bytes,PVM_UINT);

  if (buf==(unsigned int *)NULL) {
    PRINTF("%s:%d pvm_upkuint() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(np, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(np+i*std,buf+i,sizeof(unsigned int));
  }
  return 0;
}

int pvm_upklong(long *np, int cnt, int std)
{
  int i, n_bytes;
  long *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upklong");
#endif

  n_bytes = cnt * sizeof(long);

  buf = (long *)pvmc_getitem(n_bytes,PVM_LONG);

  if (buf==(long *)NULL) {
    PRINTF("%s:%d pvm_upklong() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(np, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(np+i*std,buf+i,sizeof(long));
  }
  return 0;
}

int pvm_upkulong(unsigned long *np, int cnt, int std)
{
  int i, n_bytes;
  unsigned long *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkulong");
#endif

  n_bytes = cnt * sizeof(unsigned long);

  buf = (unsigned long *)pvmc_getitem(n_bytes,PVM_ULONG);

  if (buf==(unsigned long *)NULL) {
    PRINTF("%s:%d pvm_upkulong() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(np, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(np+i*std,buf+i,sizeof(unsigned long));
  }
  return 0;
}

int pvm_upkshort(short *np, int cnt, int std)
{
  int i, n_bytes;
  short *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkshort");
#endif

  n_bytes = cnt * sizeof(short);

  buf = (short *)pvmc_getitem(n_bytes,PVM_SHORT);

  if (buf==(short *)NULL) {
    PRINTF("%s:%d pvm_upkshort() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(np, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(np+i*std,buf+i,sizeof(short));
  }
  return 0;
}

int pvm_upkushort(unsigned short *np, int cnt, int std)
{
  int i, n_bytes;
  unsigned short *buf;

#ifdef PVM_DEBUG
  PARG("pvm_upkushort");
#endif

  n_bytes = cnt * sizeof(unsigned short);

  buf = (unsigned short *)pvmc_getitem(n_bytes,PVM_USHORT);

  if (buf==(unsigned short *)NULL) {
    PRINTF("%s:%d pvm_upkushort() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (std==1) {
    memcpy(np, buf, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      memcpy(np+i*std,buf+i,sizeof(unsigned short));
  }
  return 0;
}

int pvm_upkstr(char *cp)
{
  int i, n_bytes;
  short *buf;

#ifdef PVM_DEBUG
  PRINTF("%s:%d %s() called.",__FILE__,__LINE__,"pvm_upkstr");
#endif

  buf = (short *)pvmc_getstritem(&n_bytes);

  if (buf==(short *)NULL) {
    PRINTF("%s:%d pvm_upkshort() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  memcpy(cp, buf, n_bytes);
  cp[n_bytes]='\0';

  return 0;
}

/**********************************************************************/

int pvm_pkbyte(char *cp, int cnt, int std)
{
  int i, n_bytes;
  char *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkbyte");
#endif

  n_bytes = cnt * sizeof(char);

  buf = (char *)pvmc_mkitem(n_bytes,PVM_BYTE);

  if (buf==(char *)NULL) {
    PRINTF("%s:%d pvm_pkbyte() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*cp;
  else if (std==1) {
    memcpy(buf, cp, n_bytes);
  } else {
    for(i=0; i<cnt; i++)
      buf[i]=cp[i*std];
  }
  return 0;
}

int pvm_pkcplx(float *xp, int cnt, int std)
{
  int i, n_bytes;
  float *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkcplx");
#endif

  n_bytes = cnt*2*sizeof(float);

  buf = (float *)pvmc_mkitem(n_bytes,PVM_CPLX);

  if (buf==(float *)NULL) {
    PRINTF("%s:%d pvm_pkcplx() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*xp;
  else if (std==1) {
    memcpy(buf, xp, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[2*i]=xp[i*std];
      buf[2*i+1]=xp[i*std+1];
    }
  }
  return 0;
}

int pvm_pkdcplx(double *zp, int cnt, int std)
{
  int i, n_bytes;
  double *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkdcplx");
#endif

  n_bytes = cnt*2*sizeof(double);

  buf = (double *)pvmc_mkitem(n_bytes,PVM_DCPLX);
  if (buf==(double *)NULL) {
    PRINTF("%s:%d pvm_pkdcplx() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*zp;
  else if (std==1) {
    memcpy(buf, zp, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[2*i]=zp[i*std];
      buf[2*i+1]=zp[i*std+1];
    }
  }
  return 0;
}

int pvm_pkdouble(double *dp, int cnt, int std)
{
  int i, n_bytes;
  double *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkdouble");
#endif

  n_bytes = cnt*sizeof(double);

  buf = (double *)pvmc_mkitem(n_bytes,PVM_DOUBLE);
  if (buf==(double *)NULL) {
    PRINTF("%s:%d pvm_pkdouble() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*dp;
  else if (std==1) {
    memcpy(buf, dp, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[i]=dp[i*std];
    }
  }
  return 0;
}

int pvm_pkfloat(float *fp, int cnt, int std)
{
  int i, n_bytes;
  float *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkfloat");
#endif

  n_bytes = cnt*sizeof(float);

  buf = (float *)pvmc_mkitem(n_bytes,PVM_FLOAT);
  if (buf==(float *)NULL) {
    PRINTF("%s:%d pvm_pkfloat() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*fp;
  else if (std==1) {
    memcpy(buf, fp, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[i]=fp[i*std];
    }
  }
  return 0;
}

int pvm_pkint(int *np, int cnt, int std)
{
  int i, n_bytes;
  int *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkint");
#endif

  n_bytes = cnt*sizeof(int);

  buf = (int *)pvmc_mkitem(n_bytes,PVM_INT);
  if (buf==(int *)NULL) {
    PRINTF("%s:%d pvm_pkint() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*np;
  else if (std==1) {
    memcpy(buf, np, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[i]=np[i*std];
    }
  }
  return 0;
}

int pvm_pkuint(unsigned int *np, int cnt, int std)
{
  int i, n_bytes;
  unsigned int *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkuint");
#endif

  n_bytes = cnt*sizeof(unsigned int);

  buf = (unsigned int *)pvmc_mkitem(n_bytes,PVM_UINT);
  if (buf==(unsigned int *)NULL) {
    PRINTF("%s:%d pvm_pkuint() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*np;
  else if (std==1) {
    memcpy(buf, np, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[i]=np[i*std];
    }
  }
  return 0;
}

int pvm_pklong(long *np, int cnt, int std)
{
  int i, n_bytes;
  long *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pklong");
#endif

  n_bytes = cnt*sizeof(long);

  buf = (long *)pvmc_mkitem(n_bytes,PVM_LONG);
  if (buf==(long *)NULL) {
    PRINTF("%s:%d pvm_pklong() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*np;
  else if (std==1) {
    memcpy(buf, np, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[i]=np[i*std];
    }
  }
  return 0;
}

int pvm_pkulong(unsigned long *np, int cnt, int std)
{
  int i, n_bytes;
  unsigned long *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkulong");
#endif

  n_bytes = cnt*sizeof(unsigned long);

  buf = (unsigned long *)pvmc_mkitem(n_bytes,PVM_ULONG);
  if (buf==(unsigned long *)NULL) {
    PRINTF("%s:%d pvm_pkulong() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*np;
  else if (std==1) {
    memcpy(buf, np, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[i]=np[i*std];
    }
  }
  return 0;
}

int pvm_pkshort(short *np, int cnt, int std)
{
  int i, n_bytes;
  short *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkshort");
#endif

  n_bytes = cnt*sizeof(short);

  buf = (short *)pvmc_mkitem(n_bytes,PVM_SHORT);
  if (buf==(short *)NULL) {
    PRINTF("%s:%d pvm_pkshort() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*np;
  else if (std==1) {
    memcpy(buf, np, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[i]=np[i*std];
    }
  }
  return 0;
}

int pvm_pkushort(unsigned short *np, int cnt, int std)
{
  int i, n_bytes;
  unsigned short *buf;

#ifdef PVM_DEBUG
  PARG("pvm_pkushort");
#endif

  n_bytes = cnt*sizeof(unsigned short);

  buf = (unsigned short *)pvmc_mkitem(n_bytes,PVM_USHORT);
  if (buf==(unsigned short *)NULL) {
    PRINTF("%s:%d pvm_pkushort() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  if (cnt==1)
    *buf=*np;
  else if (std==1) {
    memcpy(buf, np, n_bytes);
  } else {
    for(i=0; i<cnt; i++) {
      buf[i]=np[i*std];
    }
  }
  return 0;
}

int pvm_pkstr(char *cp)
{
  int i, n_bytes;
  char *buf;

#ifdef PVM_DEBUG
  PRINTF("%s:%d %s() called.",__FILE__,__LINE__,"pvm_pkstr");
#endif

  n_bytes = strlen(cp);

  buf = (char *)pvmc_mkitem(n_bytes,PVM_STR);
  if (buf==(char *)NULL) {
    PRINTF("%s:%d pvm_pkstr() no data mem\n",
	   __FILE__,__LINE__);
    return -1;
  }

  memcpy(buf, cp, n_bytes);

  return 0;
}

