/*******************************************************
Hashing and authentication routines for CCS.

Orion Sky Lawlor, olawlor@acm.org, 7/21/2001
*/

#include <string.h>
#include <stdio.h>
#include "sockRoutines.h"
#include "conv-ccs.h"
#include "ccs-auth.h"

/* Parse this secret key as a hex string*/
int CCS_AUTH_makeSecretKey(const char *str,CcsSec_secretKey *key)
{
  int i;
  memset(key->data,0,sizeof(CcsSec_secretKey));
  for (i=0;i<sizeof(key->data);i++) {
    int cur=0;
    char tmp[3];
    tmp[0]=str[2*i+0];
    tmp[1]=str[2*i+1];
    if (tmp[1]==0 || tmp[1]==' ' || tmp[1]=='\n') tmp[1]='0'; /*zero-pad*/
    tmp[2]=0;
    if (1!=sscanf(tmp,"%x",&cur)) break;
    key->data[i]=(unsigned char)cur;
  }
  if (i==0) return 0;
  else return 1;
}


/*************************************************************
Perform one round of the SHA-1 message hash.  Input is a set of
16 32-bit native words (512 bits); output is 5 32-bit native
words (160 bits). Because it uses native arithmetic, the 
implementation works equally well with 32 and 64-bit big-
and little-endian systems. However, when the input or output
is interpreted as bytes, they should be considered big-endian.
The speed is about 400,000 transformed blocks per second on a 
1 GHz machine.

Implemented and placed in the public domain by Steve Reid
Collected by Wei Dai (http://www.eskimo.com/~weidai/cryptlib.html)

Adapted for Charm++ by Orion Sky Lawlor, olawlor@acm.org, 7/20/2001
*/
/*Contains at least the low 32 bits of a big-endian integer.*/
typedef unsigned int word32;
typedef unsigned char byte8;

static void SHA1_init(word32 *state)
{
        state[0] = 0x67452301u;
        state[1] = 0xEFCDAB89u;
        state[2] = 0x98BADCFEu;
        state[3] = 0x10325476u;
        state[4] = 0xC3D2E1F0u;
}

static word32 rotlFixed(word32 x, word32 y)
{
#if defined(_MSC_VER) || defined(__BCPLUSPLUS__)
	return y ? _lrotl(x, y) : x;
#elif defined(__MWERKS__) && TARGET_CPU_PPC
	return y ? __rlwinm(x,y,0,31) : x;
#else /*Default C version*/
	return ((0xFFffFFffu)&(x<<y)) | (((0xFFffFFffu)&x)>>(32-y));
#endif
}

#define blk0(i) (W[i] = data[i])
#define blk1(i) (W[i&15] = rotlFixed(W[(i+13)&15]^W[(i+8)&15]^W[(i+2)&15]^W[i&15],1))

#define f1(x,y,z) (z^(x&(y^z)))
#define f2(x,y,z) (x^y^z)
#define f3(x,y,z) ((x&y)|(z&(x|y)))
#define f4(x,y,z) (x^y^z)

/* (R0+R1), R2, R3, R4 are the different operations used in SHA1 */
#define R0(v,w,x,y,z,i) z+=f1(w,x,y)+blk0(i)+0x5A827999u+rotlFixed(v,5);w=rotlFixed(w,30);
#define R1(v,w,x,y,z,i) z+=f1(w,x,y)+blk1(i)+0x5A827999u+rotlFixed(v,5);w=rotlFixed(w,30);
#define R2(v,w,x,y,z,i) z+=f2(w,x,y)+blk1(i)+0x6ED9EBA1u+rotlFixed(v,5);w=rotlFixed(w,30);
#define R3(v,w,x,y,z,i) z+=f3(w,x,y)+blk1(i)+0x8F1BBCDCu+rotlFixed(v,5);w=rotlFixed(w,30);
#define R4(v,w,x,y,z,i) z+=f4(w,x,y)+blk1(i)+0xCA62C1D6u+rotlFixed(v,5);w=rotlFixed(w,30);

static void SHA1_transform(word32 *state, const word32 *data)
{
    word32 W[16];
    /* Copy context->state[] to working vars */
    word32 a = state[0];
    word32 b = state[1];
    word32 c = state[2];
    word32 d = state[3];
    word32 e = state[4];
    /* 4 rounds of 20 operations each. Loop unrolled. */
    R0(a,b,c,d,e, 0); R0(e,a,b,c,d, 1); R0(d,e,a,b,c, 2); R0(c,d,e,a,b, 3);
    R0(b,c,d,e,a, 4); R0(a,b,c,d,e, 5); R0(e,a,b,c,d, 6); R0(d,e,a,b,c, 7);
    R0(c,d,e,a,b, 8); R0(b,c,d,e,a, 9); R0(a,b,c,d,e,10); R0(e,a,b,c,d,11);
    R0(d,e,a,b,c,12); R0(c,d,e,a,b,13); R0(b,c,d,e,a,14); R0(a,b,c,d,e,15);
    R1(e,a,b,c,d,16); R1(d,e,a,b,c,17); R1(c,d,e,a,b,18); R1(b,c,d,e,a,19);
    R2(a,b,c,d,e,20); R2(e,a,b,c,d,21); R2(d,e,a,b,c,22); R2(c,d,e,a,b,23);
    R2(b,c,d,e,a,24); R2(a,b,c,d,e,25); R2(e,a,b,c,d,26); R2(d,e,a,b,c,27);
    R2(c,d,e,a,b,28); R2(b,c,d,e,a,29); R2(a,b,c,d,e,30); R2(e,a,b,c,d,31);
    R2(d,e,a,b,c,32); R2(c,d,e,a,b,33); R2(b,c,d,e,a,34); R2(a,b,c,d,e,35);
    R2(e,a,b,c,d,36); R2(d,e,a,b,c,37); R2(c,d,e,a,b,38); R2(b,c,d,e,a,39);
    R3(a,b,c,d,e,40); R3(e,a,b,c,d,41); R3(d,e,a,b,c,42); R3(c,d,e,a,b,43);
    R3(b,c,d,e,a,44); R3(a,b,c,d,e,45); R3(e,a,b,c,d,46); R3(d,e,a,b,c,47);
    R3(c,d,e,a,b,48); R3(b,c,d,e,a,49); R3(a,b,c,d,e,50); R3(e,a,b,c,d,51);
    R3(d,e,a,b,c,52); R3(c,d,e,a,b,53); R3(b,c,d,e,a,54); R3(a,b,c,d,e,55);
    R3(e,a,b,c,d,56); R3(d,e,a,b,c,57); R3(c,d,e,a,b,58); R3(b,c,d,e,a,59);
    R4(a,b,c,d,e,60); R4(e,a,b,c,d,61); R4(d,e,a,b,c,62); R4(c,d,e,a,b,63);
    R4(b,c,d,e,a,64); R4(a,b,c,d,e,65); R4(e,a,b,c,d,66); R4(d,e,a,b,c,67);
    R4(c,d,e,a,b,68); R4(b,c,d,e,a,69); R4(a,b,c,d,e,70); R4(e,a,b,c,d,71);
    R4(d,e,a,b,c,72); R4(c,d,e,a,b,73); R4(b,c,d,e,a,74); R4(a,b,c,d,e,75);
    R4(e,a,b,c,d,76); R4(d,e,a,b,c,77); R4(c,d,e,a,b,78); R4(b,c,d,e,a,79);
    /* Add the working vars back into context.state[] */
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

/********************************************************
Compute the signed SHA-1 hash of this 52-byte input.
The 52 bytes comes about because a SHA block is 64 bytes,
minus an 8-byte length block and 4-byte end-of-message
code.  It is permissible for in to equal out.

The resulting hash code is 20 bytes long.
*/
static void SHA1_hash(const byte8 *in,SHA1_hash_t *out)
{
	int i;
#define SHA1_data_len 16 /*Length of input data (words)*/
#define SHA1_hash_len 5 /*Length of output hash code (words)*/
	word32 message[SHA1_data_len];
	word32 hash[SHA1_hash_len];
	
	/*Assemble the message from the user data by
	  interpreting the bytes as big-endian words.*/
	for (i=0;i<SHA1_data_len-3;i++)
		message[i]=(in[i*4+0]<<24)+(in[i*4+1]<<16)+
		           (in[i*4+2]<< 8)+(in[i*4+3]<< 0);
	/*Paste on the end-of-message and length fields*/
	message[13]=0x80000000u;/*End-of-message: one followed by zeros*/
	message[14]=0x00000000u;/*High word of message length (all zero)*/
	message[15]=512-64-32;/*Low word: message length, in bits*/
	
	/*Do the hash*/
	SHA1_init(hash);
	SHA1_transform(hash,message);
	
	/*Convert the result from words back to bytes*/
	for (i=0;i<SHA1_hash_len;i++) {
		out->data[i*4+0]=0xffu & (hash[i]>>24);
		out->data[i*4+1]=0xffu & (hash[i]>>16);
		out->data[i*4+2]=0xffu & (hash[i]>> 8);
		out->data[i*4+3]=0xffu & (hash[i]>> 0);
	}
}

#if SHA1_TEST_DRIVER
/* Tiny test driver routine-- should print out:
F9693B3AE7791C4ACE70CE31E4C2213F21CE900A
*/
int main(int argc,char *argv[])
{
	int i;
	SHA1_hash_t h;
	byte8 message[52]; memset(message,0,52);
	message[0]=0x01;
	SHA1_hash(message, &h);
	for (i=0;i<sizeof(h);i++) printf("%02X",h.data[i]);
	printf("\n");
}
#endif

/*Compare two hashed values-- return 1 if they differ; 0 else
 */
static int SHA1_differ(const SHA1_hash_t *a,const SHA1_hash_t *b)
{
  return 0!=memcmp(a->data,b->data,sizeof(SHA1_hash_t));
}

/********************************************************
Authentication routines: create a hash code; compare
hash codes.
*/

/*Create a hash code of this secret key, varying "salt" value,
and (optional, may be NULL) request header.
*/
void CCS_AUTH_hash(const CcsSec_secretKey *key,unsigned int salt,
		   const CcsMessageHeader *hdrOrNull,SHA1_hash_t *out)
{
  /*Fill the message buffer*/
	byte8 mess[64];
	byte8 *messCur=mess;

	memset(mess,0,64);
	memcpy(messCur,key,sizeof(CcsSec_secretKey));
	messCur+=sizeof(CcsSec_secretKey);
	
	*(ChMessageInt_t *)messCur=ChMessageInt_new(salt);
	messCur+=sizeof(ChMessageInt_t);
	
	if (hdrOrNull!=NULL) {
	  const int headerBytes=16; /*Only copy start of header.*/
	  memcpy(messCur,hdrOrNull,headerBytes);
	}
	
	SHA1_hash(mess,out);
}
	   
/*Create a hash code as above, and compare it to the given code.*/
int  CCS_AUTH_differ(const CcsSec_secretKey *key,unsigned int salt,
		   const CcsMessageHeader *hdrOrNull,SHA1_hash_t *given)
{
	SHA1_hash_t cur;
	CCS_AUTH_hash(key,salt,hdrOrNull,&cur);
	return SHA1_differ(&cur,given);
}

/********************************************************
Randomness routines: return a good 32-bit random number.
*/
#include <stdlib.h>

#if defined(_WIN32) && ! defined(__CYGWIN__)
#include <sys/timeb.h>
#else /*UNIX machine*/
#include <sys/time.h>
#include <fcntl.h>
#endif

void CCS_RAND_new(CCS_RAND_state *s)
{
  int i,randFD;
  static int newCount=0;
  byte8 tmp[sizeof(s->state)];

  /* State buffer starts out uninitialized. */
  /* XOR in a linear counter */
  s->state[32] ^= newCount++;

/*Fill the state buffer with random noise*/

#if defined(_WIN32) && ! defined(__CYGWIN__)
  _ftime((struct _timeb *)tmp);
  for (i=0;i<sizeof(s->state);i++)
    s->state[i]^=tmp[i];
#else /*UNIX machine*/
  /* XOR the current time of day into the state buffer*/
  gettimeofday((struct timeval *)tmp,NULL);
  for (i=0;i<sizeof(s->state);i++)
    s->state[i]^=tmp[i];

  /* XOR bytes from /dev/urandom into the state buffer*/
  randFD=open("/dev/urandom",O_RDONLY);
  if (randFD!=-1) {
    if (sizeof(s->state)==read(randFD,tmp,sizeof(s->state)))
      for (i=0;i<sizeof(s->state);i++)
	s->state[i]^=tmp[i];
    close(randFD);
  }
#endif
}

word32 CCS_RAND_next(CCS_RAND_state *s) {
  SHA1_hash_t ret;
  /*Stir the state*/
  (*(int *)(s->state))++;
  SHA1_hash(s->state,&ret);
  return *(word32 *)(&ret);
}












