/**************************************************
 CCS Authentication utility routines 

Orion Sky Lawlor, olawlor@acm.org, 7/23/2001
*/
#ifndef __CCS_AUTH_H
#define __CCS_AUTH_H

#ifdef __cplusplus
extern "C" {
#endif

/*A secret key, used to authenticate a client or server.
This could be human-readable text, a random one-time pad,
some shared common knowledge, or any combination.
*/
typedef struct {
	unsigned char data[16];
} CcsSec_secretKey;
int CCS_AUTH_makeSecretKey(const char *str,CcsSec_secretKey *key);


/*The output of a SHA-1 hash algorithm*/
typedef struct {
	unsigned char data[20];
} SHA1_hash_t;

void CCS_AUTH_hash(const CcsSec_secretKey *key,unsigned int salt,
		   const CcsMessageHeader *hdrOrNull,SHA1_hash_t *out);
int  CCS_AUTH_differ(const CcsSec_secretKey *key,unsigned int salt,
		   const CcsMessageHeader *hdrOrNull,SHA1_hash_t *given);


/*Strong (but rather slow) random stream*/
typedef struct {
  unsigned char state[64]; /*Random number stream state*/
} CCS_RAND_state;

void CCS_RAND_new(CCS_RAND_state *s);
unsigned int CCS_RAND_next(CCS_RAND_state *s);

#ifdef __cplusplus
};
#endif

#endif /* def(thisHeader) */

