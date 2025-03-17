
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "qt.h"
#include <setjmp.h>
#ifdef ALLOCA_H
#include <alloca.h>
#endif

#include "conv-config.h"
#include "converse.h"

struct helpdesc { qt_helper_t *hfn; qt_t *jb; void *oldptr; void *newptr; };

#ifdef __CYGWIN__
# ifdef QT_GROW_DOWN
#define SHIFTSP(pos) asm ( "mov %0, %%esp\n"::"m"((char*)pos-256));
# else
#define SHIFTSP(pos) asm ( "mov %0, %%esp\n"::"m"((char*)pos+256));
# endif
#else
# ifdef QT_GROW_DOWN
#define SHIFTSP(pos) {char *osp = (char *)alloca(0); alloca((osp-((char*)pos))+256); }
# else
#define SHIFTSP(pos) {char *osp = (char *)alloca(0); alloca((((char*)pos)-osp)+256); }
# endif
#endif

#define MAXTABLE 1000

#if CMK_SMP && CMK_HAS_TLS_VARIABLES
#define TLS_SPECIFIER CMK_THREADLOCAL
#else
#define TLS_SPECIFIER
#endif

static TLS_SPECIFIER void * pbuf[MAXTABLE] = {0};
static TLS_SPECIFIER int    pcounter = 1;

#if __GNUC__ && !__clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

static int push_buf(void *ptr)
{
  int cur = pcounter;
  pbuf[pcounter] = ptr;
  pcounter++;
  if (pcounter >= MAXTABLE) pcounter = 1;   /* reuse the table */
  return cur;
}

static void qt_args_1(qt_t *rjb, void *u, void *t,
		      qt_userf_t *userf, qt_only_t *only)
{
  jmp_buf jb; struct helpdesc *rhelp;
  int index;
  index = setjmp(jb);
  rhelp = (struct helpdesc *)pbuf[index];
  if (rhelp == 0) {
    SHIFTSP(rjb);
    longjmp(*(jmp_buf *)&rjb, push_buf((void *)jb));
  }
  rhelp->hfn(rhelp->jb, rhelp->oldptr, rhelp->newptr);
  only(u, t, userf);
  (void)!write(2,"Never get here 2.\n",18);
}

qt_t *qt_args(qt_t *sp, void *u, void *t, qt_userf_t *userf, qt_only_t *only) CMI_NOOPTIMIZE;

qt_t *qt_args(qt_t *sp, void *u, void *t,
	      qt_userf_t *userf, qt_only_t *only)
{
  jmp_buf jb; qt_t *result;
  int index;
  index = setjmp(jb);
  result = (qt_t*)pbuf[index];
  if (result==0) {
    SHIFTSP(sp);
    qt_args_1((qt_t*)jb,u,t,userf,only);
    (void)!write(2,"Never get here 1.\n",18);
  }
  return result;
}

void *qt_block(qt_helper_t *hfn, void *oldptr, void *newptr, qt_t *sp)
{
  struct helpdesc help, *rhelp; char *oldsp; int offs;
  jmp_buf jb;
  int index;
  help.hfn = hfn;
  help.jb  = (qt_t*)&jb;
  help.oldptr = oldptr;
  help.newptr = newptr;
  index = setjmp(jb);
  rhelp = (struct helpdesc *)pbuf[index];
  if (rhelp==0) {
    SHIFTSP(sp);
    longjmp(*(jmp_buf *)&sp, push_buf((void *)&help));
  }
  rhelp->hfn(rhelp->jb, rhelp->oldptr, rhelp->newptr);
}

void *qt_abort(qt_helper_t *hfn, void *oldptr, void *newptr, qt_t *sp)
{
  struct helpdesc help, *rhelp;
  help.hfn = hfn;
  help.jb  = (qt_t*)&help;
  help.oldptr = oldptr;
  help.newptr = newptr;
  SHIFTSP(sp);
  longjmp(*(jmp_buf *)&sp, push_buf((void *)&help));
}

#if __GNUC__ && !__clang__
#pragma GCC diagnostic pop
#endif
