
#include "qt.h"
#include <setjmp.h>
#ifdef ALLOCA_H
#include <alloca.h>
#endif

#include "conv-config.h"

struct helpdesc { qt_helper_t *hfn; qt_t *jb; void *old; void *new; };

#ifdef __CYGWIN__
# ifdef QT_GROW_DOWN
#define SHIFTSP(pos) asm ( "mov %0, %%esp\n"::"m"((char*)pos-256));
# else
#define SHIFTSP(pos) asm ( "mov %0, %%esp\n"::"m"((char*)pos+256));
# endif
#else
# ifdef QT_GROW_DOWN
#define SHIFTSP(pos) {char *osp=alloca(0); alloca((osp-((char*)pos))+256); }
# else
#define SHIFTSP(pos) {char *osp=alloca(0); alloca((((char*)pos)-osp)+256); }
# endif
#endif

#define MAXTABLE 1000

#if CMK_SMP && CMK_HAS_TLS_VARIABLES
#define TLS_SPECIFIER         __thread
#else
#define TLS_SPECIFIER
#endif

static TLS_SPECIFIER void * pbuf[MAXTABLE] = {0};
static TLS_SPECIFIER int    pcounter = 1;


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
    longjmp((unsigned long*)rjb, push_buf((void *)jb));
  }
  rhelp->hfn(rhelp->jb, rhelp->old, rhelp->new);
  only(u, t, userf);
  write(2,"Never get here 2.\n",18);
}

qt_t *qt_args(qt_t *sp, void *u, void *t, qt_userf_t *userf, qt_only_t *only)  __attribute__((optimize(0)));

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
    write(2,"Never get here 1.\n",18);
  }
  return result;
}

void *qt_block(qt_helper_t *hfn, void *old, void *new, qt_t *sp)
{
  struct helpdesc help, *rhelp; char *oldsp; int offs;
  jmp_buf jb;
  int index;
  help.hfn = hfn;
  help.jb  = (qt_t*)&jb;
  help.old = old;
  help.new = new;
  index = setjmp(jb);
  rhelp = (struct helpdesc *)pbuf[index];
  if (rhelp==0) {
    SHIFTSP(sp);
    longjmp((unsigned long*)sp, push_buf((void *)&help));
  }
  rhelp->hfn(rhelp->jb, rhelp->old, rhelp->new);
}

void *qt_abort(qt_helper_t *hfn, void *old, void *new, qt_t *sp)
{
  struct helpdesc help, *rhelp;
  help.hfn = hfn;
  help.jb  = (qt_t*)&help;
  help.old = old;
  help.new = new;
  SHIFTSP(sp);
  longjmp((unsigned long*)sp, push_buf((void *)&help));
}
