
#include "qt.h"
#include <setjmp.h>

#ifdef ALLOCA_H
#include <alloca.h>
#endif

struct helpdesc { qt_helper_t *hfn; qt_t *jb; void *oldptr; void *newptr; };

#ifdef __CYGWIN__
# ifdef QT_GROW_DOWN
#define SHIFTSP(pos) {char *newpos=(char *)pos-256; asm ( "mov %0, %%esp\n"::"m"(newpos));}
# else
#define SHIFTSP(pos) {char *newpos=(char *)pos+256; asm ( "mov %0, %%esp\n"::"m"(newpos));}
# endif
#else
# ifdef QT_GROW_DOWN
#define SHIFTSP(pos) {char *osp=alloca(0); alloca((osp-((char*)pos))+256); }
# else
#define SHIFTSP(pos) {char *osp=alloca(0); alloca((((char*)pos)-osp)+256); }
# endif
#endif

static void qt_args_1(qt_t *rjb, void *u, void *t,
		      qt_userf_t *userf, qt_only_t *only)
{
  jmp_buf jb; struct helpdesc *rhelp;
  rhelp = (struct helpdesc *)setjmp(jb);
  if (rhelp == 0) {
    SHIFTSP(rjb);
    longjmp(*(jmp_buf *)&rjb, (int)jb);
  }
  rhelp->hfn(rhelp->jb, rhelp->oldptr, rhelp->newptr);
  only(u, t, userf);
  write(2,"Never get here 2.\n",18);
}

qt_t *qt_args(qt_t *sp, void *u, void *t,
	      qt_userf_t *userf, qt_only_t *only)
{
  jmp_buf jb; qt_t *result;

  result = (qt_t*)setjmp(jb);
  if (result==0) {
    SHIFTSP(sp);
    qt_args_1((qt_t*)jb,u,t,userf,only);
    write(2,"Never get here 1.\n",18);
  }
  return result;
}

void *qt_block(qt_helper_t *hfn, void *oldptr, void *newptr, qt_t *sp)
{
  struct helpdesc help, *rhelp; char *oldsp; int offs;
  jmp_buf jb;
  help.hfn = hfn;
  help.jb  = (qt_t*)&jb;
  help.oldptr = oldptr;
  help.newptr = newptr;
  rhelp = (struct helpdesc *)setjmp(jb);
  if (rhelp==0) {
    SHIFTSP(sp);
    longjmp(*(jmp_buf *)&sp, (int)&help);
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
  longjmp(*(jmp_buf *)&sp, (int)&help);
}
