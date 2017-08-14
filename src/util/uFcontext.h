#ifndef CMK_FCONTEXT_H
#define CMK_FCONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif
  typedef void * fcontext_t;

  typedef struct uFcontext_stack_t {
    void *ss_sp;
    int ss_flags;
    size_t ss_size;
  } uFcontext_stack_t;

  typedef struct transfer_t {
    fcontext_t fctx;
    void * data;
  } transfer_t;

  extern void CthStartThread(transfer_t);
  typedef void (*uFcontext_fn_t)(transfer_t);

  typedef struct data_t {
    void * from;
    void * data;
  } data_t;

  typedef struct uFcontext_t {
    fcontext_t fctx;
    void (* func)(void *);
    uFcontext_stack_t uc_stack;
    struct uFcontext_t *uc_link;
    void *arg;
    data_t param;
  } uFcontext_t;

  transfer_t jump_fcontext(fcontext_t const to, void *vp);
  fcontext_t make_fcontext(void *sp, size_t size, void (*fn)(transfer_t));
  transfer_t ontop_fcontext(fcontext_t const to, void *vp, transfer_t (*fn)(transfer_t));

/* Get user context and store it in variable pointed to by UCP.  */
  extern int getJcontext (uFcontext_t *__ucp);

/* Set user context from information of variable pointed to by UCP.  */
  extern int setJcontext (uFcontext_t *__ucp);

/* Save current context in context variable pointed to by OUCP and set
   context from variable pointed to by UCP.  */
  extern int swapJcontext (uFcontext_t *__oucp, uFcontext_t *__ucp);

  extern void makeJcontext(uFcontext_t *__ucp, uFcontext_fn_t, void (*fn)(void*), void *arg);
/* To keep the interface of uFcontext the same as the ucontext and uJcontext*/
  int getJcontext (uFcontext_t *__ucp) {
    return 0;
  }

  int setJcontext (uFcontext_t *__ucp) {
    return swapJcontext(NULL, __ucp);
  }

  void makeJcontext (uFcontext_t *__ucp, uFcontext_fn_t __func, void (*fn)(void *), void *arg) {
    __ucp->arg = arg;
    __ucp->uc_link = NULL;
    __ucp->func = fn;
    fcontext_t t = make_fcontext(__ucp->uc_stack.ss_sp, __ucp->uc_stack.ss_size, __func);
    __ucp->fctx = t;
  }

  int swapJcontext(uFcontext_t *old_ucp,  uFcontext_t *new_ucp) {
     new_ucp->param.from = old_ucp;
     new_ucp->param.data = new_ucp;
     transfer_t t = jump_fcontext(new_ucp->fctx, &(new_ucp->param));
     data_t *old_data = (data_t *)t.data;
     uFcontext_t *prev_ucp = (uFcontext_t *)old_data->from;
     if (prev_ucp)
       prev_ucp->fctx = t.fctx;
     return 0;
  }
#ifdef __cplusplus
}
#endif
#endif
