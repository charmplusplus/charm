#ifndef __TRACEC_H__
#define __TRACEC_H__

#ifdef __cplusplus
extern "C" {
#endif

  extern void traceMalloc_c(void *where, int size, void **stack, int stackSize);
  extern void traceFree_c(void *where, int size);

#ifdef __cplusplus
}
#endif

#endif
