

#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "converse.h"

#if CMK_HAS_PTHREAD_SETAFFINITY
int set_pthread_affinity(int cpuid)
{
#if CMK_SMP
    int s, j;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();

    CPU_ZERO(&cpuset);
    CPU_SET(cpuid, &cpuset);

    s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) {
      perror("pthread_setaffinity");
      return -1;
    }
#endif
    return 0;
#if 0
    /* old implementation */
  SET_MASK(cpuid)
  /* PID 0 refers to the current process */
  if (pthread_setaffinity_np(pthread_self(), len, &mask) < 0) {
    perror("pthread_setaffinity");
    return -1;
  }
#endif
}

int print_pthread_affinity() {
#if CMK_SMP
    int s, j;
    cpu_set_t cpuset;
    pthread_t thread;
    char str[256], pe[16];

    thread = pthread_self();
    s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) {
      perror("pthread_getaffinity");
      return -1;
    }

    sprintf(str, "[%d] %s affinity is: ", CmiMyPe(), CmiMyPe()>=CmiNumPes()?"communication pthread":"pthread");
    for (j = 0; j < CPU_SETSIZE; j++)
        if (CPU_ISSET(j, &cpuset)) {
            sprintf(pe, " %d", j);
            strcat(str, pe);
        }
    CmiPrintf("%s\n", str);
#endif
  return 0;
#if 0
  if (pthread_getaffinity_np(pthread_self(), len, &mask) < 0) {
    perror("pthread_setaffinity");
    return -1;
  }
  CmiPrintf("[%d] %s affinity mask is: 0x%08lx\n", CmiMyPe(), CmiMyPe()>=CmiNumPes()?"communication pthread":"pthread", mask);
#endif
}

#if 0
int which_core()
{
int apic_id,a,b;

   a = 1;
#if 1
   asm ( "mov %1, %%eax; " // a into eax
          "cpuid;"
          "mov %%eax, %0;" // eeax into b
          :"=r"(b) /* output */
          :"r"(a) /* input */
          :"%eax" /* clobbered register */
         );
#else
   asm ( "mov $1, %%eax; " // a into eax
          "cpuid;"
          "mov %%eax, %0;" // eeax into b
          :"=r"(b) /* output */
          : /* input */
          :"%eax" /* clobbered register */
         );
#endif

/*  windows
__asm__   
(
mov     eax, 1;
cpuid;
shr     ebx, 24;
mov     apic_id, ebx;
}
*/
   apic_id = (b >> 24 );
  return apic_id;
}
#endif

#endif
