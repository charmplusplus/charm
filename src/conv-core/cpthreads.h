
#ifndef CPTHREADS_H
#define CPTHREADS_H

#define pthread_once_t                  Cpthread_once_t
#define pthread_attr_t                  Cpthread_attr_t
#define pthread_key_t                   Cpthread_key_t
#define pthread_cleanup_t               Cpthread_cleanup_t
#define pthread_mutexattr_t             Cpthread_mutexattr_t
#define pthread_condattr_t              Cpthread_condattr_t
#define pthread_mutex_t                 Cpthread_mutex_t
#define pthread_cond_t                  Cpthread_cond_t
#define pthread_t                       Cpthread_t
#define pthread_attr_init               Cpthread_attr_init
#define pthread_attr_destroy            Cpthread_attr_destroy
#define pthread_attr_getstacksize       Cpthread_attr_getstacksize
#define pthread_attr_setstacksize       Cpthread_attr_setstacksize
#define pthread_attr_getdetachstate     Cpthread_attr_getdetachstate
#define pthread_attr_setdetachstate     Cpthread_attr_setdetachstate
#define pthread_key_create              Cpthread_key_create
#define pthread_key_delete              Cpthread_key_delete
#define pthread_setspecific             Cpthread_setspecific
#define pthread_getspecific             Cpthread_getspecific
#define pthread_cleanup_push            Cpthread_cleanup_push
#define pthread_cleanup_pop             Cpthread_cleanup_pop
#define pthread_exit                    Cpthread_exit
#define pthread_top                     Cpthread_top
#define pthread_create                  Cpthread_create
#define pthread_equal                   Cpthread_equal
#define pthread_self                    Cpthread_self
#define pthread_detach                  Cpthread_detach
#define pthread_join                    Cpthread_join
#define pthread_mutexattr_init          Cpthread_mutexattr_init
#define pthread_mutexattr_destroy       Cpthread_mutexattr_destroy
#define pthread_mutexattr_getpshared    Cpthread_mutexattr_getpshared
#define pthread_mutexattr_setpshared    Cpthread_mutexattr_setpshared
#define pthread_mutex_init              Cpthread_mutex_init
#define pthread_mutex_destroy           Cpthread_mutex_destroy
#define pthread_mutex_lock              Cpthread_mutex_lock
#define pthread_mutex_trylock           Cpthread_mutex_trylock
#define pthread_mutex_unlock            Cpthread_mutex_unlock
#define pthread_condattr_init           Cpthread_condattr_init
#define pthread_condattr_destroy        Cpthread_condattr_destroy
#define pthread_condattr_getpshared     Cpthread_condattr_getpshared
#define pthread_condattr_setpshared     Cpthread_condattr_setpshared
#define pthread_cond_init               Cpthread_cond_init
#define pthread_cond_destroy            Cpthread_cond_destroy
#define pthread_cond_wait               Cpthread_cond_wait
#define pthread_cond_signal             Cpthread_cond_signal
#define pthread_cond_broadcast          Cpthread_cond_broadcast
#define pthread_once                    Cpthread_once

#endif /* CPTHREAD_H */

