/*
 * campi (coroutine AMPI): application header. Requires C++20.
 *
 * Usage:
 *   #include "mpi.h"
 *   #include "campi.h"
 *   CAMPI_MAIN(argc, argv) {
 *     MPI_Init(&argc, &argv);          // expands to co_await campi::Init(...)
 *     ...
 *     MPI_Recv(buf, n, MPI_INT, 0, 0, MPI_COMM_WORLD, &sts);   // co_await
 *     ...
 *     MPI_Finalize();
 *     co_return 0;
 *   }
 *
 * The blocking MPI names below are redefined as co_await expressions on the
 * campi awaitables (campi_core.h). Using one of them in a function that is
 * not a coroutine is a compile error: any function that (transitively) calls
 * a blocking MPI routine must itself return campi::task<T> and be awaited.
 * Blocking routines campi does not provide yet expand to an undeclared
 * identifier naming the routine, so they fail to compile instead of
 * aborting at run time. Nonblocking and local routines are AMPI's own.
 */
#ifndef CAMPI_H
#define CAMPI_H

#include "campi_core.h"

#define CAMPI_MAIN(argc_, argv_)                                              \
  ::campi::task<int> campi_main(int argc_, char **argv_);                     \
  static ::campi::main_registrar campi_main_registrar_(&campi_main);          \
  ::campi::task<int> campi_main(int argc_, char **argv_)

#define CAMPI_AWAIT(expr) (co_await (expr))

#undef MPI_Init
#undef MPI_Init_thread
#undef MPI_Finalize
#undef MPI_Send
#undef MPI_Ssend
#undef MPI_Recv
#undef MPI_Sendrecv
#undef MPI_Wait
#undef MPI_Waitall
#undef MPI_Barrier
#undef MPI_Bcast
#undef MPI_Reduce
#undef MPI_Allreduce
#undef MPI_Comm_split
#undef MPI_Comm_create
#undef AMPI_Migrate

#define MPI_Init(...)        CAMPI_AWAIT(::campi::Init(__VA_ARGS__))
#define MPI_Init_thread(...) CAMPI_AWAIT(::campi::Init_thread(__VA_ARGS__))
#define MPI_Finalize()       CAMPI_AWAIT(::campi::Finalize())
#define MPI_Send(...)        CAMPI_AWAIT(::campi::Send(__VA_ARGS__))
#define MPI_Ssend(...)       CAMPI_AWAIT(::campi::Ssend(__VA_ARGS__))
#define MPI_Recv(...)        CAMPI_AWAIT(::campi::Recv(__VA_ARGS__))
#define MPI_Sendrecv(...)    CAMPI_AWAIT(::campi::Sendrecv(__VA_ARGS__))
#define MPI_Wait(...)        CAMPI_AWAIT(::campi::Wait(__VA_ARGS__))
#define MPI_Waitall(...)     CAMPI_AWAIT(::campi::Waitall(__VA_ARGS__))
#define MPI_Barrier(...)     CAMPI_AWAIT(::campi::Barrier(__VA_ARGS__))
#define MPI_Bcast(...)       CAMPI_AWAIT(::campi::Bcast(__VA_ARGS__))
#define MPI_Reduce(...)      CAMPI_AWAIT(::campi::Reduce(__VA_ARGS__))
#define MPI_Allreduce(...)   CAMPI_AWAIT(::campi::Allreduce(__VA_ARGS__))
#define MPI_Comm_split(...)  CAMPI_AWAIT(::campi::Comm_split(__VA_ARGS__))
#define MPI_Comm_create(...) CAMPI_AWAIT(::campi::Comm_create(__VA_ARGS__))
#define AMPI_Migrate(...)    CAMPI_AWAIT(::campi::Migrate(__VA_ARGS__))

/* Blocking routines without a campi awaitable yet: a use fails to compile. */
#undef MPI_Probe
#undef MPI_Mprobe
#undef MPI_Mrecv
#undef MPI_Waitany
#undef MPI_Waitsome
#undef MPI_Bsend
#undef MPI_Rsend
#undef MPI_Sendrecv_replace
#undef MPI_Reduce_scatter
#undef MPI_Reduce_scatter_block
#undef MPI_Alltoallw
#undef MPI_Intercomm_create
#undef MPI_Intercomm_merge
#undef MPI_Cart_create
#undef MPI_Cart_sub
#undef MPI_Graph_create
#undef MPI_Dist_graph_create
#undef MPI_Dist_graph_create_adjacent
#undef MPI_Comm_split_type
#undef AMPI_Yield
#undef AMPI_Suspend
#define MPI_Probe(...)                      campi_unsupported_blocking_call_MPI_Probe
#define MPI_Mprobe(...)                     campi_unsupported_blocking_call_MPI_Mprobe
#define MPI_Mrecv(...)                      campi_unsupported_blocking_call_MPI_Mrecv
#define MPI_Waitany(...)                    campi_unsupported_blocking_call_MPI_Waitany
#define MPI_Waitsome(...)                   campi_unsupported_blocking_call_MPI_Waitsome
#define MPI_Bsend(...)                      campi_unsupported_blocking_call_MPI_Bsend
#define MPI_Rsend(...)                      campi_unsupported_blocking_call_MPI_Rsend
#define MPI_Sendrecv_replace(...)           campi_unsupported_blocking_call_MPI_Sendrecv_replace
#define MPI_Reduce_scatter(...)             campi_unsupported_blocking_call_MPI_Reduce_scatter
#define MPI_Reduce_scatter_block(...)       campi_unsupported_blocking_call_MPI_Reduce_scatter_block
#define MPI_Alltoallw(...)                  campi_unsupported_blocking_call_MPI_Alltoallw
#define MPI_Intercomm_create(...)           campi_unsupported_blocking_call_MPI_Intercomm_create
#define MPI_Intercomm_merge(...)            campi_unsupported_blocking_call_MPI_Intercomm_merge
#define MPI_Cart_create(...)                campi_unsupported_blocking_call_MPI_Cart_create
#define MPI_Cart_sub(...)                   campi_unsupported_blocking_call_MPI_Cart_sub
#define MPI_Graph_create(...)               campi_unsupported_blocking_call_MPI_Graph_create
#define MPI_Dist_graph_create(...)          campi_unsupported_blocking_call_MPI_Dist_graph_create
#define MPI_Dist_graph_create_adjacent(...) campi_unsupported_blocking_call_MPI_Dist_graph_create_adjacent
#define MPI_Comm_split_type(...)            campi_unsupported_blocking_call_MPI_Comm_split_type
#define AMPI_Yield(...)                     CAMPI_AWAIT(::campi::yield())
#define AMPI_Suspend(...)                   campi_unsupported_blocking_call_AMPI_Suspend

#endif // CAMPI_H
