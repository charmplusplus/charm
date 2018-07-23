/*
 * MPIX: Semi-official extensions to the MPI standard.
 *
 * Documented in:
 * - http://wgropp.cs.illinois.edu/bib/papers/pdata/2007/latham_grequest-enhance-4.pdf
 */

#include "ampiimpl.h"

AMPI_API_IMPL(int, MPIX_Grequest_start, MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn, MPIX_Grequest_poll_function *poll_fn, void *extra_state, MPI_Request *request)
{
  AMPI_API("AMPIX_Grequest_start");

  ampi* ptr = getAmpiInstance(MPI_COMM_SELF); // All GReq's are posted to MPI_COMM_SELF
  GReq *newreq = new GReq(query_fn, free_fn, cancel_fn, poll_fn, extra_state);
  *request = ptr->postReq(newreq);

  return MPI_SUCCESS;
}
