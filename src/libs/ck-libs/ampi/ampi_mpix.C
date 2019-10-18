/*
 * MPIX: Semi-official extensions to the MPI standard.
 *
 * Documented in:
 * - http://wgropp.cs.illinois.edu/bib/papers/pdata/2007/latham_grequest-enhance-4.pdf
 */

#include "ampiimpl.h"

AMPI_API_IMPL(int, MPIX_Grequest_start, MPI_Grequest_query_function *query_fn,
  MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn,
  MPIX_Grequest_poll_function *poll_fn, void *extra_state, MPI_Request *request)
{
  AMPI_API("AMPIX_Grequest_start", query_fn, free_fn, cancel_fn, poll_fn, extra_state, request);

  ampi* ptr = getAmpiInstance(MPI_COMM_SELF); // All GReq's are posted to MPI_COMM_SELF
  GReq *newreq = new GReq(query_fn, free_fn, cancel_fn, poll_fn, extra_state);
  *request = ptr->postReq(newreq);

  return MPI_SUCCESS;
}


AMPI_API_IMPL(int, MPIX_Grequest_class_create, MPI_Grequest_query_function *query_fn,
  MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn,
  MPIX_Grequest_poll_function *poll_fn, MPIX_Grequest_wait_function *wait_fn,
  MPIX_Grequest_class *greq_class)
{
  AMPI_API("AMPIX_Grequest_class_create", query_fn, free_fn, cancel_fn, poll_fn, wait_fn, greq_class);

  greq_class_desc g;
  g.query_fn = query_fn;
  g.free_fn = free_fn;
  g.cancel_fn = cancel_fn;
  g.poll_fn = poll_fn;
  g.wait_fn = wait_fn;

  ampi* ptr = getAmpiInstance(MPI_COMM_SELF); // All GReq's are posted to MPI_COMM_SELF

  ptr->greq_classes.push_back(g);

  *greq_class = ptr->greq_classes.size()-1;

  return MPI_SUCCESS;
}

AMPI_API_IMPL(int, MPIX_Grequest_class_allocate, MPIX_Grequest_class greq_class,
  void *extra_state, MPI_Request *request)
{
  AMPI_API("AMPIX_Grequest_class_allocate", greq_class, extra_state, request);

  ampi* ptr = getAmpiInstance(MPI_COMM_SELF); // All GReq's are posted to MPI_COMM_SELF

  greq_class_desc g = ptr->greq_classes[greq_class];
  CkAssert(greq_class < ptr->greq_classes.size());

  GReq *newreq = new GReq(g.query_fn, g.free_fn, g.cancel_fn, g.poll_fn, g.wait_fn, extra_state);
  *request = ptr->postReq(newreq);

  return MPI_SUCCESS;
}
