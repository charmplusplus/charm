/*Charm++ Finite Element Framework:
Orion Sky Lawlor, olawlor@acm.org, 12/20/2002

Implementation file for FEM: FEM user data types.
These allow FEM data to be scooped right out of a user's
data structure.
*/
#ifndef _FEM_DATATYPE_H
#define _FEM_DATATYPE_H
#include "fem.h"

/**
 * This class represents the layout of a user data structure in user memory.
 *  It's used to read and write values from the user data structure.
 */
// This should probably be replaced by the MPI-like DDT library.
struct DType {
  int base_type; //FEM_* datatype
  int vec_len; //Number of items of this datatype
  int init_offset; // offset of field in bytes from the beginning of data
  int fdistance; // distance in bytes between successive records
  int idistance; // distance in bytes between successive items in a record
  DType(void) {}
  DType( const int b,  const int v=1,  const int i=0,  const int fd=0, const int id=0)
    : base_type(b), vec_len(v), init_offset(i), fdistance(fd), idistance(id)
  {
    if (fdistance==0) fdistance=length();
    if (idistance==0) idistance=type_size(base_type);
  }
  //Default copy constructor, assignment operator

  /// Return the total number of bytes required by this FEM_* data type
  static int type_size(int dataType);
  
  ///Return a human-readable string describing this FEM_* data type
  static const char *type_name(int dataType);
  
  /// Return the total number of bytes required by the 
  /// compressed form of this DType
  int length(const int nitems=1) const {
    return type_size(base_type) * vec_len * nitems;
  }
  
  /**
   * For each record in nodes[0..nNodes-1], copy the
   * user data in v_in into the compressed data in v_out.
   */
  void gather(int nNodes,const int *nodes,
              const void *v_in,void *v_out) const;

  /**
   * For each record in nodes[0..nNodes-1], copy the
   * compressed data from v_in into the user data in v_out.
   */
  void scatter(int nNodes,const int *nodes,
               const void *v_in,void *v_out) const;
  
  /**
   * For each record in nodes[0..nNodes-1], add the
   * compressed data from v_in into the user data in v_out.
   */
  void scatteradd(int nNodes,const int *nodes,
                  const void *v_in,void *v_out) const;
};

/**
 * Reduction support: Initialize the compressed data in lhs with
 * the appropriate initial value for the reduction op.
 */
void reduction_initialize(const DType& dt, void *lhs, int op);

/**
 * Reduction support: get a function pointer that can be used to
 * combine data of this type using this operation.
 */
typedef void (*reduction_combine_fn)(const int len,void *lhs,const void *rhs);
reduction_combine_fn reduction_combine(const DType& dt, int op);

#endif

