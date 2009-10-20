/*Charm++ IDXL Library:
Orion Sky Lawlor, olawlor@acm.org, 12/20/2002

Implementation file for IDXL: IDXL user data layouts.
These allow IDXL data to be scooped right out of a user's
data structure.
*/
#ifndef _CHARM_IDXL_LAYOUT_H
#define _CHARM_IDXL_LAYOUT_H
#include "idxlc.h" /* for IDXL_ datatypes and IDXL_Layout_t */
#include "pup.h"

/**
 * This class represents the layout of a user data structure in user memory.
 *  It's used to read and write values from the user data structure.
 * 
 * This structure must be layout-compatible with a 5-integer array.
 */
// It's possible this should be replaced by the MPI-like DDT library.
class IDXL_Layout {
public:
  int type; ///< IDXL data type, like IDXL_INT or IDXL_DOUBLE
  int width; ///< Number of data fields per record
  int offset; ///< Bytes to jump from start of the array to first field
  int distance; ///< Bytes per record
  int skew; ///< Bytes between start of each field
  
  IDXL_Layout(void) { type=0; width=0; }
  IDXL_Layout( const int b,  const int v=1,  const int i=0,  const int fd=0, const int id=0)
  {
    type=b; width=v; offset=i; distance=fd; skew=id;
    if (distance==0) distance=compressedBytes();
    if (skew==0) skew=type_size(type);
  }
  //Default copy constructor, assignment operator
  void pup(PUP::er &p) {
    p|type; p|width; p|offset; p|distance; p|skew;
  }

  /// Return the total number of bytes required by this IDXL_* data type
  static int type_size(int dataType,const char *callingRoutine="");
  
  /// Return a human-readable string describing this IDXL_* data type
  static const char *type_name(int dataType,const char *callingRoutine="");
  
  /// Return the total number of bytes per user data record 
  inline int userBytes(void) const {
    return distance;
  }
  
  /// Return the total number of bytes per compressed record
  inline int compressedBytes(void) const {
    return type_size(type) * width;
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
void reduction_initialize(const IDXL_Layout& dt, void *lhs, int op,const char *callingRoutine="");

/**
 * Reduction support: get a function pointer that can be used to
 * combine a row of user data (src) into a row of compressed data (dest)
 * using this reduction operation.
 */
typedef void (*reduction_combine_fn)(unsigned char *dest,const unsigned char *src,const IDXL_Layout *srcLayout);
reduction_combine_fn reduction_combine(const IDXL_Layout& dt, int op,const char *callingRoutine="");

/// List the prototypes for parameters needed by the DEREF macro.
#define IDXL_LAYOUT_PARAM int width,int offset,int distance,int skew
/// Pass the parameters needed by the DEREF macro.
#define IDXL_LAYOUT_CALL(dt) (dt).width,(dt).offset,(dt).distance,(dt).skew
/// Find this record and field, of this type, in this user array.
/// Requires the paramters listed in IDX_LAYOUT_PARAM.
#define IDXL_LAYOUT_DEREF(T,src,record,field) \
	*(T *)(((unsigned char *)src)+offset+(record)*distance+(field)*skew)

/// Keeps a list of dynamically-allocated IDXL_Layout objects:
class IDXL_Layout_List {
  enum {FIRST_DT=IDXL_FIRST_IDXL_LAYOUT_T, MAX_DT=20};
  IDXL_Layout *list[MAX_DT]; 
  void badLayout(IDXL_Layout_t l,const char *callingRoutine) const;
public:
	IDXL_Layout_List();
	void pup(PUP::er &p);
	~IDXL_Layout_List();
	
	/// If this isn't a valid, allocated layout, abort.
	inline void check(IDXL_Layout_t l,const char *callingRoutine) const {
		if (l<FIRST_DT || l>=FIRST_DT+MAX_DT || list[l-FIRST_DT]==NULL) 
			badLayout(l,callingRoutine);
	}
	
	/// Insert a new layout
	IDXL_Layout_t put(const IDXL_Layout &dt);
	
	/// Look up an old layout
	inline const IDXL_Layout &get(IDXL_Layout_t l,const char *callingRoutine) const {
		check(l,callingRoutine);
		return *list[l-FIRST_DT];
	}
	
	/// Free this old layout
	void destroy(IDXL_Layout_t l,const char *callingRoutine);
	
	/// Clear all stored layouts:
	void empty(void);
	
	static IDXL_Layout_List &get(void);
};
PUPmarshall(IDXL_Layout_List)

#endif

