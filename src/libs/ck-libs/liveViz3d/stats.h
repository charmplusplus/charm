/*
Tools for measuring timing and profiling statistics.

Orion Sky Lawlor, olawlor@acm.org, 2004/8/18
*/
#ifndef __OSL_GL_STATS_H
#define __OSL_GL_STATS_H

#if CMK_LIVEVIZ3D_CLIENT /* serial version */
#  include "osl/osl.h"
#else 
#  include "charm.h"
#endif

/**
Used to compute, store, and later plot the time 
for various operations.
*/
namespace stats {

#if CMK_LIVEVIZ3D_CLIENT /* serial version */
	/// Return the wall time, in seconds.
	inline double time(void) {return osl::time();}
#else
	/// Return the wall time, in seconds.
	inline double time(void) {return CkWallTimer();}
#endif

	/**
	  Describes a particular operation, such as a kind of drawing.
	*/
	class op_t {
	public:
		int idx;
	};
	
	/// Can only define up to this many operations (to avoid dynamic allocation)
	enum {op_null=0, op_max=100};
	
	/// Number of operations currently defined (0..op_max)
	extern int op_len;
	
	/**
	  Define a new timing operation.  Can be called at startup time
	  to initialize a static variable.
	*/
	op_t time_op(const char *shortName,const char *desc);
	op_t count_op(const char *shortName,const char *desc,const char *units);
	
	/**
	  Sums up the time taken by each operation.
	*/
	class stats {
	public:
		/** Accumulators for each op_t: */
		double t[op_max];
		
		stats() {zero();}
		
		/// Clear all accumulators.
		void zero(void) {
			for (int op=0;op<op_len;op++) t[op]=0.0;
		}
		/// Add this value to this accumulator.
		void add(double val,op_t op) {t[op.idx]+=val;}

		/// Look up the value in this accumulator.
		double get(op_t op) const {return t[op.idx];}
		void set(double val,op_t op) {t[op.idx]=val;}
		
		/// Add everything in this object, scaled by scale, to us.
		void add(const stats &s,double scale=1.0) {
			for (int op=0;op<op_len;op++) t[op]+=s.t[op]*scale;
		}
		
		/// Print these stats, scaled by scale, where they exceed threshold.
		void print(FILE *f,const char *what,double scale=1.0,double thresh=0.001) const;
	};
	
	/// Return the current stats object.
	stats *get(void);
	
	/// Start running this operation.  Returns the old operation that was running.
	op_t swap(op_t op);
	
	/// Sentry class: wraps timing of one operation,
	///   including save/restore for nested operations.
	class op_sentry {
		op_t prev_op, op;
	public:
		op_sentry(op_t op_) :op(op_) 
		{
			prev_op=swap(op);
		}
		~op_sentry() {
			swap(prev_op);
		}
	};
};


#endif
