/*
Performance statistics collection.

Orion Sky Lawlor, olawlor@acm.org, 2004/8/18
*/
#include <stdio.h>
#include "converse.h"
#include "stats.h"

using stats::op_t;

// stats::get
static stats::stats staticStats;

stats::stats *stats::get(void)
{
	return &staticStats;
}

// stats::swap support
static op_t last_op={stats::op_null};
static double last_op_start=0;
op_t stats::swap(op_t op)
{
	double cur=time();
	op_t ret=last_op;
	staticStats.add(cur-last_op_start,ret);
	last_op_start=cur;
	last_op=op;
	return ret;
}

// stats::op_t registration
namespace stats { 
class op_info_t {
public:
	const char *name; // Short human-readable no-spaces name.
	const char *desc; // Long human-readable description.
	bool isTime; // if true, this is a timing field.
	const char *units; // Human-readable units, like "seconds" or "bytes"
}; 
};
static stats::op_info_t op_info[stats::op_max];
int stats::op_len;
static op_t addOp(const stats::op_info_t &i) {
	if (stats::op_len==stats::op_max) {
		CmiAbort("Registered too many operations!\n");
	}
	op_info[stats::op_len]=i;
	op_t ret;
	ret.idx=stats::op_len++;
	return ret;
}

op_t stats::time_op(const char *shortName,const char *desc_)
{
	op_info_t i;
	i.name=shortName; i.desc=desc_; i.isTime=true; i.units="seconds";
	return addOp(i);
}

op_t stats::count_op(const char *shortName,const char *desc_,const char *units_)
{
	op_info_t i;
	i.name=shortName; i.desc=desc_; i.isTime=false; i.units=units_;
	return addOp(i);
}

void stats::stats::print(FILE *f,const char *what,double scale,double thresh) const
{
	fprintf(f,"%s stats { \n",what);
	for (int op=1;op<op_len;op++) 
		if (t[op]*scale>0) {
			double val=t[op]*scale;
			op_info_t &i=op_info[op];
			const char *units=i.units;
			if (i.isTime) {
				if (val<1.0) {val*=1.0e3; units="ms";}
			}
			fprintf(f,"  %s_%s: %.2f %s\n",what,i.name,val,units);
		}
	fprintf(f,"} \n");
}

