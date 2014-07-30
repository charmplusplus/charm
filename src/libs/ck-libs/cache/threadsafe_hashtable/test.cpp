/*
  Multithread-friendly hashtable: test driver code.

  Dr. Orion Sky Lawlor, lawlor@alaska.edu, 2011-05-24 (Public Domain)
*/
#include <iostream>
#include <iomanip>
#include <math.h>
#include <omp.h> /* for omp_set_num_threads */
#include "inc.c" /* for time_function */


int collisions=0;
#define DEBUG_HASH_COLLISIONS collisions++;
#include "hashtable_mt.h"


int key_randomizer=0; /* used in hash generator */
int key_mask=~0; /* used in hash generator */
inline int make_key(int i) {
	i*=key_randomizer; /* randomizing function */
	return key_mask&(i^(i>>16));
}
hashtable_mt<int,float,int (*)(int)> h(0,make_key, -1,-99.0);

int n=0; /* total number of hashtable operations */

int hashstart=0,hashdir=1;
void setdir(int dir) {
	if (dir==0) {hashstart=0; hashdir=1;}
	else {hashstart=n-1;hashdir=-1;}
}

int do_hashtable_writes(void) {
	collisions=0;
#pragma omp parallel for 
	for (int j=0;j<n;j++) {
		int i=hashstart+hashdir*j;
		h.put(i,i*1.234);
	}
	return 0;
}

double do_hashtable_reads(void) {
	collisions=0;
	double sum=0.0;
#pragma omp parallel for reduction(+:sum) 
	for (int j=0;j<n;j++) {
		int i=hashstart+hashdir*j;
		sum+=h.get(i);
	}
	return sum;
}

int time_hashtable_reads(void) { return (int)do_hashtable_reads(); }

int main() {
	for (int nthreads=1;nthreads<=8;nthreads*=2) {
		printf("%d threads:\n",nthreads);
		omp_set_num_threads(nthreads);
		
	/* check correctness: */
		srand(nthreads);
		for (int repeat=0;repeat<100;repeat++) {
			std::cout<<"."; std::cout.flush(); /* progress indicator */
			
			bool bad=false;
			
			/* Randomize... */
			n=10+rand()%1000000; // length of table
			key_randomizer=8192+rand()%10000; // hashtable key generator
			if (rand()%2) key_mask=0x7f7f7f7f; /* lose lots of bits-> collisions! */
			else	key_mask=~0; /* keep all bits */
			int initlen=rand()%(2*n); // initial table size */
			int repeatwrite=1+rand()%2; // number of write passes (single or multi)
			int writedir=rand()%2, readdir=rand()%2; // direction/thread assignment
			
			/* Fill the table */
			h.reinit(initlen); /* flush the old table */
			for (int writepass=0;writepass<repeatwrite;writepass++) {
				setdir(writedir^writepass);
				do_hashtable_writes();
				if (h.size()!=n) {
					std::cout<<"expected "<<n<<", got "<<h.size()<<" entries!";
					bad=true;
				}
			}
			
			/* Read the entries back */
			collisions=0;
			setdir(readdir);
			double sum=do_hashtable_reads();
			double expected=n*(n-1.0)/2.0*1.234;
			double diff=expected-sum;
			if (fabs(diff)/expected>1.0e-9) {
				std::cout<<"expected sum of "<<std::setprecision(15)<<expected<<", got "<<sum<<"\n";
				bad=true;
			}
			
			if (bad) {
				std::cout<<"CORRECTNESS ERRROR on pass "<<repeat<<" with "<<nthreads<<" threads:\n";
				std::cout<<"  n="<<n<<"\n";
				std::cout<<"  key_randomizer="<<key_randomizer<<"\n";
				std::cout<<"  key_mask="<<std::hex<<key_mask<<std::dec<<"\n";
				std::cout<<"  initlen="<<initlen<<"\n";
				std::cout<<"  repeatwrite="<<repeatwrite<<"\n";
				std::cout<<"  dirs="<<writedir<<","<<readdir<<"\n";
				std::cout<<"  collisions="<<collisions<<"\n";
			}
		}
		std::cout<<"\n";
		
	/* check performance: */
		n=100000;
		h.reinit(1);
		key_randomizer=8193;
		key_mask=~0;
		setdir(0);
		double t=time_function(do_hashtable_writes);
		printf("	writes: %.3f ns per (%.3f ms total)\n",t*1.0e9/n,t*1.0e3);
		std::cout<<"	collisions: "<<collisions<<"\n";
		std::cout<<"	total values: "<<h.size()<<"\n";
	
		t=time_function(time_hashtable_reads);
		printf("	reads: %.3f ns per (%.3f ms total)\n",t*1.0e9/n,t*1.0e3);
		std::cout<<"	collisions: "<<collisions<<"\n";
		std::cout<<"	total values: "<<h.size()<<"\n";
	}
	
	return 0;
}

