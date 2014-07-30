/**
  Weird assorted helpful include files for debugging and 
  testing assembly code.
  
  Orion Sky Lawlor, olawlor@acm.org, 2005/09/14 (Public Domain)
*/
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> /* for std::sort */
#include "inc.h"

/* To avoid cluttering up the screen during timing tests... */
int timer_only_dont_print=0;

/* Read one input integer from the user. */
int read_input(void) {
	int ret=0;

	if (timer_only_dont_print) return 0;
	printf("Please enter an input value:\n");
	fflush(stdout);
	if (1!=scanf("%i",&ret)) {
		if (feof(stdin))
			printf("read_input> No input to read!  Exiting...\n");
		else
			printf("read_input> Invalid input format!  Exiting...\n"); 
		exit(1);
	}
	printf("read_input> Returning %d (0x%X)\n",ret,ret);
	return ret;
}

/* Read one input string from the user. Returns 0 if no more input ready. */
int read_string(char *dest_str) {
	if (0==fgets(dest_str,100,stdin)) {
		return 0;
	}
	dest_str[99]=0;
	return 1;
}

/* Print this integer parameter (on the stack) */
void print_int(int i) {
	if (timer_only_dont_print) return;
	printf("Printing integer %d (0x%X)\n",i,i);
}
void print_long(long i) {
	if (timer_only_dont_print) return;
	printf("Printing integer %ld (0x%lX)\n",i,i);
}
void print_float(float f) {
	if (timer_only_dont_print) return;
	printf("Printing float %f (%g)\n",f,f);
}
CDECL void print_int_(int *i) {print_int(*i);} /* fortran, gfortran compiler */
CDECL void print_int__(int *i) {print_int(*i);} /* fortran, g77 compiler */

/******* Function Performance Profiling ***********/
/**
  Return the current time in seconds (since something or other).
*/
#if defined(_WIN32)
#  include <sys/timeb.h>
#  define time_in_seconds_granularity 0.1 /* seconds */
double time_in_seconds(void) { /* This seems to give terrible resolution (60ms!) */
        struct _timeb t;
        _ftime(&t);
        return t.millitm*1.0e-3+t.time*1.0;
}
#else /* UNIX or other system */
#  include <sys/time.h> //For gettimeofday time implementation
#  define time_in_seconds_granularity 0.01 /* seconds */
double time_in_seconds(void) {
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return tv.tv_usec*1.0e-6+tv.tv_sec*1.0;
}
#endif

/* A little empty subroutine, just to measure call/return overhead 
  inside time_fn, below. */
CDECL int timeable_fn_empty(void) {
	return 0;
}

/**
  Return the number of seconds this function takes to run.
  May run the function several times (to average out 
  timer granularity).
*/
double time_function_onepass(timeable_fn fn)
{
	unsigned int i,count=1;
	double timePer=0;
	for (count=1;count!=0;count*=2) {
		double start, end, elapsed;
		timer_only_dont_print=1;
		start=time_in_seconds();
		for (i=0;i<count;i++) fn();
		end=time_in_seconds();
		timer_only_dont_print=0;
		elapsed=end-start;
		timePer=elapsed/count;
		if (elapsed>time_in_seconds_granularity) /* Took long enough */
			return timePer;
	}
	/* woa-- if we got here, "count" reached integer wraparound before 
	  the timer ran long enough.  Return the last known time */
	return timePer;
}

/**
  Return the number of seconds this function takes to run.
  May run the function several times (to average out 
  timer granularity).
*/
double time_function(timeable_fn fn)
{
	static double empty_time=-1;
	if (empty_time<0) { /* Estimate overhead of subroutine call alone */
		empty_time=0; /* To avoid infinite recursion! */
		/* empty_time=time_function(timeable_fn_empty); */
	}
	enum {
#if defined(_WIN32) /* Win32 timer has coarse granularity--too slow otherwise! */
		ntimes=3
#else
		ntimes=5
#endif
	};
	double times[ntimes];
	for (int t=0;t<ntimes;t++)
		times[t]=time_function_onepass(fn);
	std::sort(&times[0],&times[ntimes]);
	return times[ntimes/2]-empty_time;
}

/**
  Time a function's execution, and print this time out.
*/
void print_time(const char *fnName,timeable_fn fn)
{
	double sec=time_function(fn);
	printf("%s: ",fnName);
	if (1 || sec<1.0e-6) printf("%.2f ns/call\n",sec*1.0e9);
	else if (sec<1.0e-3) printf("%.2f us/call\n",sec*1.0e6);
	else if (sec<1.0e0) printf("%.2f ms/call\n",sec*1.0e3);
	else printf("%.2f s/call\n",sec);
}

/********* Checksums ***************/
int iarray_print(int *arr,int n)
{
	int i=0,p;
	if (n<0 || n>1000) {
		printf("ERROR in iarray_print: passed invalid number of elements %d (0x%08x) for array %p.  Did you pass the arguments in the wrong order?\n",
			n,n,arr);
		exit(1);
	}
	if (timer_only_dont_print) return n;
	p=n;
	if (p>10) p=10; /* Only print first 10 elements */
	printf("iarray_print: %d elements\n",n);
	for (i=0;i<p;i++)
		printf("  arr[%d]=%d  (0x%08x)\n",i,arr[i],arr[i]);
	if (p<n) {
		i=n/2;
		printf("  arr[%d]=%d  (0x%08x)\n",i,arr[i],arr[i]);
		i=n-1;
		printf("  arr[%d]=%d  (0x%08x)\n",i,arr[i],arr[i]);
	}
	return n;
}

int farray_print(float *arr,int n)
{
	int i=0,p;
	if (n<0 || n>1000000) {
		printf("ERROR in farray_print: passed invalid number of elements %d (0x%08x) for array %p.  Did you pass the arguments in the wrong order?\n",
			n,n,arr);
		exit(1);
	}
	if (timer_only_dont_print) return n;
	p=n;
	if (p>10) p=10; /* Only print first 10 elements */
	printf("farray_print: %d elements\n",n);
	for (i=0;i<p;i++)
		printf("  arr[%d]=%f\n",i,arr[i]);
	return n;
}
void farray_fill(float *f,int n,float tol)
{
	int i,v=1;
	for (i=0;i<n;i++) {
		f[i]=(v&0xff)*tol;
		v=v*3+14;
	}
	f[n]=1776.0; /* sentinal value */
}
void farray_fill2(float *f,int n,float tol)
{
	int i,v=0xbadf00d;
	for (i=0;i<n;i++) {
		f[i]=(((v>>8)&0xff)+1)*tol;
		v=v*69069+1;
	}
	f[n]=1776.0; /* sentinal value */
}
int farray_checksum(float *f,int n,float tol)
{
	int i,v=0;
	double t=1.0/tol;
	if (f[n]!=1776.0) {
		printf("ERROR!  You wrote past the end of the array!\n");
		exit(0);
	}
	for (i=0;i<n;i++) {
		int k=(int)(f[i]*t);
		v+=k*(i+10+v);
		v&=0xffff;
	}
	printf("First 5 values: %d %d %d %d %d\n",
		(int)(f[0]*t),(int)(f[1]*t),(int)(f[2]*t),(int)(f[3]*t),(int)(f[4]*t));
	return v;
}

void cat(const char *fName) {
	FILE *f=fopen(fName,"rb");
	int binary=0, len=0, i;
	if (f==0) { printf("File '%s' does not exist.\n",fName); return; }
	fseek(f,0,SEEK_END);
	len=ftell(f);
	fseek(f,0,SEEK_SET);
	printf("-- %s: %d bytes --\n",fName,len);
	if (len>1000) len=1000;
	for (i=0;i<len;i++) {
		unsigned char c=fgetc(f);
		if (c<9 || c>127) binary=1;
		if (binary) {
			if (i%16 == 0) printf("\n");
			printf("0x%02x ",c);
		} else {
			printf("%c",c);
		}
	}
	printf("-- end of %s --\n",fName);
	fclose(f);
}

