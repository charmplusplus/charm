#ifndef  _DEFINES_H_
#define _DEFINES_H_

#define  LOW_VALUE 0 
#define  HIGH_VALUE 255 

#define wrap_x(a)	(((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)	(((a)+num_chare_y)%num_chare_y)
#define wrap_z(a)	(((a)+num_chare_z)%num_chare_z)
#define index(a, b, c)	( (a)*(blockDimY+2)*(blockDimZ+2) + (b)*(blockDimZ+2) + (c) )

#define  START_ITER     10
#define   END_ITER      20
#define PRINT_FREQ     100 
#define CKP_FREQ		100
#define MAX_ITER	 10	
#define WARM_ITER		5
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define FRONT			5
#define BACK			6
#define DIVIDEBY7       	0.14285714285714285714

#endif
