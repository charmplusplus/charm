#ifndef _DEFINES
#define _DEFINES

#include <stdlib.h> /* for NULL */

typedef unsigned char byte;

struct point
{
	int x;
	int y;
};

typedef struct point Point;

struct imageRect
{
	int l,r;
	int t,b;
};

typedef struct imageRect Rect;

#endif
