#ifndef _DEFINES
#define _DEFINES

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

#define NULL 0

#endif
