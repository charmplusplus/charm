#ifndef _DEFINES
#define _DEFINES

#include <stdlib.h> /* for NULL */

typedef unsigned char byte;

struct point
{
	int x;
	int y;
	point() {}
	point(int x_,int y_) :x(x_), y(y_) {}
	bool operator==(const point &p) {
		return x==p.x && y==p.y;
	}
};

typedef struct point Point;

struct imageRect
{
	int l,r;
	int t,b;
};

typedef struct imageRect Rect;

#endif
