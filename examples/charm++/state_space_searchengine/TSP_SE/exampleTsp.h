/*
 * exampleTsp.h
 *
 *  Created on: Mar 4, 2010
 *      Author: gagan
 */

#ifndef EXAMPLETSP_H_
#define EXAMPLETSP_H_

#define Nmax 20
extern int  N;
extern int maxD;

class IntMap2on2{
private:
	int **Map;
	int keyXmax;
	int keyYmax;
public:
	IntMap2on2() {
		keyXmax = 0;
		keyYmax = 0;
		Map = NULL;
	}
	~IntMap2on2() {
		if (Map != NULL) {
			delete[] Map[0];
			delete[] Map;
			Map = NULL;
		}
	}
	IntMap2on2(int keyX, int keyY) :
		keyXmax(keyX), keyYmax(keyY) {
		CkAssert(keyXmax >= 0);
		CkAssert(keyYmax >= 0);
		CkAssert(keyXmax < 10000000);
		CkAssert(keyYmax < 10000000);
		Map = new int*[keyXmax];
		int *mapbuf = new int[keyXmax * keyYmax];
		for (int x = 0; x < keyXmax; x++) {
			Map[x] = mapbuf + keyYmax * x;
			memset(Map[x], -1, keyYmax * sizeof(int));
		}

	}
	void buildMap(int keyX = 1, int keyY = 1) {

		CkAssert(keyX > 0);
		CkAssert(keyY > 0);
		keyXmax = keyX;
		keyYmax = keyY;
		CkAssert(keyXmax > 0);
		CkAssert(keyYmax > 0);
		CkAssert(keyXmax < 10000000);
		CkAssert(keyYmax < 10000000);

		Map = new int*[keyXmax];
		int *mapbuf = new int[keyXmax * keyYmax];
		for (int x = 0; x < keyXmax; x++) {
			Map[x] = mapbuf + keyYmax * x;
			memset(Map[x], -1, keyYmax * sizeof(int));
		}

	}
	void pup(PUP::er &p) {
		p | keyXmax;
		p | keyYmax;
		CkAssert(keyXmax >= 0);
		CkAssert(keyYmax >= 0);
		CkAssert(keyXmax < 10000000);
		CkAssert(keyYmax < 10000000);
		int *mapbuf = NULL;
		if (keyXmax > 0) {
			CkAssert(keyYmax > 0);
			if (p.isUnpacking()) {
				Map = new int*[keyXmax];

				mapbuf = new int[keyXmax * keyYmax];
			}
			for (int x = 0; x < keyXmax; x++) {
				if (keyYmax > 0) {
					if (p.isUnpacking())
						Map[x] = mapbuf + keyYmax * x;
					PUParray(p, Map[x], keyYmax);
				}
			}
		}
	}
	inline int getXmax() {
		return (keyXmax);
	}
	inline int getYmax() {
		return (keyYmax);
	}
	int getCentroid(int torusMap);
	inline int get(int X, int Y) {
		/*
		 CkAssert(X<keyXmax);
		 CkAssert(Y<keyYmax);
		 */
		return (Map[X][Y]);
	}
	//    inline &int put(int X, int Y){&(Map[X][Y]);}
	inline void set(int X, int Y, int value) {
		// CkAssert(numPes>value);
		CkAssert(X < keyXmax);
		CkAssert(Y < keyYmax);
		Map[X][Y] = value;
	}
	void dump() {
		for (int x = 0; x < keyXmax; x++)
			for (int y = 0; y < keyYmax; y++)
				CkPrintf("%d %d %d \n", x, y, get(x, y));
	}
};


#endif /* EXAMPLETSP_H_ */
