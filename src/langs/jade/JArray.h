// emacs mode line -*- mode: c++; tab-width: 4 -*-

#ifndef JARRAY_H
#define JARRAY_H

#include <iostream>
#include <assert.h>

//for PUP::er
#include <charm++.h>

// for memcopy
#include <string.h>

#ifndef NULL
#define NULL 0
#endif
#define DPRINT(a)

typedef unsigned int uint;

template<class T>
class JArray {
	uint numDimensions;
	// e.g. a[10][20] dims = 10, 20	 dimSize = 20, 1
	uint *dims;		 // array containing size of dim 1, 2, ...; 1-based
	uint *dimSize;	 // array containing num elements in dim 1, 2, ...; 1-based
	T *data;

	// virtual array to be used for next PUP
	uint useVirtual;
	uint *start;
	uint *end;
	uint *stride;

public:
	// Ways to set the dimension of the array:
	// cons(dims)
	// cons(), setDimension
	// cons(), pup
	// cons(), resize

	// Constructor with dimensionality
	JArray(const int numDims):dims(NULL), data(NULL), dimSize(NULL), useVirtual(0), start(NULL), end(NULL), stride(NULL)
	{
		//ckout << CkMyPe() << ": JArray(" << numDims << ") reached" << endl;
//		assert(numDims<=3); // @@ arbitrary limitation, code should work ok for larger dims
		setDimension(numDims);
	}

	// Constructor without dimensionality
	// something should be called immediately to set the dimensionality
	JArray():numDimensions(0), dims(NULL), data(NULL), dimSize(NULL), useVirtual(0), start(NULL), end(NULL), stride(NULL)
	{
		//ckout << CkMyPe() << ": JArray() reached" << endl;
	}

	// Constructor from 1D C array
	JArray(const int numElements, T *rawdata):numDimensions(0), dims(NULL), data(NULL), dimSize(NULL), useVirtual(0), start(NULL), end(NULL), stride(NULL)
	{
		//CkPrintf("DEBUG: %d %d %d\n", numElements, rawdata[0], rawdata[1]);
		setDimension(1);
		resize1D(numElements);
		for(uint i=0; i<numElements; i++)
			data[i] = rawdata[i];
	}

	// Allocates dims and dimSize, but does not create the data array;
	// resize or pup must be called.
	void setDimension(uint numDims) {
		//ckout << CkMyPe() << ": JArray::setDimension(" << numDims << ") reached" << endl;
		numDimensions = numDims;
		dims = new uint[numDimensions];
		dimSize = new uint[numDimensions];
		start = new uint[numDimensions];
		end = new uint[numDimensions];
		stride = new uint[numDimensions];
		for(uint i=0; i<numDimensions; i++)
			dims[i] = dimSize[i] = start[i] = end[i] = stride[i] = 0;
	}

	// Copy constructor.  Needed for param marshalling.
	// virtual array stuff is not copied
	//
	// Differs from copy assignment because cc deals with
	// unallocated memory, but ca deals with a constructed object.
	JArray(const JArray &rhs) {
		int i = 0;

		//CkPrintf("DEBUG: Copy constructor called\n");
		setDimension(rhs.numDimensions);
		//CkPrintf("DEBUG: CC: rhs.numDimensions = %d numDimensions = %d\n", rhs.numDimensions, numDimensions);
		for(i=0; i<numDimensions; i++) {
			dims[i]=rhs.dims[i];
			dimSize[i]=rhs.dimSize[i];
			start[i] = end[i] = stride[i] = 0;
		}
		useVirtual=0;

		//memcpy(dims, rhs.dims, numDimensions*sizeof(uint));
		//CkPrintf("DEBUG: CC: rhs.dims[0] = %d, dims[0] = %d\n", rhs.dims[0], dims[0]);
		//std::copy(dims, dims+numDimensions, rhs.dims);
		//CkPrintf("DEBUG: CC: rhs.dims[0] = %d, dims[0] = %d\n", rhs.dims[0], dims[0]);
		//memcpy(dimSize, rhs.dimSize, numDimensions*sizeof(uint));
		//std::copy(dimSize, dimSize+numDimensions, rhs.dimSize);

		int numElements = dims[0] * dimSize[0];
		//	   CkPrintf("DEBUG: CC: numElements = %d\n", numElements);
		//	   delete [] data;
		data = new T[numElements];
		//memcpy(data, rhs.data, numElements);
		//std::copy(data, data+numElements, rhs.data);
		for(i=0; i<numElements; i++)
			data[i] = rhs.data[i];
		//	   CkPrintf("DEBUG: CC: *data = %d *rhs.data = %d\n", *data, *rhs.data);
	}

	~JArray() {
		delete [] dims;
		delete [] dimSize;
		delete [] data;
		delete [] start;
		delete [] end;
		delete [] stride;
		dims = NULL;
		dimSize = NULL;
		data = NULL;
		start = NULL;
		end = NULL;
		stride = NULL;
	}

	// resize populates dims and dimSize, and allocates the data array
	// numDimensions may already be set, numDims is provided again just for verification.
	// if numDimensions is not set, we set it to numDims.
	//
	// We do not allow resizing a JArray into a different number of
	// dimensions.	We could, but will disallow it until I can think
	// of a need for it.
	JArray& resize(const uint numDims, const uint d[]){
		//	   ckout << CkMyPe() << ": JArray::resize(" << numDims << ".. reached" << endl;
		int i;
		if (numDimensions == 0)
			setDimension(numDims);
		else
			assert(numDimensions==numDims);

		uint dimSz = 1;
		for(i=numDimensions-1; i>=0; i--) {
			dimSize[i] = dimSz;
			dimSz *= d[i];
			dims[i] = d[i];
		}

		uint numElements=dimSz;

		//	   cout << "DEBUG: " << numElements << " elements " << ", dimSize ";
		//	   for(i=0; i<numDimensions; i++)
		//		   cout << dimSize[i] << " ";
		//	   cout << endl;

		// resizing an already allocated array blows away the data.
		if (data != NULL)
			delete [] data;
		data = new T[numElements];

		return *this;
	}

	// resize
	JArray& resize1D(const uint d1){
		uint d[] = { d1 };
		return resize(1, d);
	}

	// resize
	JArray& resize2D(const uint d1, const uint d2){
		uint d[] = { d1, d2 };
		return resize(2, d);
	}

	// resize
	JArray& resize3D(const uint d1, const uint d2, const uint d3){
		uint d[] = { d1, d2, d3 };
		return resize(3, d);
	}

	// resize
	JArray& resize4D(const uint d1, const uint d2, const uint d3, const uint d4){
		uint d[] = { d1, d2, d3, d4 };
		return resize(4, d);
	}

	// resize
	JArray& resize5D(const uint d1, const uint d2, const uint d3, const uint d4, const uint d5){
		uint d[] = { d1, d2, d3, d4, d5 };
		return resize(5, d);
	}

	// resize
	JArray& resize6D(const uint d1, const uint d2, const uint d3, const uint d4, const uint d5, const uint d6){
		uint d[] = { d1, d2, d3, d4, d5, d6 };
		return resize(6, d);
	}

	// Get the size of the n'th dimension (1D, 2D ...)
	int getDim(const uint n) {
		assert(n>0 && n<=numDimensions);
		return dims[n-1];
	}

	T* getBaseAddress() const { return data; }

	//================================================================

	// nD
	// multi-dimensional JArray's.	a[b] where b is an array of
	// the dimensions desired.
	inline T& getElement(uint idx[]) {
		uint index = 0;
		for(uint i=0; i<numDimensions; i++) {
			assert(//idx[i] >= 0 && 
				idx[i] <= dims[i]);
			index += idx[i] * dimSize[i];
		}
		return data[index];
	}
	// multi-dimensional JArray's.	a[b] where b is an array of
	// the dimensions desired.
	//	   inline T& operator[](uint i[]) {
	//	   T &tmp = getElement(i);
	// //		  CkPrintf("DEBUG: operator[[]] = %d *data=%d\n", tmp, *data);
	//	   return tmp;
	//	   }

	// 1D
	inline T& getElement(int i) {
		return data[i];
	}
	inline T& getElementSlow(int i) {
		uint idx[] = { i };
		return getElement(idx);
	}
	inline T& operator () (int i0) {
		return getElement(i0);
	}
	// a[10]
	inline T& operator[](int i) {
		//	   uint idx[] = { i };
		//	   return (*this)[idx];
		return getElement(i);
	}

	// 2D
	inline T& getElement(int i, int j) {
		return data[i*dimSize[0]+j];
	}
	inline T& getElementSlow(int i, int j) {
		uint idx[] = { i, j };
		return getElement(idx);
	}
	inline T& operator () (int i0,int i1) {
		return getElement(i0, i1);
	}

	// 3D
	inline T& getElement(int i, int j, int k) {
		return data[i*dimSize[0]+j*dimSize[1]+k];
	}
	inline T& getElementSlow(int i, int j, int k) {
		uint idx[] = { i, j, k };
		return getElement(idx);
	}
	inline T& operator () (int i0, int i1, int i2) {
		return getElement(i0, i1, i2);
	}

	//	   inline const T& operator[](int i) const;

	// ================================================================

	// 1D
	// set Virtual Array
	JArray& sV(uint s, uint e, uint str) {
		assert(start != NULL);
		assert(end != NULL);
		assert(stride != NULL);
		start[0] = s;
		end[0] = e;
		stride[0] = str;
		useVirtual = 1;
		return *this;
	}

	// 2D
	// set Virtual Array
	JArray& sV(uint s, uint e, uint str,
			   uint s1, uint e1, uint str1) {
		sV(s, e, str);
		assert(numDimensions >= 2);
		start[1] = s1;
		end[1] = e1;
		stride[1] = str1;
		return *this;
	}
	// 2D set Row
	JArray& sR(uint row) {
		sV(row, row, 1, 0, getDim(2)-1, 1);
		return *this;
	}
	// 2D set Column
	JArray& sC(uint col) {
		sV(0, getDim(1)-1, 1, col, col, 1);
		return *this;
	}

	void pupHelper(T *vdata, uint *vindex, uint *path, int depth) {
		if (depth == numDimensions-1) {
			for(int i=start[depth]; i<=end[depth]; i+=stride[depth]) {
				path[depth] = i;
				vdata[*vindex] = getElement(path);
				(*vindex)++;
			}
		} else {
			for(int i=start[depth]; i<=end[depth]; i+=stride[depth]) {
				path[depth] = i;
				pupHelper(vdata, vindex, path, depth+1);
			}
		}
	}

	inline int ceiling(float f) {
		int fi = (int)f;
		return (f- fi)>=0.5 ? fi+1: fi;
	}

	virtual void pup(PUP::er &p){
		// virtual case, and is packing or sizing
		if (!p.isUnpacking() && useVirtual==1) {
			p|numDimensions;
			uint *vdims = new uint[numDimensions];
			uint *vdimSize = new uint[numDimensions];
			int i=0;
			for (i=0; i<numDimensions; i++)
				vdims[i] = ceiling( (end[i]-start[i]+1)/stride[i] );
			uint dimSz = 1;
			for(i=numDimensions-1; i>=0; i--) {
				vdimSize[i] = dimSz;
				dimSz *= vdims[i];
			}
			uint numElements=dimSz;
			p(vdims, numDimensions);
			p(vdimSize, numDimensions);
			T *vdata = new T[numElements];
			uint vindex = 0;
			uint *path = new uint[numDimensions];
			pupHelper(vdata, &vindex, path, 0);
			p(vdata, numElements);
			if (p.isPacking()) useVirtual=0;
			delete [] vdims;
			delete [] vdimSize;
			delete [] vdata;
			delete [] path;
		} else {// virtual case unpacking, or normal case pup
			p|numDimensions;
			if (p.isUnpacking()) {
				//		   dims = new uint[numDimensions];
				//		   dimSize = new uint[numDimensions];
				setDimension(numDimensions);
			}
			p(dims, numDimensions);
			p(dimSize, numDimensions);
			int numElements = dims[0] * dimSize[0];
			if (p.isUnpacking()) {
				data = new T[numElements];
			}
			p(data, numElements);
		}
	}

};

#endif
// JARRAY_H
