#ifndef ARRAY_H
#define ARRAY_H

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <charm++.h>
#include <cblas.h>

namespace CharjArray {
  class Range {
  public:
    int size, start, stop;
    Range() {}
    Range(int size_) : size(size_), start(0), stop(size) {}
    Range(int start_, int stop_) :
    size(stop_ - start_), start(start_), stop(stop_) {
      assert(stop >= start);
    }
  };

  template<int dims>
  class Domain {
  public:
    Range ranges[dims];
    
    Domain() {}

    Domain(Range ranges_[]) {
      for (int i = 0; i < dims; i++) 
	ranges[i] = ranges_[i];      
    }

    Domain(Range range) {
      ranges[0] = range;
    }

    Domain(Range range1, Range range2) {
      // TODO: fix Charj generator so it uses the array
      ranges[0] = range1;
      ranges[1] = range2;
    }

    int size() const {
      int total = 0;

      for (int i = 0; i < dims; i++)
	if (total == 0)
	  total = ranges[i].size;
	else
	  total *= ranges[i].size;

      return total;
    }
  };

  template<int dims>
  class RowMajor {
  public:
    static int access(const int i, const Domain<dims> &d) {
      return i - d.ranges[0].start;
    }
    static int access(const int i, const int j, const Domain<dims> &d) {
      return (i - d.ranges[0].start) * d.ranges[1].size + j -
        d.ranges[1].start;
    }
    // Generic access method, not used right now.
    // static int access(const int *i, const Domain<dims> &d) {
    //   int off = i[0];
    //   int dimoff = 1;
    //   for (int j = ndims-1; j > 0; --j) {
    //     dimoff *= d.ranges[j].size;
    //     off += dimoff * (i[j] - d.ranges[j].start);
    //   }
    //   return off;
    // }
  };

  template<int dims>
  class ColMajor {
  public:
    static int access(const int i, const Domain<dims> &d) {
      return i - d.ranges[0].start;
    }
    static int access(const int i, const int j, const Domain<dims> &d) {
      return (j - d.ranges[1].start) * d.ranges[1].size + i -
        d.ranges[0].start;
    }
  };

  template<class type, int dims = 1, class atype = RowMajor<dims> >
  class Array {
  private:
    Domain<dims> domain;
    type *block;
    int ref_cout;
    Array* ref_parent;

  public:
    Array(Domain<dims> domain_) : ref_parent(0) {
      init(domain_);
    }

    Array(type **block_) {
      block = *block_;
    }

    Array() : ref_parent(0) {

    }

    Array(Array* parent, Domain<dims> domain_) : ref_parent(parent) {
      domain = domain_;
      block = parent->block;
    }

    void init(Domain<dims> &domain_) {
      domain = domain_;
      //if (atype == ROW_MAJOR)
      block = new type[domain.size()];
    }

    ~Array() {
      delete[] block;
    }

    /*type* operator[] (const Domain<dims> &domain) {
      return block[domain.ranges[0].size];
      }*/

    type& operator[] (const int i) {
      return block[atype::access(i, domain)];
    }

    const type& operator[] (const int i) const {
      return block[atype::access(i, domain)];
    }

    type& access(const int i, const int j) {
      return block[atype::access(i, j, domain)];
    }

    const type& access(const int i, const int j) const {
      return block[atype::access(i, j, domain)];
    }

    Array<type, dims, atype>* operator[] (const Domain<dims> &domain) {
      return new Array<type, dims, atype>(this, domain);
    }

    int size() const {
      return domain.size();
    }

    int size(int dim) const {
      return domain.ranges[dim].size;
    }

    void pup(PUP::er& p) { }

    void fill(const type &t) {
      for (int i = 0; i < domain.size(); ++i)
	block[i] = t;
    }

    /// Do these arrays have the same shape and contents?
    bool operator==(const Array &rhs) const
    {
      for (int i = 0; i < dims; ++i)
	if (this->size(i) != rhs.size(i))
	  return false;

      for (int i = 0; i < this->size(); ++i)
	if (this->block[i] != rhs.block[i])
	  return false;

      return true;
    }
    bool operator!=(const Array &rhs) const
    {
      return !(*this == rhs);
    }
  };

  /**
     A local Matrix class for various sorts of linear-algebra work.

     Indexed from 0, to reflect the C-heritage of Charj.
   */
  template <typename V, class atype = RowMajor<2> >
  class Matrix : public Array<V, 2, atype>
  {
  public:
    Matrix() { }
    /// A square matrix
    Matrix(unsigned int n) : Array<V,2,atype>(Domain<2>(n,n)) { }

    /// A identity matrix
    static Matrix* ident(int n)
    {
      Matrix *ret = new Matrix(n);
      ret->fill(0);

      for (int i = 0; i < n; ++i)
	ret->access(i,i) = 1;

      return ret;
    }
  };

  template <typename T, class atype = RowMajor<1> >
  class Vector : public Array<T, 1, atype>
  {
  public:
    Vector() { }
    Vector(unsigned int n) : Array<T, 1, atype>(Range(n)) { }
  };

  /// Compute the inner (dot) product v1^T * v2
  // To compute v1^H * v2, call as dot(v1.C(), v2)
  template<typename T, class atype1, class atype2>
  T dot(const Vector<T, atype1> *pv1, const Vector<T, atype2> *pv2)
  {
    const Vector<T, atype1> &v1 = *pv1, &v2 = *pv2;
    assert(v1.size() == v2.size());
    // XXX: This default initialization worries me some, since it
    // won't necessarily be an additive identity for all T. - Phil
    T ret = T();
    int size = v1.size();
    for (int i = 0; i < size; ++i)
      ret += v1[i] * v2[i];
    return ret;
  }
  template <>
  float dot<float, RowMajor<1>, RowMajor<1> >(const Vector<float, RowMajor<1> > *pv1,
					      const Vector<float, RowMajor<1> > *pv2)
  {
    const Vector<float, RowMajor<1> > &v1 = *pv1, &v2 = *pv2;
    assert(v1.size() == v2.size());
    return cblas_sdot(v1.size(), &(v1[0]), 1, &(v2[0]), 1);
  }
  template <>
  double dot<double, RowMajor<1>, RowMajor<1> >(const Vector<double, RowMajor<1> > *pv1,
						const Vector<double, RowMajor<1> > *pv2)
  {
    const Vector<double, RowMajor<1> > &v1 = *pv1, &v2 = *pv2;
    assert(v1.size() == v2.size());
    return cblas_ddot(v1.size(), &(v1[0]), 1, &(v2[0]), 1);
  }
}

#endif
