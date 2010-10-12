#ifndef ARRAY_H
#define ARRAY_H

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <charm++.h>

namespace CharjArray {
  class Range {
  public:
    int size, start, stop;
    Range() {}
    Range(int size_) : size(size_), start(0), stop(size) {}
    Range(int start_, int stop_) :
      start(start_), stop(stop_), size(stop_ - start_) {
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

    int size() {
      int total = 0;
      for (int i = 0; i < dims; i++)
	if (total == 0)
	  total = ranges[i].size;
	else
	  total *= ranges[i].size;
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
      delete block;
    }

    /*type* operator[] (const Domain<dims> &domain) {
      return block[domain.ranges[0].size];
      }*/

    type& operator[] (const int i) {
      return block[atype::access(i, domain)];
    }

    type& access(const int i, const int j) {
      return block[atype::access(i, j, domain)];
    }

    Array<type, dims, atype>* operator[] (const Domain<dims> &domain) {
      return new Array<type, dims, atype>(this, domain);
    }

    int size() {
      return domain.size();
    }

    int size(int dim) {
      return domain.ranges[dim].size;
    }

    void pup(PUP::er& p) { }
  };
}

#endif
