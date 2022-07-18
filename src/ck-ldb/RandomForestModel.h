/*
 * The file is based on the Random-Forest-Matlab at
 * https://github.com/karpathy/Random-Forest-Matlab available under BSD license

 * Copyright (c) <2014>, <Andrej Karpathy>
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * purpose.
 */

#include "charm.h"
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#define NUM_TREES 20
#define LB_CLASSES 6
#define DEPTH 9

namespace rfmodel {
struct DataMatrix {
  std::vector<double> data;
  int num_rows;
  int num_cols;
  DataMatrix() {}
  DataMatrix(int nrows, int ncols, bool ones = false) : num_rows(nrows), num_cols(ncols) {
    if (ones)
      data.resize((size_t)nrows * ncols, 1);
    else
      data.resize((size_t)nrows * ncols, 0);
  }
  DataMatrix(const std::vector<double>& d, int nrows, int ncols)
      : data(d), num_rows(nrows), num_cols(ncols) {
    CkAssert(d.size() == (size_t)nrows * (size_t)ncols);
  }

  inline double& data_at(int x, int y) { return data[x * num_cols + y]; }

  // Repeat data's rows asize times and it's columns bsize times, store result in R
  inline void repmat(int asize, int bsize, DataMatrix& R) const {
    for (int i = 0; i < num_rows * asize; i++)
      for (int j = 0; j < num_cols * bsize; j++)
        R.data[i * num_cols * bsize + j] =
            data[(i % num_rows) * num_cols + (j % num_cols)];
  }

  // Find indices less than scalar
  inline void findIndicesLT(double scalar, DataMatrix& R) const {
    int count = 0;
    for (int i = 0; i < num_rows * num_cols; i++) {
      if (data[i] < scalar) R.data[count++] = i;
    }
    R.data.resize(count);
    R.num_rows = count;
  }

  // Store data from all rows specified as indices in select_rows[] into R
  inline void subset_rows(int* select_rows, int s_rows_size, DataMatrix& R) const {
    for (int j = 0; j < s_rows_size; j++)
      for (int i = 0; i < num_cols; i++)
        R.data[j * num_cols + i] = data[select_rows[j] * num_cols + i];
  }

  // Store data from all cols specified as indices in select_cols[] into R
  inline void subset_cols(int* select_cols, int s_cols_size, DataMatrix& R) const {
    for (int j = 0; j < s_cols_size; j++)
      for (int i = 0; i < num_rows; i++)
        R.data[i * num_rows + j] = data[i * num_rows + select_cols[j]];
  }

  // Set 1 where data matches value
  inline void findValue(double value, DataMatrix& R) const {
    for (int i = 0; i < num_rows; i++)
      for (int j = 0; j < num_cols; j++)
        if (fabs(data[i * num_rows + j] - value) < 0.0001)
          R.data[i * num_rows + j] = 1;
        else
          R.data[i * num_rows + j] = 0;
  }

  // Find indices Not Equal to scalar
  inline void findIndicesNE(double scalar, DataMatrix& R) const {
    int count = 0;
    for (int i = 0; i < num_rows * num_cols; i++) {
      if (fabs(data[i] - scalar) > 0.0001) {
        R.data[count++] = i;
      }
    }
    R.data.resize(count);
    R.num_rows = count;
  }

  // Find indices Equal to scalar
  inline void findIndicesE(double scalar, DataMatrix& R) const {
    int count = 0;
    for (int i = 0; i < num_rows * num_cols; i++) {
      if (fabs(data[i] - scalar) < 0.0001) {
        R.data[count++] = i;
      }
    }
    R.data.resize(count);
    R.num_rows = count;
    R.num_cols = 1;
  }

  // Find index with max value
  inline int maxIndex() const {
    int mIndex = 0;
    double max = data[0];
    for (int i = 1; i < num_rows * num_cols; i++)
      if (max < data[i]) {
        max = data[i];
        mIndex = i;
      }
    return mIndex;
  }

  // return a random value from matrix
  inline double randomValue() const {
    int randIndex = rand() % (num_rows * num_cols);
    return data[randIndex];
  }

  // Store matrix multiply output of X * Y in data
  inline void matrix_multiply(const DataMatrix& X, const DataMatrix& Y) {
    int x_rows = X.num_rows;
    int x_y_cols_rows = X.num_cols;
    int y_cols = Y.num_cols;

    for (int i = 0; i < x_rows; i++)
      for (int j = 0; j < x_y_cols_rows; j++)
        for (int k = 0; k < y_cols; k++)
          data_at(i, k) += X.data[i * x_y_cols_rows + j] * Y.data[j * y_cols + k];
  }

  // Concatenate two matrices
  inline void combine(const DataMatrix& A, const DataMatrix& B) {
    int N = A.num_rows;
    int asize = A.num_cols;
    int bsize = B.num_cols;
    int cols = asize + bsize;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < cols; j++)
        if (j < asize)
          data[i * cols + j] = A.data[i * asize + j];
        else
          data[i * cols + j] = B.data[i * bsize + (j - asize)];
  }
};

struct Model {
  int classifierID;
  int r1;
  int r2;
  std::vector<double> w;
  double weakTest(const DataMatrix& X) const;
};

struct TreeModel {
  int l_X, l_D;
  int* classes;
  std::vector<Model> weakModels;
  std::vector<double> leafdist;
  void treeTest(const DataMatrix& X, std::vector<double>& Ysoft) const;
};

struct ForestModel {
  int classes[LB_CLASSES];
  std::vector<TreeModel> treeModels;
  void readModel(const char* dir);
  int forestTest(std::vector<double>& X, int num_rows, int num_cols);
};

}
