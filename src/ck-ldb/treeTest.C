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

#include "RandomForestModel.h"

using namespace rfmodel;

// Test a tree
void TreeModel::treeTest(const DataMatrix& X, vector<double>& Ysoft) const {
  //[N, D]= size(X); //Returns number of rows and columns in matrix X
  int N = X.num_rows;
  int D = X.num_cols;

  int nd = (int)pow(2, DEPTH) - 1;

  int numInternals = (nd + 1) / 2 - 1;

  // Propagate data down the tree using weak classifiers at each node
  vector<double> dataix(N * nd, 0.0);

  for (int n = 1; n <= numInternals; n++) {
    DataMatrix* reld;
    DataMatrix* Xrel;

    // get relevant data at this node
    if (n == 1) {
      reld = new DataMatrix(N, 1, true);  //=1;
      Xrel = new DataMatrix(X.data, N, D);
    } else {
      int select_cols[1];
      select_cols[0] = n - 1;
      DataMatrix dataix_array(dataix, N, D);
      DataMatrix reld_resp(dataix_array.num_rows, 1);
      dataix_array.subset_cols(select_cols, 1, reld_resp);
      reld = new DataMatrix(reld_resp.num_rows, reld_resp.num_cols);
      reld_resp.findValue(1, *reld);
      int* select = new int[reld->num_rows * reld->num_cols];
      int c = 0;
      for (int i = 0; i < reld->num_rows * reld->num_cols; i++)
        if (reld->data[i] == 1) select[c++] = i;
      Xrel = new DataMatrix(c, X.num_cols);
      X.subset_rows(select, c, *Xrel);
      delete select;
    }

    if (Xrel->num_rows * Xrel->num_cols == 0) {
      delete reld;
      delete Xrel;
      continue;  // empty branch
    }

    double yhat = weakModels[n - 1].weakTest(*Xrel);

    for (int i = 0; i < reld->num_rows * reld->num_cols; i++)
      dataix[(((int)reld->data[i] - 1) * nd + 2 * n) - 1] = yhat;

    for (int i = 0; i < reld->num_rows * reld->num_cols; i++)
      dataix[(((int)reld->data[i] - 1) * nd + 2 * n + 1) - 1] = 1.0 - yhat;

    delete reld;
    delete Xrel;
  }

  // Go over leafs and assign class probabilities
  for (int n = ((nd + 1) / 2); n < nd; n++) {
    int select_cols[1];
    select_cols[0] = n - 1;

    DataMatrix dataix_array(dataix, N, nd);
    DataMatrix dataix_subset(dataix_array.num_rows, 1);
    dataix_array.subset_cols(select_cols, 1, dataix_subset);

    DataMatrix ff(dataix_subset.num_rows, dataix_subset.num_cols);
    dataix_subset.findIndicesNE(0.0, ff);

    int select_rows[1];
    select_rows[0] = n - (nd + 1) / 2 + 1 - 1;  // c indexing
    DataMatrix leafdist_array(leafdist, l_X, l_D);
    DataMatrix hc(1, leafdist_array.num_cols);
    leafdist_array.subset_rows(select_rows, 1, hc);

    if (ff.num_rows * ff.num_cols > 0) {
      DataMatrix rep_matrix(hc.num_rows * ff.num_rows * ff.num_cols, hc.num_cols);
      hc.repmat(ff.num_rows * ff.num_cols, 1, rep_matrix);
      std::copy(rep_matrix.data.begin(), rep_matrix.data.end(), Ysoft.begin());
    }
  }
}
