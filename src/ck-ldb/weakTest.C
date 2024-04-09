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

double Model::weakTest(const DataMatrix& X) const {
  // X is NxD
  double final_yhat = 1.0;
  int N = X.num_rows;
#ifdef DEBUG_RF
  if (classifierID == 2.0)
    printf("\nTesting on weakmodel r1[%d], r2[%d], w[%lf,%lf,%lf]\n", r1, r2, w[0], w[1],
           w[2]);
#endif
// Retaining pseudo code for decision stump classifier, distance based learner
// as placeholders to be potentially implemented later
#if 0
  if (classifierID == 1) {
    // Decision stump classifier

      yhat = (double)X(:, model->r) < model->t;

      int* select_cols;
      select_cols = new int[1];
      select_cols[0] = r;

      DataMatrix *yhat_op = MatrixOp::_subset_cols(X, num_rows, num_cols, select_cols,
      1);
      yhat = yhat_op->data;
      final_yhat = MatrixOp::findIndicesLT(yhat,N, t);
    } else
#endif
  if (classifierID == 2) {
    // 2-D linear classifier stump

    int select_cols[2];
    select_cols[0] = r1 - 1;
    select_cols[1] = r2 - 1;

    DataMatrix subset_arr(X.num_rows, 2);
    X.subset_cols(select_cols, 2, subset_arr);
    const DataMatrix ones_arr(N, 1, true);
    DataMatrix combined_arr(subset_arr.num_rows, subset_arr.num_cols + ones_arr.num_rows);
    combined_arr.combine((const DataMatrix)subset_arr, ones_arr);
    const DataMatrix w_arr(w, 3, 1);
    DataMatrix mm(combined_arr.num_rows, w_arr.num_cols);
    mm.matrix_multiply(combined_arr, w_arr);

    double zero = 0.0;
    DataMatrix yhat(mm.num_rows, mm.num_cols);
    mm.findIndicesLT(zero, yhat);
    if (yhat.data.size() == 0)
      final_yhat = 0.0;
    else
      final_yhat = yhat.data[0] + 1;
  }

  return final_yhat;
}
