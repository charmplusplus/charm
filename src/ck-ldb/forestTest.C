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
#include <cassert>

using namespace rfmodel;

int ForestModel::forestTest(std::vector<double>& X, int num_rows, int num_cols) {
  // X is NxD, where rows are data points

  int numTrees = NUM_TREES;
  vector<double> Ysoft(LB_CLASSES);

  vector<double> model_Ysoft(LB_CLASSES);
  DataMatrix Xarray(X, num_rows, num_cols);

  // Assuming initialization to 0s
  for (int i = 0; i < numTrees; i++) {
    treeModels[i].treeTest(Xarray, model_Ysoft);

    for (int ii = 0; ii < LB_CLASSES; ii++) Ysoft[ii] += model_Ysoft[ii];
  }
  return classes[DataMatrix(Ysoft, LB_CLASSES, 1).maxIndex()];
}
