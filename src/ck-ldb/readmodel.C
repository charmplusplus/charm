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
#if CMK_USE_ZLIB
#include "zlib.h"
#endif
#include <stdio.h>
#ifndef PATH_MAX
#define PATH_MAX 65535
#endif
#define LINE_SIZE 1024

using namespace rfmodel;

void ForestModel::readModel(const char* dir) {
#if CMK_USE_ZLIB
  treeModels.resize(NUM_TREES);

  gzFile pFile;
  double f;

  for (int i = 0; i < LB_CLASSES; i++) classes[i] = i + 1;

  char buffer[PATH_MAX];
  char linebuffer[LINE_SIZE];
  sprintf(buffer, "%s/big_leafdist.txt.gz", dir);
  pFile = gzopen(buffer, "r");

  if (pFile == NULL) CkAbort("\nUnable to open model files.\n");
  for (int k = 0; k < NUM_TREES; k++) {
    int i, X, D;
    gzgets(pFile, linebuffer, LINE_SIZE);
    sscanf(linebuffer, "%d %d %d", &i, &X, &D);

    treeModels[i - 1].classes = classes;
    treeModels[i - 1].l_X = X;
    treeModels[i - 1].l_D = D;
    treeModels[i - 1].leafdist.resize(X * D);

    for (int j = 0; j < X * D; j++) {
      gzgets(pFile, linebuffer, LINE_SIZE);
      sscanf(linebuffer, "%lf", &f);
      treeModels[i - 1].leafdist[j] = f;
    }
  }
  gzclose(pFile);

  sprintf(buffer, "%s/big_weakmodel.txt.gz", dir);
  pFile = gzopen(buffer, "r");

  if (pFile == NULL) CkAbort("\nUnable to open model files.\n");
  for (int k = 0; k < NUM_TREES; k++) {
    int classifierID, r1, r2;
    double val;
    std::vector<double> w(3);

    int w_i;
    gzgets(pFile, linebuffer, LINE_SIZE);
    sscanf(linebuffer, "%d", &w_i);

    int X, D;

    X = treeModels[w_i - 1].l_X;
    D = treeModels[w_i - 1].l_D;

    treeModels[w_i - 1].weakModels.resize(X * D);

    for (int j = 0; j < X - 1; j++) {
      gzgets(pFile, linebuffer, LINE_SIZE);
      sscanf(linebuffer, "%lf", &val);
      classifierID = (int)val;
      if (classifierID != 0) {
        double val1, val2;
        sscanf(linebuffer, "%lf %lf %lf %lf %lf %lf", &val, &val1, &val2, &w[0], &w[1],
               &w[2]);
        r1 = (int)val1;
        r2 = (int)val2;
      }

#ifdef DEBUG
      if (classifierID != 0)
        printf("\nclassifier = %d %d %d %lf %lf %lf \n", classifierID, r1, r2, w[0], w[1],
               w[2]);
#endif

      treeModels[w_i - 1].weakModels[j].classifierID = classifierID;
      treeModels[w_i - 1].weakModels[j].r1 = r1;
      treeModels[w_i - 1].weakModels[j].r2 = r2;
      treeModels[w_i - 1].weakModels[j].w = w;
    }
  }

  gzclose(pFile);
#endif

  return;
}
