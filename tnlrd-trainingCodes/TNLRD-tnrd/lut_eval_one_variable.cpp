//LUT_EVAL - Evaluate RBFMIX function value and gradients via lookup table with linear interpolation.
//
//   Compiling:
//    Linux:   mex lut_eval.c CFLAGS="\$CFLAGS -Wall -std=c99 -mtune=native -O3 -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
//    Windows: mex lut_eval.c COMPFLAGS="$COMPFLAGS /Wall /TP" OPTIMFLAGS="$OPTIMFLAGS /openmp /O2"
//
//   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
//
//   This file is part of the implementation as described in the CVPR 2014 paper:
//   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
//   Please see the file LICENSE.txt for the license governing this code.

#include "mex.h"
#include <math.h>
#include <omp.h>

void lookup(const int nargout, double *outP, const double *x, const double origin, const double step, const double *P, const int ndata, const int nbins) {
  omp_set_num_threads(64);
  #pragma omp parallel for
  for (int i = 0; i < ndata; i++) {

    const double x_hit = (x[i] - origin) / step;
    int xl = (int) floor(x_hit), xh = (int) ceil(x_hit);

    // boundary checks
    if (xl < 0) xl = 0; if (xl >= nbins) xl = nbins-1;
    if (xh < 0) xh = 0; if (xh >= nbins) xh = nbins-1;    
    const double wh = x_hit - (double)xl;

    const double pl = P[xl], ph = P[xh];
    outP[i] = pl + (ph-pl) * wh;
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  if (nlhs == 0) return;
  if (nrhs != 4) mexErrMsgTxt("Seven inputs expected: lut_eval(x, origin, step, P)");
 
  const int ndata     = mxGetNumberOfElements(prhs[0]);
  const int nbins     = mxGetNumberOfElements(prhs[3]);
  const double *data  = (const double*) mxGetPr(prhs[0]);
  const double origin = mxGetScalar(prhs[1]);
  const double step   = mxGetScalar(prhs[2]);
  const double *P     = (const double*) mxGetPr(prhs[3]);

  double *outP = NULL;
  plhs[0] = mxCreateDoubleMatrix(1, ndata, mxREAL);
  outP = (double*) mxGetPr(plhs[0]);
  
  lookup(nlhs, outP, data, origin, step, P, ndata, nbins);
}