/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : 
 * Module      : 
 * Class       : $RCSfile$
 * Language    : C++
 * Description : 
 *
 * Author     : Yunjin Chen
 * EMail      : cheny@icg.tugraz.at
 *
 */
#include "mex.h"
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <string.h>
///////////////////////////////////////////////////////////
// compile with: mex CXXFLAGS="\$CXXFLAGS -fopenmp -O3" LDFLAGS="\$LDFLAGS -fopenmp" convolution_transposeOMP.cpp
///////////////////////////////////////////////////////////
// entry function
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
    double *input,*output,*k,*filter;
	int filter_size,nRow,nCol,N;
	if (nrhs !=4 ||  nlhs!=1)
		mexErrMsgTxt("Invalid number of input/output arguments!");
	input = mxGetPr(prhs[0]);
	k = mxGetPr(prhs[1]);
	filter_size = mxGetM(prhs[1]);
	nRow = (int) mxGetScalar(prhs[2]);
	nCol = (int) mxGetScalar(prhs[3]);
	N = (filter_size - 1)/2;
	plhs[0] = mxCreateDoubleMatrix(nRow*nCol,1,mxREAL);
	output = mxGetPr(plhs[0]);
	if (filter_size != mxGetN(prhs[1]) || filter_size%2 != 1)
		mexErrMsgTxt("Invalid filter(only valid for square and odd size filter)!");
//	reverse filter
	filter = (double *)calloc(filter_size*filter_size,sizeof(double));
	for (int idx = 0; idx < filter_size*filter_size; idx++)
		filter[idx] = k[filter_size*filter_size - idx - 1];

	omp_set_num_threads(64);
	//handle central region
	int col,row;
	#pragma omp parallel for private (col, row)
	for (col = N; col < nCol-N; col++)
	{
		for (row = N; row < nRow-N; row++)
		{
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;
					response = response + input[idx_row + idx_col*nRow] * filter[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
	}
	// handle boudaryies
	//left side boundary
	#pragma omp parallel for private (col, row)
	for (col = 0; col < N; col++)
	{
		double *temp_filter = (double *)calloc(filter_size*filter_size,sizeof(double));
		for (int idx_col = 0; idx_col < N-col; idx_col++)
		{
			for (int idx_row = 0; idx_row < filter_size; idx_row++)
				temp_filter[idx_row + idx_col*filter_size] = filter[idx_row + (2*N-idx_col)*filter_size];
		}
		for (int idx_col = N-col; idx_col < filter_size; idx_col++)
		{
			for (int idx_row = 0; idx_row < filter_size; idx_row++)
				temp_filter[idx_row + idx_col*filter_size] = filter[idx_row + idx_col*filter_size];
		}
		for (row = N; row < nRow-N; row++)
		{
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;

					if (idx_col < 0)
						idx_col = -1 - idx_col;
					response = response + input[idx_row + idx_col*nRow] * temp_filter[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
		free(temp_filter); temp_filter = NULL;
	}
	//right side boundary
	#pragma omp parallel for private (col, row)
	for (col = nCol-N; col < nCol; col++)
	{
		double *temp_filter = (double *)calloc(filter_size*filter_size,sizeof(double));
		for (int idx_col = 0; idx_col < N-(nCol-col)+1; idx_col++)
		{
			for (int idx_row = 0; idx_row < filter_size; idx_row++)
				temp_filter[idx_row + (2*N-idx_col)*filter_size] = filter[idx_row + idx_col*filter_size];
		}
		for (int idx_col = N-(nCol-col)+1; idx_col < filter_size; idx_col++)
		{
			for (int idx_row = 0; idx_row < filter_size; idx_row++)
				temp_filter[idx_row + (2*N-idx_col)*filter_size] = filter[idx_row + (2*N-idx_col)*filter_size];
		}
		for (row = N; row < nRow-N; row++)
		{
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;

					if (idx_col >= nCol)
						idx_col = 2*nCol - (idx_col+1);
					response = response + input[idx_row + idx_col*nRow] * temp_filter[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
		free(temp_filter); temp_filter = NULL;
	}
	//top boundary
	#pragma omp parallel for private (col, row)
	for (row = 0; row < N; row++)
	{
		double *temp_filter = (double *)calloc(filter_size*filter_size,sizeof(double));
		for (int idx_row = 0; idx_row < N-row; idx_row++)
		{
			for (int idx_col = 0; idx_col < filter_size; idx_col++)
				temp_filter[idx_row + idx_col*filter_size] = filter[2*N - idx_row + idx_col*filter_size];
		}
		for (int idx_row = N-row; idx_row < filter_size; idx_row++)
		{
			for (int idx_col = 0; idx_col < filter_size; idx_col++)
				temp_filter[idx_row + idx_col*filter_size] = filter[idx_row + idx_col*filter_size];
		}
		for (col = N; col < nCol-N; col++)
		{
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;

					if (idx_row < 0)
						idx_row = -1 - idx_row;
					response = response + input[idx_row + idx_col*nRow] * temp_filter[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
		free(temp_filter); temp_filter = NULL;
	}
	//bottom boudanry
	#pragma omp parallel for private (col, row)
	for (row = nRow-N; row < nRow; row++)
	{
		double *temp_filter = (double *)calloc(filter_size*filter_size,sizeof(double));
		for (int idx_row = 0; idx_row < N-(nRow-row)+1; idx_row++)
		{
			for (int idx_col = 0; idx_col < filter_size; idx_col++)
				temp_filter[2*N - idx_row + idx_col*filter_size] = filter[idx_row + idx_col*filter_size];
		}
		for (int idx_row = N-(nRow-row)+1; idx_row < filter_size; idx_row++)
		{
			for (int idx_col = 0; idx_col < filter_size; idx_col++)
				temp_filter[2*N - idx_row + idx_col*filter_size] = filter[2*N - idx_row + idx_col*filter_size];
		}
		for (col = N; col < nCol-N; col++)
		{
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;

					if (idx_row >= nRow)
						idx_row = 2*nRow - (idx_row+1);
					response = response + input[idx_row + idx_col*nRow] * temp_filter[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
		free(temp_filter); temp_filter = NULL;
	}
	//handle left two blocks
	for (col = 0; col < N; col++)
	{
		double *temp_filter = (double *)mxCalloc(filter_size*filter_size,sizeof(double));
		for (int idx_col = 0; idx_col < N-col; idx_col++)
		{
			for (int idx_row = 0; idx_row < filter_size; idx_row++)
				temp_filter[idx_row + idx_col*filter_size] = filter[idx_row + (2*N-idx_col)*filter_size];
		}
		for (int idx_col = N-col; idx_col < filter_size; idx_col++)
		{
			for (int idx_row = 0; idx_row < filter_size; idx_row++)
				temp_filter[idx_row + idx_col*filter_size] = filter[idx_row + idx_col*filter_size];
		}
		//top block
		for (row = 0; row < N; row++)
		{
			double *temp_filter2 = (double *)mxCalloc(filter_size*filter_size,sizeof(double));
			for (int idx_row = 0; idx_row < N-row; idx_row++)
			{
				for (int idx_col = 0; idx_col < filter_size; idx_col++)
					temp_filter2[idx_row + idx_col*filter_size] = temp_filter[2*N - idx_row + idx_col*filter_size];
			}
			for (int idx_row = N-row; idx_row < filter_size; idx_row++)
			{
				for (int idx_col = 0; idx_col < filter_size; idx_col++)
					temp_filter2[idx_row + idx_col*filter_size] = temp_filter[idx_row + idx_col*filter_size];
			}
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;

					if (idx_col < 0)
						idx_col = -1 - idx_col;
					if (idx_row < 0)
						idx_row = -1 - idx_row;
					response = response + input[idx_row + idx_col*nRow] * temp_filter2[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
		//bottom block
		for (row = nRow-N; row < nRow; row++)
		{
			double *temp_filter2 = (double *)mxCalloc(filter_size*filter_size,sizeof(double));
			for (int idx_row = 0; idx_row < N-(nRow-row)+1; idx_row++)
			{
				for (int idx_col = 0; idx_col < filter_size; idx_col++)
					temp_filter2[2*N - idx_row + idx_col*filter_size] = temp_filter[idx_row + idx_col*filter_size];
			}
			for (int idx_row = N-(nRow-row)+1; idx_row < filter_size; idx_row++)
			{
				for (int idx_col = 0; idx_col < filter_size; idx_col++)
					temp_filter2[2*N - idx_row + idx_col*filter_size] = temp_filter[2*N - idx_row + idx_col*filter_size];
			}
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;

					if (idx_col < 0)
						idx_col = -1 - idx_col;
					if (idx_row >= nRow)
						idx_row = 2*nRow - (idx_row+1);
					response = response + input[idx_row + idx_col*nRow] * temp_filter2[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
	}
	//handle right two blocks
	for (col = nCol-N; col < nCol; col++)
	{
		double *temp_filter = (double *)mxCalloc(filter_size*filter_size,sizeof(double));
		for (int idx_col = 0; idx_col < N-(nCol-col)+1; idx_col++)
		{
			for (int idx_row = 0; idx_row < filter_size; idx_row++)
				temp_filter[idx_row + (2*N-idx_col)*filter_size] = filter[idx_row + idx_col*filter_size];
		}
		for (int idx_col = N-(nCol-col)+1; idx_col < filter_size; idx_col++)
		{
			for (int idx_row = 0; idx_row < filter_size; idx_row++)
				temp_filter[idx_row + (2*N-idx_col)*filter_size] = filter[idx_row + (2*N-idx_col)*filter_size];
		}
		//top block
		for (row = 0; row < N; row++)
		{
			double *temp_filter2 = (double *)mxCalloc(filter_size*filter_size,sizeof(double));
			for (int idx_row = 0; idx_row < N-row; idx_row++)
			{
				for (int idx_col = 0; idx_col < filter_size; idx_col++)
					temp_filter2[idx_row + idx_col*filter_size] = temp_filter[2*N - idx_row + idx_col*filter_size];
			}
			for (int idx_row = N-row; idx_row < filter_size; idx_row++)
			{
				for (int idx_col = 0; idx_col < filter_size; idx_col++)
					temp_filter2[idx_row + idx_col*filter_size] = temp_filter[idx_row + idx_col*filter_size];
			}
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;

					if (idx_col >= nCol)
						idx_col = 2*nCol - (idx_col+1);
					if (idx_row < 0)
						idx_row = -1 - idx_row;
					response = response + input[idx_row + idx_col*nRow] * temp_filter2[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
		//bottom block
		for (row = nRow-N; row < nRow; row++)
		{
			double *temp_filter2 = (double *)mxCalloc(filter_size*filter_size,sizeof(double));
			for (int idx_row = 0; idx_row < N-(nRow-row)+1; idx_row++)
			{
				for (int idx_col = 0; idx_col < filter_size; idx_col++)
					temp_filter2[2*N - idx_row + idx_col*filter_size] = temp_filter[idx_row + idx_col*filter_size];
			}
			for (int idx_row = N-(nRow-row)+1; idx_row < filter_size; idx_row++)
			{
				for (int idx_col = 0; idx_col < filter_size; idx_col++)
					temp_filter2[2*N - idx_row + idx_col*filter_size] = temp_filter[2*N - idx_row + idx_col*filter_size];
			}
			int idx = 0;
			int idx_col,idx_row;
			double response = 0;
			for (int filter_col = -N; filter_col<=N; filter_col++)
			{
				for (int filter_row = -N; filter_row<=N; filter_row++)
				{
					idx_col = col + filter_col;
					idx_row = row + filter_row;

					if (idx_col >= nCol)
						idx_col = 2*nCol - (idx_col+1);
					if (idx_row >= nRow)
						idx_row = 2*nRow - (idx_row+1);
					response = response + input[idx_row + idx_col*nRow] * temp_filter2[idx];
					idx++;
				}
			}
			output[row + col*nRow] = response;
		}
	}
	free(filter); filter = NULL;
}