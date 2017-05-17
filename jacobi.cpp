/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <vector>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = 0;
        for (int j = 0; j < n; j++)
        {
            y[i] += A[i*n + j] * x[j];
        }
	}
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = 0;
        for (int j = 0; j < m; j++)
        {
            y[i] += A[i*m + j] * x[j];
        }
	}
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
	double norm;
    std::vector<double> D = std::vector<double>(n);
	std::vector<double> R = std::vector<double>(n*n);
	std::vector<double> temp_y = std::vector<double>(n, 0);
	
	for (int i = 0; i < n; i++) x[i] = 0;
	
	for(int i = 0; i < n; i++){
		D[i] = A[i*n + i];

		for(int j = 0; j < n; j++){
			if (i == j) 
				R[i*n + j] = 0;
			else
			    R[i*n + j] = A[i*n + j];
		}
	}

	for (int iter = 0; iter < max_iter; iter++){
		norm = 0;
		matrix_vector_mult(n, &R[0], x, &temp_y[0]);//temp_y=Rx&R[0]

		for (int j = 0; j < n; j++){
			x[j] = (b[j] - temp_y[j])/D[j];
		}

		matrix_vector_mult(n, A, x, &temp_y[0]);//temp_y=Ax

		for (int j = 0; j < n; j++){
			norm += pow((b[j] - temp_y[j]),2);
		}

		if (sqrt(norm) <= l2_termination)
			break;
	}    
}
