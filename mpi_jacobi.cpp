/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

#include <iostream>

/*
 * TODO: Implement your solutions here
 */
int size = -1;
int rank = -1;
int q = -1;
int coords[2] = {-1,-1};
int rank00 = -1;
unsigned row_size = 0;
unsigned col_size = 0;


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    // TODO

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);


    MPI_Cart_coords(comm, rank, 2, coords);
	

    int coords00[2] = {0,0};
    MPI_Cart_rank(comm, coords00, &rank00);

    q = sqrt(size);


    int firstcolumn = 0;
    if(coords[1] == 0) firstcolumn = 1;

    MPI_Comm firstcolumn_comm;
    MPI_Comm_split(comm, firstcolumn, coords[0], &firstcolumn_comm);

    
    if(coords[0] < (n%q)) row_size = ceil(double(n)/double(q));
    else row_size = floor(double(n)/double(q));
	
    if (firstcolumn == 1){

	std::vector<int> send_counts = std::vector<int>(q,0);
	std::vector<int> disps = std::vector<int>(q,0);
	
	if (rank == rank00){
		for (int i = 0; i< q; i++){
			if(i < (n%q)) send_counts[i] = ceil(double(n)/double(q));
			else send_counts[i] = floor(double(n)/double(q));
		}

		disps[0] = 0;
		for (int ii = 1; ii<q; ii++) disps[ii] = disps[ii-1]+ send_counts[ii-1];
	}
	
	*local_vector = new double[row_size];

	MPI_Scatterv(input_vector, &send_counts[0], &disps[0], MPI_DOUBLE, 
		      *local_vector, row_size, MPI_DOUBLE, 0, firstcolumn_comm);
    } 
    
    MPI_Comm_free(&firstcolumn_comm);


}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    // TODO

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    q = sqrt(size);

    MPI_Cart_coords(comm, rank, 2, coords);
	

    int coords00[2] = {0,0};
    MPI_Cart_rank(comm, coords00, &rank00);

    int firstcolumn = 0;
    if(coords[1] == 0) firstcolumn = 1;

    MPI_Comm firstcolumn_comm;
    MPI_Comm_split(comm, firstcolumn, coords[0], &firstcolumn_comm);

    //int firstcol_rank;
    //MPI_Comm_rank(firstcolumn_comm, &firstcol_rank);
	

    if(coords[0] < (n%q)) row_size = ceil(double(n)/double(q));
    else row_size = floor(double(n)/double(q));

    if(firstcolumn == 1){
      std::vector<int> disps = std::vector<int>(q, 0);
      std::vector<int> recv_counts = std::vector<int>(q, 0);
    
      if(rank == rank00){
	//Calculate recv_counts and disps
	for (int i = 0; i < q; i++){
	    if(i < (n%q)) recv_counts[i] = ceil(double(n)/double(q));
	    else recv_counts[i] = floor(double(n)/double(q));
	}

	disps[0] = 0;
	for (int ii = 1; ii < q; ii++) 
		disps[ii] = disps[ii-1]+ recv_counts[ii-1];
      }

      MPI_Gatherv(local_vector, row_size, MPI_DOUBLE, output_vector, &recv_counts[0],
	      &disps[0], MPI_DOUBLE, 0, firstcolumn_comm);

    }
    MPI_Comm_free(&firstcolumn_comm);

}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

  
    MPI_Cart_coords(comm, rank, 2, coords);
	

    int coords00[2] = {0,0};
    MPI_Cart_rank(comm, coords00, &rank00);
    
    q = sqrt(size);

    if(coords[0] < (n%q)){
	row_size = ceil(double(n)/double(q));
	if(coords[1] < (n%q)) col_size = ceil(double(n)/double(q));
	else col_size = floor(double(n)/double(q));
    } 
    else{ 
	row_size = floor(double(n)/double(q));
	if(coords[1] < (n%q)) col_size = ceil(double(n)/double(q));
	else col_size = floor(double(n)/double(q));
    }
    

    
    MPI_Comm comm_inOrder;
    MPI_Comm_split(comm, 0, coords[0]*q + coords[1], &comm_inOrder);
    
    double* send_buffer = NULL;
    std::vector<int> send_counts = std::vector<int>(size, 0);
    std::vector<int> disps = std::vector<int>(size, 0);
    
    if(rank == rank00){
	unsigned input_matrix_idx = 0;
	unsigned row_marking = 0;
	unsigned col_marking = 0;
	send_buffer = new double[n*n];
	
	unsigned local_row = 0;
	unsigned local_col = 0;
	
	for(int i=0; i<q; i++){ //i represents ith row of processor grid
	    if(i < (n%q)) local_row = ceil(double(n)/double(q));
	    else local_row = floor(double(n)/double(q));
	    
	    col_marking = 0;
	    for(int j=0; j<q; j++){//j represents jth column of processor grid
		if(j < (n%q)) local_col = ceil(double(n)/double(q));
		else local_col = floor(double(n)/double(q));		
		

		for(unsigned k=0; k<local_row; k++){
		    memcpy(&send_buffer[input_matrix_idx], &input_matrix[(row_marking+k)*n + col_marking], local_col*sizeof(double));
		    input_matrix_idx = input_matrix_idx + local_col;
		}
	      
		col_marking = col_marking + local_col;
		
		send_counts[i*q + j] = local_row * local_col;
		
		disps[0] = 0;
		if((i*q + j) > 0) disps[i*q + j] = disps[i*q + j -1] + send_counts[i*q + j -1];
		
	    }
	    
	    row_marking = row_marking + local_row;
	}

    }
    
    int recv_counts = row_size*col_size;
    *local_matrix = new double[recv_counts];
    MPI_Scatterv(send_buffer, &send_counts[0], &disps[0], MPI_DOUBLE, *local_matrix, recv_counts, MPI_DOUBLE, 0, comm_inOrder);
     

	if (send_buffer != NULL)
        delete[] send_buffer;
    MPI_Comm_free(&comm_inOrder);

}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // TODO
	//std::cout <<col_vector<< std::endl;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    
    MPI_Cart_coords(comm, rank, 2, coords);
	
    int coords00[2] = {0,0};
    MPI_Cart_rank(comm, coords00, &rank00);
    
    q = sqrt(size);

    
    if(coords[0] < (n%q)){
	row_size = ceil(double(n)/double(q));
	if(coords[1] < (n%q)) col_size = ceil(double(n)/double(q));
	else col_size = floor(double(n)/double(q));
    } 
    else{ 
	row_size = floor(double(n)/double(q));
	if(coords[1] < (n%q)) col_size = ceil(double(n)/double(q));
	else col_size = floor(double(n)/double(q));
    }

    if (coords[0] == 0 && coords[1] == 0){
	memcpy(row_vector, col_vector, row_size*sizeof(double));
    }else if (coords[1] == 0){ //if (i,0), sends to (i,i)
	int dest_coord[2] = {coords[0], coords[0]};
        int dest_rank = 0;
	MPI_Cart_rank(comm, dest_coord, &dest_rank);
		//std::cout << dest_rank << std::endl;
        MPI_Send(col_vector, row_size, MPI_DOUBLE, dest_rank, 0, comm);

    }else if (coords[0] == coords[1]){ //if (i,i), receive from (i,0)
	int src_coord[2] = {coords[0], 0};
	int src_rank = 0;
	MPI_Cart_rank(comm, src_coord, &src_rank);
        //std::cout << src_rank << std::endl;
	MPI_Status status;
        MPI_Recv(row_vector, col_size, MPI_DOUBLE, src_rank, 0, comm, &status);
    }

    // creat column sub-communicator
    MPI_Comm column_comm;
    MPI_Comm_split(comm, coords[1], coords[0], &column_comm);

    //int root_coords[2] = {coords[0],coords[0]};
    //int root_rank;
    //MPI_Cart_rank(column_comm, root_coords, &root_rank);
    MPI_Bcast(row_vector, col_size, MPI_DOUBLE, coords[1], column_comm);

    MPI_Comm_free(&column_comm);
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO


    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);


    MPI_Cart_coords(comm, rank, 2, coords);
	

    int coords00[2] = {0,0};
    MPI_Cart_rank(comm, coords00, &rank00);
    
    q = sqrt(size);

    if(coords[0] < (n%q)){
	row_size = ceil(double(n)/double(q));
	if(coords[1] < (n%q)) col_size = ceil(double(n)/double(q));
	else col_size = floor(double(n)/double(q));
    } 
    else{ 
	row_size = floor(double(n)/double(q));
	if(coords[1] < (n%q)) col_size = ceil(double(n)/double(q));
	else col_size = floor(double(n)/double(q));
    }


    std::vector<double> row_vector = std::vector<double>(col_size, 0);
    
    transpose_bcast_vector(n, local_x, &row_vector[0], comm);

    std::vector<double> temp_y = std::vector<double>(row_size, 0);
    for (unsigned i=0; i < row_size; i++){
	local_y[i] = 0;
        for (unsigned j=0; j < col_size; j++){
            local_y[i] = local_y[i] + local_A[i*col_size + j]*row_vector[j];
        }
    }

    // creat column sub-communicator
    //int remain[2] = {1,0};
    //MPI_Comm row_comm;
    //MPI_Cart_sub(comm, remain, &row_comm);

    MPI_Comm row_comm;
    MPI_Comm_split(comm, coords[0], coords[1], &row_comm);

    memcpy(&temp_y[0], local_y, row_size*sizeof(double));
    //int root_coords[2] = {coords[0],0};
    //int root_rank;
    //MPI_Cart_rank(row_comm, root_coords, &root_rank);
    MPI_Reduce(&temp_y[0], local_y, row_size, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    MPI_Comm_free(&row_comm);
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // TODO

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);


    MPI_Cart_coords(comm, rank, 2, coords);
    

    int coord00[2] = {0,0};
    MPI_Cart_rank(comm, coord00, &rank00);
    q = sqrt(size);

    if(coords[0] < (n%q)){
	row_size = ceil(double(n)/double(q));
	if(coords[1] < (n%q)) col_size = ceil(double(n)/double(q));
	else col_size = floor(double(n)/double(q));
    } 
    else{ 
	row_size = floor(double(n)/double(q));
	if(coords[1] < (n%q)) col_size = ceil(double(n)/double(q));
	else col_size = floor(double(n)/double(q));
    }

    // Collect local_A at diagnal to first column (i,0)
    std::vector<double> local_D = std::vector<double>(row_size, 0);
    // Copy A and set diagonal to zero
    std::vector<double> local_R = std::vector<double>(row_size*col_size);
    memcpy(&local_R[0], local_A, row_size*col_size*sizeof(double));
    
    
    
    // Init x to zero
    if(coords[1] == 0) {
      for (unsigned i = 0; i < row_size; ++i) local_x[i] = 0;
    }
    
    if(coords[0] == 0 && coords[1] ==0 ){
      for(unsigned i=0; i<row_size; i++){
	local_D[i] = local_A[i*col_size + i];
	local_R[i*col_size + i] = 0;
      }
    }else if(coords[0] == coords[1]){
	for(unsigned i = 0; i < row_size; i++) {
	  local_D[i] = local_A[i*col_size + i];
	  local_R[i*col_size + i] = 0;
	}
	int dest_coord[2] = {coords[0], 0};
        int dest_rank =0;
	MPI_Cart_rank(comm, dest_coord, &dest_rank);

		//int dest;
        //dest = MPI_Cart_shift(comm, 0, -coords[1], &rank, &dest);
        MPI_Send(&local_D[0], row_size, MPI_DOUBLE, dest_rank, 101, comm);
    }else if (coords[1] == 0){
        int src_coord[2] = {coords[0], coords[0]};
	int src_rank;
	MPI_Cart_rank(comm, src_coord, &src_rank);
		
		//int source;
        //source = MPI_Cart_shift(comm, 0, coords[1], &rank, &source);

	MPI_Status status;
        MPI_Recv(&local_D[0], row_size, MPI_DOUBLE, src_rank, 101, comm, &status);
    }


	
    int firstcolumn = 0;
    if(coords[1] == 0) firstcolumn = 1;

    MPI_Comm firstcolumn_comm;
    MPI_Comm_split(comm, firstcolumn, coords[0], &firstcolumn_comm);

    MPI_Comm row_comm;
    MPI_Comm_split(comm, coords[0], coords[1], &row_comm);

    std::vector<double> local_P = std::vector<double>(row_size, 0);
    std::vector<double> local_w = std::vector<double>(row_size, 0);
    double l2_square = 0;
    double l2_sum = 0;
    int breakFlag = 0;


    for (int i=0; i<max_iter; i++){
	distributed_matrix_vector_mult(n, &local_R[0], local_x, &local_P[0], comm); //P=Rx
	
	if(coords[1] == 0){
	    for (unsigned j=0; j<row_size; j++){
		local_x[j] = (local_b[j] - local_P[j])/local_D[j];
	    }
	}
	
	distributed_matrix_vector_mult(n, local_A, local_x, &local_w[0], comm); //w=Ax
	    
	if(coords[1] == 0){
	    for (unsigned j = 0; j < row_size; j++){
		  l2_square = l2_square + (local_b[j] - local_w[j])*(local_b[j] - local_w[j]);
	    }
	    
	    MPI_Allreduce(&l2_square, &l2_sum, 1, MPI_DOUBLE, MPI_SUM, firstcolumn_comm);
	    l2_sum = sqrt(l2_sum);
	    
	    if (l2_sum <= l2_termination) breakFlag = 1;	
	    
	}
	
	MPI_Bcast(&breakFlag, 1, MPI_INT, 0, row_comm);
    

	if (breakFlag == 1) break; 
	
    }
	MPI_Comm_free(&firstcolumn_comm);
	MPI_Comm_free(&row_comm);

}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);

	if (local_y != NULL)
        delete[] local_y;

    if (local_A != NULL)
        delete[] local_A;
    if (local_x != NULL)
        delete[] local_x;
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
	
	if (local_x != NULL)
        delete[] local_x;

    if (local_A != NULL)
        delete[] local_A;
    if (local_b != NULL)
        delete[] local_b;
}
