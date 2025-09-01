#include <stdio.h>
#define MAX 100

double* find_matrix_mean(int size, int num_of_features , double** vars){
	double* mean_matrix = (double*)malloc(sizeof(double)*num_of_features);
	for(int j =0; j<num_of_features; j++){
		double mean = 0;
		for(int i =0; i<size; i++){
			mean += vars[i][j];
		}
		mean = mean/size;
		mean_matrix[j] = mean;
	}

	return mean_matrix;
}

double** transpose(double** matrix, int rows, int cols){
	double** transposed_matrix = (double**)malloc(sizeof(double*)*cols);
	for(int i =0; i<cols; i++){
		transposed_matrix[i] = (double*)malloc(sizeof(double)*rows);
	}

	for(int i =0; i<rows; i++){
		for(int j =0; j<cols; j++){
			transposed_matrix[j][i] = matrix[i][j];
		}
	}

	return transposed_matrix;
}



