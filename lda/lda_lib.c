#include <stdio.h>
#define MAX 100

double* find_matrix_mean(int size, int num_of_features , int vars[][MAX]){
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

double


