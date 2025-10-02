#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>

#define SIZE 10
#define MAX_NUM_LAYERS 4

typedef struct Network{
	int neurons_per_layer[SIZE];
	int num_of_layers;
	int weights[MAX_NUM_LAYERS-1][SIZE][SIZE];
	int biases[MAX_NUM_LAYERS-1][SIZE][1];
} Network;

Network initialize_network(int num_layers, int[] neuron_per_layer){
	Network net = {0};
	net.num_of_layers = num_layers;
	net.neuron_per_layer = neuron_per_layer;

	return net;
}

double sigmoid(double x){
double	sig = 1.0/(1.0 + exp(-x));
	return sig;
}

double sigmoid_derivative(double x){
	double derivative = exp(-x)/((1+exp(-x))*(1+exp(-x)));

	return derivative;
}

void sigmoid_list(double x_list[], int size_of_list, double activated_array[]){
	for(int i =0; i<size_of_list; i++){
		activated_array[i] = sigmoid(x_list[i]);
	}
}

void sigmoid_derivative_list(double x_list[], int size_of_list, double derivative_array[]){
	for(int i =0; i<size_of_list; i++){
		derivative_array[i] = sigmoid_derivative(x_list[i]);
	}
}

void transpose(double arr[][SIZE], int rows, int cols, double transpose_arr[][SIZE]){
	for(int i =0; i<rows; i++){
		for(int j =0; j<cols; j++){
			transpose_arr[j][i] = arr[i][j];
		}
	}
}

void transpose_singleD(double arr[], int length, double transpose_arr[][SIZE]){
	for(int i =0; i<length; i++){
		transpose_arr[0][i] = arr[i];
	}
}




