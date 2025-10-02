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

void sigmoid(double x, double sig){
	sig = 1.0/(1.0 + exp(-x));
}

void sigmoid_derivative(double x, double derivative){
	double derivative = exp(-x)/((1+exp(-x))*(1+exp(-x)));

	return derivative;
}

void sigmoid_list(int x_list[], int size_of_list, int activated_array[]){
	for(int i =0; i<size_of_list; i++){
		int x = 0;
		sigmoid(x_list[i], x);
		activated_array[i] = x;
	}
}

void sigmoid_derivative_list(int x_list[], int size_of_list, int derivative_array[]){
	for(int i =0; i<size_of_list; i++){
		int x =0;
		sigmoid_derivative(x_list[i], x);
		derivative_array[i] = x;
	}
}




