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
	double sig = 1.0/(1.0 + exp(-x));
	return sig;
}

double sigmoid_derivative(double x){
	double derivative = exp(-x)/((1+exp(-x))*(1+exp(-x)));

	return derivative;
}







