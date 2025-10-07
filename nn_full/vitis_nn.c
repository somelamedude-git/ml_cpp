#include <stdio.h>
#include <math.h>

#define LAYER_ONE 784
#define LAYER_TWO 128
#define LAYER_THREE 64
#define LAYER_FOUR 10
#define number_of_layers 4

typedef struct Network{
	int neurons_per_layer[4];
	int num_of_layers;
	double weights[number_of_layers-1][LAYER_ONE][LAYER_ONE]; // Assigning the max value of shape
	double biases[number_of_layers-1][LAYER_ONE][1];
	double activations[number_of_layers][LAYER_ONE][1];
	double z_s[number_of_layers-1][LAYER_TWO][1];
} Network;

void fill_network(Network net,int neurons_per_layer[num_of_layers], double weights[number_of_layers-1][LAYER_ONE][LAYER_ONE], double biases[number_of_layers-1][LAYER_ONE][1]){
	net.num_of_layers = number_of_layers;

	for(int i =0; i<num_of_layers; i++){
		net.neurons_per_layer[i] = neurons_per_layer[i];
	}

	for(int k =0; k<num_of_layers-1; k++){
		for(int i =0; i<net.neurons_per_layer[k+1]; i++){
			net.biases[k][i][0] = biases[k][i][0];
			for(int j =0; j<net.neurons_per_layer[k]; j++){
				net.weights[k][i][j] = weights[k][i][j];
			}
		}
	}
}

void feed_forward(Network net)
