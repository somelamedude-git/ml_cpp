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

typedef struct Testing{
	double inputs[LAYER_ONE]; 
	double outputs[LAYER_FOUR]; // Optional to fill
	double output_predicted[LAYER_FOUR];
	double accuracy; // Optional
} Testing;

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

void sigmoid(double x){
	x = 1.0/(1.0 + exp(-x));
}

void sigmoid_derivative(double x){
	x = exp(-x)/((1+exp(-x)) *(1+ exp(-x)));
}

void sigmoid_list(double x[LAYER_ONE], int list_size){
	for(int i =0; i<list_size; i++){
		x[i] = sigmoid(x[i]);
	}
}

void sigmoid_derivative_list(double x[LAYER_ONE], int list_size){
	for(int i =0; i<list_size; i++){
		x[i] = sigmoid_derivative(x[i]);
	}
}

void matrix_multiplication(double matrix_one[LAYER_ONE][LAYER_ONE], double matrix_two[LAYER_ONE][LAYER_ONE], double product_matrix[LAYER_ONE][LAYER_ONE], int row_one,
		int col_one, int row_two, int col_two){
	for(int i =0; i<row_one; i++){
		for(int j =0; j<col_two; j++){
			for(int k =0; k<col_one; k++){
				product_matrix[i][k] += matrix_one[i][j]*matrix_two[j][k];
			}
		}
	}
}

void feed_forward(Network net, Testing data){
       int size_of_input = net.neurons_per_layer[0];
       for(int i = 0; i < size_of_input; i++){
                if(net.activations[0][i] == NULL || activation_input[i] == NULL){
                        continue;
                }
                net.activations[0][i][0] = activation_input[i][0];
        }

       int max_neurons = net.neurons_per_layer[0];
        for(int i = 0; i < net.num_of_layers; i++){
                if(max_neurons < net.neurons_per_layer[i]) max_neurons = net.neurons_per_layer[i];
        }

	double product_matrix[max_neurons][1];
	double addition_matrix[max_neurons][1];
	double activated_output[max_neurons][1];

	 if(product_matrix == NULL || addition_matrix == NULL || activated_output == NULL){
                return;
        }

	 for(int i =0; i<net.num_of_layers-1; i++){
		 int weight_rows = net.neurons_per_layer[i+1];
		 int weight_cols = net.neurons_per_layer[i];

		 
