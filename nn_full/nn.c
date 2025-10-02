#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>

#define SIZE 10
#define MAX_NUM_LAYERS 4

typedef struct Network{
	int neurons_per_layer[MAX_NUM_LAYERS];
	int num_of_layers;
	double weights[MAX_NUM_LAYERS-1][SIZE][SIZE];
	double biases[MAX_NUM_LAYERS-1][SIZE][1];
	double activations[MAX_NUM_LAYERS][SIZE][1];
	double z_s[MAX_NUM_LAYERS-1][SIZE][1];
} Network;

Network initialize_network(int num_layers, int neuron_per_layer[]){
	Network net = {0};
	net.num_of_layers = num_layers;

	for(int i=0; i<num_layers; i++){
		net.neuron_per_layer[i] = neuron_per_layer[i];
	}

	for(int i=0; i<net.num_of_layers; i++){
		int curr_neurons = net.neurons_per_layer[i];
		for(int j = 0; j<curr_neurons; j++){
			net.activations[i][j][0] = 0;
		}
	}

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

void sigmoid_list(double x_list[][SIZE], int size_of_list, double activated_array[]){
	for(int i =0; i<size_of_list; i++){
		activated_array[i][0] = sigmoid(x_list[i][0]);
	}
}

void sigmoid_derivative_list(double x_list[][SIZE], int size_of_list, double derivative_array[]){
	for(int i =0; i<size_of_list; i++){
		derivative_array[i][0] = sigmoid_derivative(x_list[i][0]);
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
		transpose_arr[i][0] = arr[i];
	}
}

void matrix_multiplication(
    float matrix_one[][SIZE],
    float matrix_two[][SIZE],
    float product_matrix[][SIZE],
    int row_one, int col_one, int row_two, int col_two
) {
    if (col_one != row_two) {
        printf("Error: col_one must equal row_two\n");
        return;
    }

    for(int i = 0; i < row_one; i++)
        for(int k = 0; k < col_two; k++)
            product_matrix[i][k] = 0;

    for(int i = 0; i < row_one; i++) {
        for(int k = 0; k < col_two; k++) {
            for(int j = 0; j < col_one; j++) {
                product_matrix[i][k] += matrix_one[i][j] * matrix_two[j][k];
            }
        }
    }
}

void matrix_addition(double matrix_one[][SIZE], double matrix_two[][SIZE], int rows, int cols,  double addition_matrix[][SIZE]){
	for(int i =0; i<rows; i++){
		for(int j = 0; j<cols; j++){
			addition_matrix[i][j] = matrix_one[i][j] + matrix_two[i][j];
		}
	}
}


void feed_forward(Network net, double activation_input[][1]){ // input has to be copied either way, no point of taking a transpose
	int size_of_input = net.neurons_per_layer[0];
	for(int i =0; i<size_of_input; i++){
		net.activations[0][i][0] = input[i][0];
	}

	for(int i =0; i<net.num_of_layers-1; i++){
		int weight_rows = net.neurons_per_layer[i];
		int weight_cols = net.neurons_per_layer[i+1];

		double product_matrix[weight_cols][1] = {0}; 

		 matrix_multiplication(net.weights[i], activation_input, product_matrix,  weight_cols, weight_rows, size_of_input, 1,);
		 z = product_matrix;
		
		double addition_matrix[weight_cols][1] = {0};
		matrix_addition(bias[i], z, weight_cols, 1, addition_matrix);
		z = addition_matrix;

		for(int j =0; j<weight_cols; j++){
			net.z_s[i][j][0] = z[j][0];
		}

		double activated_output[weight_cols][1] = {0};
		sigmoid_list(z, weight_cols, activated_output);

		for(int j =0; j<weight_cols; j++){
			activation_input[j][0] = activated_output[j][0];
			net.activations[i+1][j][0] = activated_output[j][0];
		}

		size_of_input = weight_cols;
	}
}

void matrix_subt(double matrix_one[][SIZE], double matrix_two[][SIZE], int rows, int cols, double difference_matrix[][SIZE]){
	for(int i =0; i<rows; i++){
		for(int j =0; j<cols; j++){
			difference_matrix[i][j] = matrix_one[i][j] - matrix_two[i][j];
		}
	}
}


void compute_cost_derivative(Network net, double expected_output[][1], double output[][1], double derivative_matrix[][1]){
	int rows = net.neurons_per_layer[net.num_of_layers-1];
	matrix_subt(output, expected_output, rows, 1, derivative_matrix);
}

	
void backprop(Network net, double input[][1], double output[][1], int input_size, int output_size){
	feed_forward(net, input);

	
	
	



