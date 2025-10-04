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
	double*** weights;
	double*** biases;
	double*** activations;
	double*** z_s;
	double*** nabla_b;
	double*** nabla_w;
} Network;

Network initialize_network(int num_layers, int neuron_per_layer[]){
	Network net = {0};
	net.num_of_layers = num_layers;

	for(int i=0; i<num_layers; i++){
		net.neurons_per_layer[i] = neuron_per_layer[i];
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

void sigmoid_list(double** x_list, int size_of_list, double** activated_array){
	for(int i =0; i<size_of_list; i++){
		activated_array[i][0] = sigmoid(x_list[i][0]);
	}
}

void sigmoid_derivative_list(double** x_list, int size_of_list, double** derivative_array){
	for(int i =0; i<size_of_list; i++){
		derivative_array[i][0] = sigmoid_derivative(x_list[i][0]);
	}
}

void transpose(double** arr, int rows, int cols, double** transpose_arr){
	for(int i =0; i<rows; i++){
		for(int j =0; j<cols; j++){
			transpose_arr[j][i] = arr[i][j];
		}
	}
}

void transpose_singleD(double* arr, int length, double** transpose_arr){
	for(int i =0; i<length; i++){
		transpose_arr[i][0] = arr[i];
	}
}

void transpose_twoD(double arr[][1], int length, double transpose_arr[]){
	for(int i =0; i<length; i++){
		transpose_arr[i] = arr[i][0];
	}
}

double** create_matrix(int rows, int cols){
	double** matrix = (double**)malloc(sizeof(double*) * rows);
	for(int i =0; i<cols; i++){
		matrix[i] = calloc(rows, sizeof(double));
	}
	return matrix;
}

void matrix_multiplication(
    double** matrix_one,
    double** matrix_two,
    double** product_matrix,
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

void matrix_addition(double** matrix_one, double** matrix_two, int rows, int cols,  double** addition_matrix){
	for(int i =0; i<rows; i++){
		for(int j = 0; j<cols; j++){
			addition_matrix[i][j] = matrix_one[i][j] + matrix_two[i][j];
		}
	}
}

void feed_forward(Network net, double** activation_input){ // input has to be copied either way, no point of taking a transpose
	int size_of_input = net.neurons_per_layer[0];
	for(int i =0; i<size_of_input; i++){
		net.activations[0][i][0] = activation_input[i][0];
	}

	for(int i =0; i<net.num_of_layers-1; i++){
		int weight_rows = net.neurons_per_layer[i];
		int weight_cols = net.neurons_per_layer[i+1];

		double** product_matrix= create_matrix(weight_cols, 1);

		 matrix_multiplication(net.weights[i], activation_input, product_matrix,  weight_cols, weight_rows, size_of_input, 1);
		 double** z = create_matrix(weight_cols,1);

		 for(int j = 0; j<weight_cols; j++){
			 z[j][0] = product_matrix[j][0];
		 }
		
		double** addition_matrix = create_matrix(weight_cols, 1);
		matrix_addition(net.biases[i], z, weight_cols, 1, addition_matrix);
		
		for(int j =0; j<weight_cols; j++){
			z[j][0] = addition_matrix[j][0];
		}

		for(int j =0; j<weight_cols; j++){
			net.z_s[i][j][0] = z[j][0];
		}

		double activated_output[weight_cols][1];
		sigmoid_list(z, weight_cols, activated_output);

		for(int j =0; j<weight_cols; j++){
			activation_input[j][0] = activated_output[j][0];
			net.activations[i+1][j][0] = activated_output[j][0];
		}

		size_of_input = weight_cols;
	}
}

void matrix_subt(double** matrix_one, double** matrix_two, int rows, int cols, double** difference_matrix){
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

void hadamard_matrix_mult(double** matrix_one, double** matrix_two, int rows, int cols, double** product_matrix){
	for(int i =0; i<rows; i++){
		for(int j =0; j<cols; j++){
			product_matrix[i][j] = matrix_one[i][j] * matrix_two[i][j];
		}
	}
}
	
void backprop(Network net, double input[][1], double output[][1], int input_size, int output_size){
	feed_forward(net, input);
	int num_neurons = net.neurons_per_layer[net.num_of_layers-1];

	double curr_z[num_neurons][1];

	for(int j =0; j<num_neurons; j++){
		curr_z[j][0] = net.z_s[net.num_of_layers-2][j][0];
	}

	double derivative_array[num_neurons][1];
	sigmoid_derivative_list(curr_z, num_neurons, derivative_array);

	// derivative array is one of the main things, don't fuck it up
	
	double output_predicted[num_neurons][1];
	for(int j =0; j<num_neurons; j++){
		output_predicted[j][0] = net.activations[net.num_of_layers-1][j][0];
	}

	double derivative_matrix[num_of_neurons][1];
	compute_cost_derivative(net, output, output_predicted, derivative_matrix);

	double sigma[num_neurons][1];
	hadamard_matrix_mult(derivative_matrix, derivative_array, num_neurons, 1, sigma);

	for(int j =0; j<num_of_neurons; j++){
		net.nabla_b[net.num_of_layers-2][j][0] = sigma[j][0];
	}

	num_neurons = net.neurons_per_layer[net.num_of_layers-2];

	double transpose_act_layer[num_neurons];
	transpose_twoD(net.activations[net.num_of_layers-2], num_neurons, transpose_act_layer);

	double nabla_w[net.neurons_per_layer[net.num_of_layers-1]][net.neurons_per_layer[net.num_of_layers-2]] = {0};
	matrix_multiplication(sigma, transpose_act_layer, nabla_w, net.neurons_per_layer[net.num_of_layers-1], 1, 1, num_neurons);

	for(int i =0; i<net.neurons_per_layer[net.num_of_layers-1]; i++){
		for(int j =0; j<num_neurons; j++){
			net.nabla_w[net.num_of_layers-2][i][j] = nabla_w[i][j];
		}
	}
}



	

	
	
	



