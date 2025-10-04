#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#define SIZE 10
#define MAX_NUM_LAYERS 4

typedef struct Network{
	int neurons_per_layer[MAX_NUM_LAYERS];
	int num_of_layers;
	double*** weights;
	double*** biases;
	double*** activations;
	double*** z_s;
	double*** delta_nabla_b;
	double*** delta_nabla_w;
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

double** create_matrix(int rows, int cols){
	double** matrix = (double**)malloc(sizeof(double*) * rows);
	for(int i =0; i<rows; i++){
		matrix[i] = calloc(cols, sizeof(double));
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

	int max_neurons = net.neurons_per_layer[0];
	for(int i =0; i<net.num_of_layers; i++){
		if(max_neurons < net.neurons_per_layer[i]) max_neurons = net.neurons_per_layer[i];
	}
	double** product_matrix = create_matrix(max_neurons, 1);
	double** addition_matrix = create_matrix(max_neurons, 1);
	double** activated_output = create_matrix(max_neurons, 1);
	for(int i =0; i<net.num_of_layers-1; i++){
		int weight_rows = net.neurons_per_layer[i];
		int weight_cols = net.neurons_per_layer[i+1];


		 matrix_multiplication(net.weights[i], activation_input, product_matrix,  weight_cols, weight_rows, size_of_input, 1);

		 for(int j = 0; j<weight_cols; j++){
			net. z_s[i][j][0] = product_matrix[j][0];
		 }
		
		matrix_addition(net.biases[i], product_matrix, weight_cols, 1, addition_matrix);
		
		for(int j =0; j<weight_cols; j++){
			net.z_s[i][j][0] = addition_matrix[j][0];
		}

		sigmoid_list(net.z_s[i], weight_cols, activated_output);

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


void compute_cost_derivative(Network net, double** expected_output, double** output, double** derivative_matrix){
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
	
void backprop(Network net, double** input, double** output){
	feed_forward(net, input);
	int num_neurons = net.neurons_per_layer[net.num_of_layers-1];

	double** derivative_array = create_matrix(num_neurons, 1);
	sigmoid_derivative_list(net.z_s[net.num_of_layers-2], num_neurons, derivative_array);

	// derivative array is one of the main things, don't fuck it up
	
	double** derivative_matrix = create_matrix(num_neurons, 1);
	compute_cost_derivative(net, output, net.activations[net.num_of_layers-1], derivative_matrix);

	double** sigma = create_matrix(num_neurons, 1);
	hadamard_matrix_mult(derivative_matrix, derivative_array, num_neurons, 1, sigma);

	for(int j =0; j<num_neurons; j++){
		net.delta_nabla_b[net.num_of_layers-2][j][0] = sigma[j][0];
	}

	num_neurons = net.neurons_per_layer[net.num_of_layers-2];

	double** transpose_act_layer = create_matrix(1, num_neurons);
	transpose(net.activations[net.num_of_layers-2], num_neurons, 1, transpose_act_layer);

	matrix_multiplication(sigma, transpose_act_layer, net.delta_nabla_w[net.num_of_layers-2], net.neurons_per_layer[net.num_of_layers-1], 1, 1, num_neurons);

	int max_neurons = net.neurons_per_layer[0];
	for(int i =0; i<net.num_of_layers; i++){
		if(max_neurons<net.neurons_per_layer[i]) max_neurons = net.neurons_per_layer[i];
	}

	double** z_sigmoid_derivative = create_matrix(max_neurons, 1);
	double** activations_transpose = create_matrix(1, max_neurons);
	double** weight_matrix_transpose = create_matrix(max_neurons, max_neurons);

	for(int k = net.num_of_layers-2; k>=1; k--){

		sigmoid_derivative_list(net.z_s[k-1], net.neurons_per_layer[k], z_sigmoid_derivative);
		
		int weight_rows = net.neurons_per_layer[k+1];
		int weight_cols = net.neurons_per_layer[k];

		transpose(net.weights[k], weight_rows, weight_cols, weight_matrix_transpose);

		matrix_multiplication(weight_matrix_transpose, sigma, sigma, weight_cols, weight_rows, net.neurons_per_layer[k+1], 1);
		hadamard_matrix_mult(sigma, z_sigmoid_derivative, weight_cols, 1, sigma);

		for(int i =0; i<net.neurons_per_layer[k]; i++){
			net.delta_nabla_b[k-1][i][0] = sigma[i][0];
		}

		transpose(net.activations[k-1], net.neurons_per_layer[k-1], 1, activations_transpose);

		 weight_rows = net.neurons_per_layer[k];
                 weight_cols = net.neurons_per_layer[k-1];

		matrix_multiplication(sigma, activations_transpose, net.delta_nabla_w[k-1], net.neurons_per_layer[k], 1, 1, net.neurons_per_layer[k-1]);

	}
}

void shuffle_data(double*** data, double*** labels, int num_samples){
	srand(time(NULL));
	for(int i = num_samples-1; i>0; i--){
		int j = rand() % (i+1);
		double** temp_data = data[i];
		data[i] = data[j];
		data[j] = temp_data;

		double** temp_label = labels[i];
		labels[i] = labels[j];
		labels[j] = temp_label;
	}
}

double*** initialize_nabla(int* size_array); // initialize this by 0

void nabla_addition(Network net, double*** nabla_w, double*** nabla_b){
	
	for(int k =0; k<net.num_of_layers-1; k++){
		matrix_addition(nabla_w[k], net.delta_nabla_w[k], net.neurons_per_layer[k+1], net.neurons_per_layer[k], nabla_w[k]);
		matrix_addition(nabla_b[k], net.delta_nabla_b[k], net.neurons_per_layer[k+1], 1, nabla_b[k]);
	}
}

void param_tuning(Network net, double*** nabla_w, double*** nabla_b){
	for(int k =0; k<net.num_of_layers-1; k++){
		matrix_addition(net.weights[k], nabla_w[k], net.neurons_per_layer[k+1], net.neurons_per_layer[k], net.weights[k]);
		matrix_addition(net.biases[k], nabla_b[k], net.neurons_per_layer[k+1], 1, net.biases[k]);
	}
}

void vector_multiplication_matrix(double multiplicand, double** matrix, int rows, int cols){
	for(int i =0; i<rows; i++){
		for(int j =0; j<cols; j++){
			matrix[i][j] = multiplicand*matrix[i][j];
		}
	}
}

void vector_list_multiplication(double multiplicand, int* dimensions, int num_of_dimensions, double*** nabla_w, double*** nabla_b){
	for(int k =0; k<num_of_dimensions-1; k++){
		vector_multiplication_matrix(multiplicand, nabla_w[k], dimensions[k+1], dimensions[k]);
		vector_multiplication_matrix(multiplicand, nabla_b[k], dimensions[k+1], 1);
	}
}

void sgd(Network net,int epochs, double*** training_data, double*** labels, int num_samples, int batch_size, double learning_rate){
	for(int i =0; i<epochs; i++){
		shuffle_data(training_data, labels, num_samples);

		double*** nabla_w = initialize_nabla();
		double*** nabla_b = initialize_nabla();

		for(int j =0; j<num_samples; j++){
			backprop(net, training_data[j], labels[j]);
			nabla_addition(net, nabla_w, nabla_b);

			if((j+1)%batch_size == 0 || j==num_samples-1){
				double mult = -1*learning_rate/batch_size;
				vector_list_multiplication(mult, net.neurons_per_layer, net.num_of_layers, nabla_w, nabla_b);
				param_tuning(net, nabla_w, nabla_b);
			}
		}
	}
}
				


