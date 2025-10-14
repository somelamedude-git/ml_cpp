#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <time.h>

typedef struct Network{
	int* neurons_per_layer;
	int num_of_layers;
	double*** weights;
	double*** biases;
	double*** activations;
	double*** z_s;
	double*** delta_nabla_b;
	double*** delta_nabla_w;
} Network;

void free_memory_twoD(double** matrix, int rows, int cols){
	if(matrix == NULL){
		return;
	}
	for(int i = 0; i < rows; i++){
		if(matrix[i] == NULL){
			continue;
		}
		free(matrix[i]);
	}
	free(matrix);
}

void print_weights(Network net, int epoch){
	printf("\n========== Weights at Epoch %d ==========\n", epoch);
	for(int layer = 0; layer < net.num_of_layers - 1; layer++){
		printf("\nLayer %d -> %d:\n", layer, layer + 1);
		for(int i = 0; i < net.neurons_per_layer[layer + 1]; i++){
			printf("  Neuron %d: [ ", i);
			for(int j = 0; j < net.neurons_per_layer[layer]; j++){
				printf("%.6f ", net.weights[layer][i][j]);
			}
			printf("]\n");
		}
	}
	printf("==========================================\n\n");
}

Network initialize_network(int num_layers, int neuron_per_layer[]){
	if(num_layers < 2){
		exit(1);
	}
	if(neuron_per_layer == NULL){
		exit(1);
	}
	
	Network net = {0};
	net.num_of_layers = num_layers;
	net.neurons_per_layer = (int*)malloc(sizeof(int) * num_layers);
	if(net.neurons_per_layer == NULL){
		exit(1);
	}
	
	for(int i = 0; i < num_layers; i++){
		net.neurons_per_layer[i] = neuron_per_layer[i];
	}
	
	srand(time(NULL));
	
	net.weights = (double***)malloc(sizeof(double**) * (num_layers - 1));
	if(net.weights == NULL){
		exit(1);
	}
	
	for(int i = 0; i < num_layers - 1; i++){
		int rows = neuron_per_layer[i+1];
		int cols = neuron_per_layer[i];
		
		net.weights[i] = (double**)malloc(sizeof(double*) * rows);
		if(net.weights[i] == NULL){
			exit(1);
		}
		
		for(int j = 0; j < rows; j++){
			net.weights[i][j] = (double*)malloc(sizeof(double) * cols);
			if(net.weights[i][j] == NULL){
				exit(1);
			}
			for(int k = 0; k < cols; k++){
				net.weights[i][j][k] = ((double)rand() / RAND_MAX - 0.5) * 2;
			}
		}
	}
	
	net.biases = (double***)malloc(sizeof(double**) * (num_layers - 1));
	if(net.biases == NULL){
		exit(1);
	}
	
	for(int i = 0; i < num_layers - 1; i++){
		int rows = neuron_per_layer[i+1];
		net.biases[i] = (double**)malloc(sizeof(double*) * rows);
		if(net.biases[i] == NULL){
			exit(1);
		}
		for(int j = 0; j < rows; j++){
			net.biases[i][j] = (double*)calloc(1, sizeof(double));
			if(net.biases[i][j] == NULL){
				exit(1);
			}
		}
	}
	
	net.activations = (double***)malloc(sizeof(double**) * num_layers);
	if(net.activations == NULL){
		exit(1);
	}
	
	for(int i = 0; i < num_layers; i++){
		int rows = neuron_per_layer[i];
		net.activations[i] = (double**)malloc(sizeof(double*) * rows);
		if(net.activations[i] == NULL){
			exit(1);
		}
		for(int j = 0; j < rows; j++){
			net.activations[i][j] = (double*)calloc(1, sizeof(double));
			if(net.activations[i][j] == NULL){
				exit(1);
			}
		}
	}
	
	net.z_s = (double***)malloc(sizeof(double**) * (num_layers - 1));
	if(net.z_s == NULL){
		exit(1);
	}
	
	for(int i = 0; i < num_layers - 1; i++){
		int rows = neuron_per_layer[i+1];
		net.z_s[i] = (double**)malloc(sizeof(double*) * rows);
		if(net.z_s[i] == NULL){
			exit(1);
		}
		for(int j = 0; j < rows; j++){
			net.z_s[i][j] = (double*)calloc(1, sizeof(double));
			if(net.z_s[i][j] == NULL){
				exit(1);
			}
		}
	}
	
	net.delta_nabla_w = (double***)malloc(sizeof(double**) * (num_layers - 1));
	if(net.delta_nabla_w == NULL){
		exit(1);
	}
	
	for(int i = 0; i < num_layers - 1; i++){
		int rows = neuron_per_layer[i+1];
		int cols = neuron_per_layer[i];
		net.delta_nabla_w[i] = (double**)malloc(sizeof(double*) * rows);
		if(net.delta_nabla_w[i] == NULL){
			exit(1);
		}
		for(int j = 0; j < rows; j++){
			net.delta_nabla_w[i][j] = (double*)calloc(cols, sizeof(double));
			if(net.delta_nabla_w[i][j] == NULL){
				exit(1);
			}
		}
	}
	
	net.delta_nabla_b = (double***)malloc(sizeof(double**) * (num_layers - 1));
	if(net.delta_nabla_b == NULL){
		exit(1);
	}
	
	for(int i = 0; i < num_layers - 1; i++){
		int rows = neuron_per_layer[i+1];
		net.delta_nabla_b[i] = (double**)malloc(sizeof(double*) * rows);
		if(net.delta_nabla_b[i] == NULL){
			exit(1);
		}
		for(int j = 0; j < rows; j++){
			net.delta_nabla_b[i][j] = (double*)calloc(1, sizeof(double));
			if(net.delta_nabla_b[i][j] == NULL){
				exit(1);
			}
		}
	}
	
	return net;
}

double sigmoid(double x){
	if(isnan(x) || isinf(x)){
		return 0.0;
	}
	double sig = 1.0/(1.0 + exp(-x));
	return sig;
}

double sigmoid_derivative(double x){
	if(isnan(x) || isinf(x)){
		return 0.0;
	}
	double derivative = exp(-x)/((1+exp(-x))*(1+exp(-x)));
	return derivative;
}

void sigmoid_list(double** x_list, int size_of_list, double** activated_array){
	if(x_list == NULL || activated_array == NULL){
		return;
	}
	if(size_of_list <= 0){
		return;
	}
	for(int i = 0; i < size_of_list; i++){
		if(x_list[i] == NULL || activated_array[i] == NULL){
			continue;
		}
		activated_array[i][0] = sigmoid(x_list[i][0]);
	}
}

void sigmoid_derivative_list(double** x_list, int size_of_list, double** derivative_array){
	if(x_list == NULL || derivative_array == NULL){
		return;
	}
	if(size_of_list <= 0){
		return;
	}
	for(int i = 0; i < size_of_list; i++){
		if(x_list[i] == NULL || derivative_array[i] == NULL){
			continue;
		}
		derivative_array[i][0] = sigmoid_derivative(x_list[i][0]);
	}
}

void transpose(double** arr, int rows, int cols, double** transpose_arr){
	if(arr == NULL || transpose_arr == NULL){
		return;
	}
	if(rows <= 0 || cols <= 0){
		return;
	}
	for(int i = 0; i < rows; i++){
		if(arr[i] == NULL){
			continue;
		}
		for(int j = 0; j < cols; j++){
			if(transpose_arr[j] == NULL){
				break;
			}
			transpose_arr[j][i] = arr[i][j];
		}
	}
}

double** create_matrix(int rows, int cols){
	if(rows <= 0 || cols <= 0){
		return NULL;
	}
	double** matrix = (double**)malloc(sizeof(double*) * rows);
	if(matrix == NULL){
		return NULL;
	}
	for(int i = 0; i < rows; i++){
		matrix[i] = calloc(cols, sizeof(double));
		if(matrix[i] == NULL){
			for(int j = 0; j < i; j++){
				free(matrix[j]);
			}
			free(matrix);
			return NULL;
		}
	}
	return matrix;
}

void matrix_multiplication(
	double** matrix_one,
	double** matrix_two,
	double** product_matrix,
	int row_one, int col_one, int row_two, int col_two
) {
	if(matrix_one == NULL || matrix_two == NULL || product_matrix == NULL){
		return;
	}
	
	if(col_one != row_two){
		return;
	}
	
	if(row_one <= 0 || col_one <= 0 || row_two <= 0 || col_two <= 0){
		return;
	}
	
	for(int i = 0; i < row_one; i++){
		if(matrix_one[i] == NULL || product_matrix[i] == NULL){
			continue;
		}
		for(int k = 0; k < col_two; k++){
			product_matrix[i][k] = 0;
		}
	}
	
	for(int i = 0; i < row_one; i++){
		for(int k = 0; k < col_two; k++){
			for(int j = 0; j < col_one; j++){
				if(matrix_two[j] == NULL){
					break;
				}
				product_matrix[i][k] += matrix_one[i][j] * matrix_two[j][k];
			}
		}
	}
}

void matrix_addition(double** matrix_one, double** matrix_two, int rows, int cols, double** addition_matrix){
	if(matrix_one == NULL || matrix_two == NULL || addition_matrix == NULL){
		return;
	}
	if(rows <= 0 || cols <= 0){
		return;
	}
	for(int i = 0; i < rows; i++){
		if(matrix_one[i] == NULL || matrix_two[i] == NULL || addition_matrix[i] == NULL){
			continue;
		}
		for(int j = 0; j < cols; j++){
			addition_matrix[i][j] = matrix_one[i][j] + matrix_two[i][j];
		}
	}
}

void feed_forward(Network net, double** activation_input){
	if(activation_input == NULL){
		return;
	}
	
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
	
	double** product_matrix = create_matrix(max_neurons, 1);
	double** addition_matrix = create_matrix(max_neurons, 1);
	double** activated_output = create_matrix(max_neurons, 1);
	
	if(product_matrix == NULL || addition_matrix == NULL || activated_output == NULL){
		return;
	}
	
	for(int i = 0; i < net.num_of_layers-1; i++){
		int weight_rows = net.neurons_per_layer[i+1];
		int weight_cols = net.neurons_per_layer[i];
		
		if(net.weights[i] == NULL){
			break;
		}
		
		matrix_multiplication(net.weights[i], activation_input, product_matrix, weight_rows, weight_cols, size_of_input, 1);
		
		if(net.biases[i] == NULL){
			break;
		}
		
		matrix_addition(net.biases[i], product_matrix, weight_rows, 1, addition_matrix);
		
		for(int j = 0; j < weight_rows; j++){
			if(net.z_s[i][j] == NULL || addition_matrix[j] == NULL){
				continue;
			}
			net.z_s[i][j][0] = addition_matrix[j][0];
		}
		
		sigmoid_list(net.z_s[i], weight_rows, activated_output);
		
		for(int j = 0; j < weight_rows; j++){
			if(activation_input[j] == NULL || activated_output[j] == NULL || net.activations[i+1][j] == NULL){
				continue;
			}
			activation_input[j][0] = activated_output[j][0];
			net.activations[i+1][j][0] = activated_output[j][0];
		}
		
		size_of_input = weight_rows;
	}
	
	free_memory_twoD(addition_matrix, max_neurons, 1);
	free_memory_twoD(activated_output, max_neurons, 1);
	free_memory_twoD(product_matrix, max_neurons, 1);
}

void matrix_subt(double** matrix_one, double** matrix_two, int rows, int cols, double** difference_matrix){
	if(matrix_one == NULL || matrix_two == NULL || difference_matrix == NULL){
		return;
	}
	if(rows <= 0 || cols <= 0){
		return;
	}
	for(int i = 0; i < rows; i++){
		if(matrix_one[i] == NULL || matrix_two[i] == NULL || difference_matrix[i] == NULL){
			continue;
		}
		for(int j = 0; j < cols; j++){
			difference_matrix[i][j] = matrix_one[i][j] - matrix_two[i][j];
		}
	}
}

void compute_cost_derivative(Network net, double** expected_output, double** output, double** derivative_matrix){
	if(expected_output == NULL || output == NULL || derivative_matrix == NULL){
		return;
	}
	int rows = net.neurons_per_layer[net.num_of_layers-1];
	matrix_subt(output, expected_output, rows, 1, derivative_matrix);
}

void hadamard_matrix_mult(double** matrix_one, double** matrix_two, int rows, int cols, double** product_matrix){
	if(matrix_one == NULL || matrix_two == NULL || product_matrix == NULL){
		return;
	}
	if(rows <= 0 || cols <= 0){
		return;
	}
	for(int i = 0; i < rows; i++){
		if(matrix_one[i] == NULL || matrix_two[i] == NULL || product_matrix[i] == NULL){
			continue;
		}
		for(int j = 0; j < cols; j++){
			product_matrix[i][j] = matrix_one[i][j] * matrix_two[i][j];
		}
	}
}

void backprop(Network net, double** input, double** output){
	if(input == NULL || output == NULL){
		return;
	}
	
	int max_neurons = net.neurons_per_layer[0];
	for(int i = 0; i < net.num_of_layers; i++){
		if(max_neurons < net.neurons_per_layer[i]) max_neurons = net.neurons_per_layer[i];
	}
	
	double** input_copy = create_matrix(max_neurons, 1);
	if(input_copy == NULL){
		return;
	}
	
	for(int i = 0; i < net.neurons_per_layer[0]; i++){
		if(input[i] == NULL || input_copy[i] == NULL){
			continue;
		}
		input_copy[i][0] = input[i][0];
	}
	
	feed_forward(net, input_copy);
	
	int num_neurons = net.neurons_per_layer[net.num_of_layers-1];
	
	double** derivative_array = create_matrix(num_neurons, 1);
	if(derivative_array == NULL){
		free_memory_twoD(input_copy, max_neurons, 1);
		return;
	}
	
	sigmoid_derivative_list(net.z_s[net.num_of_layers-2], num_neurons, derivative_array);
	
	double** derivative_matrix = create_matrix(num_neurons, 1);
	if(derivative_matrix == NULL){
		free_memory_twoD(input_copy, max_neurons, 1);
		free_memory_twoD(derivative_array, num_neurons, 1);
		return;
	}
	
	compute_cost_derivative(net, output, net.activations[net.num_of_layers-1], derivative_matrix);
	
	double** sigma = create_matrix(max_neurons, 1);
	if(sigma == NULL){
		free_memory_twoD(input_copy, max_neurons, 1);
		free_memory_twoD(derivative_array, num_neurons, 1);
		free_memory_twoD(derivative_matrix, num_neurons, 1);
		return;
	}
	
	hadamard_matrix_mult(derivative_matrix, derivative_array, num_neurons, 1, sigma);
	
	for(int j = 0; j < num_neurons; j++){
		if(net.delta_nabla_b[net.num_of_layers-2][j] == NULL || sigma[j] == NULL){
			continue;
		}
		net.delta_nabla_b[net.num_of_layers-2][j][0] = sigma[j][0];
	}
	
	num_neurons = net.neurons_per_layer[net.num_of_layers-2];
	
	double** transpose_act_layer = create_matrix(1, num_neurons);
	if(transpose_act_layer == NULL){
		free_memory_twoD(input_copy, max_neurons, 1);
		free_memory_twoD(derivative_array, net.neurons_per_layer[net.num_of_layers-1], 1);
		free_memory_twoD(derivative_matrix, net.neurons_per_layer[net.num_of_layers-1], 1);
		free_memory_twoD(sigma, net.neurons_per_layer[net.num_of_layers-1], 1);
		return;
	}
	
	transpose(net.activations[net.num_of_layers-2], num_neurons, 1, transpose_act_layer);
	
	matrix_multiplication(sigma, transpose_act_layer, net.delta_nabla_w[net.num_of_layers-2], 
	                     net.neurons_per_layer[net.num_of_layers-1], 1, 1, num_neurons);
	
	double** z_sigmoid_derivative = create_matrix(max_neurons, 1);
	double** activations_transpose = create_matrix(1, max_neurons);
	double** weight_matrix_transpose = create_matrix(max_neurons, max_neurons);
	
	if(z_sigmoid_derivative == NULL || activations_transpose == NULL || weight_matrix_transpose == NULL){
		free_memory_twoD(input_copy, max_neurons, 1);
		free_memory_twoD(derivative_array, net.neurons_per_layer[net.num_of_layers-1], 1);
		free_memory_twoD(derivative_matrix, net.neurons_per_layer[net.num_of_layers-1], 1);
		free_memory_twoD(sigma, net.neurons_per_layer[net.num_of_layers-1], 1);
		free_memory_twoD(transpose_act_layer, 1, net.neurons_per_layer[net.num_of_layers-2]);
		return;
	}
	
	for(int k = net.num_of_layers-2; k >= 1; k--){
		sigmoid_derivative_list(net.z_s[k-1], net.neurons_per_layer[k], z_sigmoid_derivative);
		
		int weight_rows = net.neurons_per_layer[k+1];
		int weight_cols = net.neurons_per_layer[k];
		
		if(net.weights[k] == NULL){
			break;
		}
		
		transpose(net.weights[k], weight_rows, weight_cols, weight_matrix_transpose);
		double** temp_sigma = create_matrix(weight_cols, 1);		
		matrix_multiplication(weight_matrix_transpose, sigma, temp_sigma, weight_cols, weight_rows, net.neurons_per_layer[k+1], 1);
		 for(int m = 0; m < weight_cols; m++){
        sigma[m][0] = temp_sigma[m][0];
    }
		  free_memory_twoD(temp_sigma, weight_cols, 1);
		hadamard_matrix_mult(sigma, z_sigmoid_derivative, weight_cols, 1, sigma);
		
		for(int i = 0; i < net.neurons_per_layer[k]; i++){
			if(net.delta_nabla_b[k-1][i] == NULL || sigma[i] == NULL){
				continue;
			}
			net.delta_nabla_b[k-1][i][0] = sigma[i][0];
		}
		
		if(net.activations[k-1] == NULL){
			break;
		}
		
		transpose(net.activations[k-1], net.neurons_per_layer[k-1], 1, activations_transpose);
		
		weight_rows = net.neurons_per_layer[k];
		weight_cols = net.neurons_per_layer[k-1];
		
		if(net.delta_nabla_w[k-1] == NULL){
			break;
		}
		
		matrix_multiplication(sigma, activations_transpose, net.delta_nabla_w[k-1], net.neurons_per_layer[k], 1, 1, net.neurons_per_layer[k-1]);
	}
	
	free_memory_twoD(input_copy, max_neurons, 1);
	free_memory_twoD(z_sigmoid_derivative, max_neurons, 1);
	free_memory_twoD(activations_transpose, 1, max_neurons);
	free_memory_twoD(weight_matrix_transpose, max_neurons, max_neurons);
	free_memory_twoD(derivative_array, net.neurons_per_layer[net.num_of_layers-1], 1);
	free_memory_twoD(derivative_matrix, net.neurons_per_layer[net.num_of_layers-1], 1);
	free_memory_twoD(sigma, net.neurons_per_layer[net.num_of_layers-1], 1);
	free_memory_twoD(transpose_act_layer, 1, net.neurons_per_layer[net.num_of_layers-2]);
}

void shuffle_data(double*** data, double*** labels, int num_samples){
	if(data == NULL || labels == NULL){
		return;
	}
	if(num_samples <= 0){
		return;
	}
	srand(time(NULL));
	for(int i = num_samples-1; i > 0; i--){
		int j = rand() % (i+1);
		double** temp_data = data[i];
		data[i] = data[j];
		data[j] = temp_data;
		
		double** temp_label = labels[i];
		labels[i] = labels[j];
		labels[j] = temp_label;
	}
}

double*** initialize_nabla(int* size_array, int num_of_layers, bool is_bias){
	if(size_array == NULL){
		return NULL;
	}
	if(num_of_layers < 2){
		return NULL;
	}
	
	if(is_bias){
		double*** bias = (double***)malloc(sizeof(double**) * (num_of_layers-1));
		if(bias == NULL){
			return NULL;
		}
		for(int k = 0; k < num_of_layers-1; k++){
			bias[k] = (double**)malloc(sizeof(double*) * size_array[k+1]);
			if(bias[k] == NULL){
				for(int m = 0; m < k; m++){
					for(int n = 0; n < size_array[m+1]; n++){
						free(bias[m][n]);
					}
					free(bias[m]);
				}
				free(bias);
				return NULL;
			}
			for(int j = 0; j < size_array[k+1]; j++){
				bias[k][j] = (double*)calloc(1, sizeof(double));
			}
		}
		return bias;
	}
	else{
		double*** weights = (double***)malloc(sizeof(double**) * (num_of_layers-1));
		if(weights == NULL){
			return NULL;
		}
		for(int k = 0; k < num_of_layers-1; k++){
			weights[k] = (double**)malloc(sizeof(double*) * size_array[k+1]);
			if(weights[k] == NULL){
				for(int m = 0; m < k; m++){
					for(int n = 0; n < size_array[m+1]; n++){
						free(weights[m][n]);
					}
					free(weights[m]);
				}
				free(weights);
				return NULL;
			}
			for(int j = 0; j < size_array[k+1]; j++){
				weights[k][j] = (double*)calloc(size_array[k], sizeof(double));
			}
		}
		return weights;
	}
}

void nabla_addition(Network net, double*** nabla_w, double*** nabla_b){
	if(nabla_w == NULL || nabla_b == NULL){
		return;
	}
	for(int k = 0; k < net.num_of_layers-1; k++){
		if(nabla_w[k] == NULL || nabla_b[k] == NULL || net.delta_nabla_w[k] == NULL || net.delta_nabla_b[k] == NULL){
			continue;
		}
		matrix_addition(nabla_w[k], net.delta_nabla_w[k], net.neurons_per_layer[k+1], net.neurons_per_layer[k], nabla_w[k]);
		matrix_addition(nabla_b[k], net.delta_nabla_b[k], net.neurons_per_layer[k+1], 1, nabla_b[k]);
	}
}

void param_tuning(Network net, double*** nabla_w, double*** nabla_b){
	if(nabla_w == NULL || nabla_b == NULL){
		return;
	}
	for(int k = 0; k < net.num_of_layers-1; k++){
		if(net.weights[k] == NULL || net.biases[k] == NULL || nabla_w[k] == NULL || nabla_b[k] == NULL){
			continue;
		}
		matrix_addition(net.weights[k], nabla_w[k], net.neurons_per_layer[k+1], net.neurons_per_layer[k], net.weights[k]);
		matrix_addition(net.biases[k], nabla_b[k], net.neurons_per_layer[k+1], 1, net.biases[k]);
	}
}

void vector_multiplication_matrix(double multiplicand, double** matrix, int rows, int cols){
	if(matrix == NULL){
		return;
	}
	if(rows <= 0 || cols <= 0){
		return;
	}
	for(int i = 0; i < rows; i++){
		if(matrix[i] == NULL){
			continue;
		}
		for(int j = 0; j < cols; j++){
			matrix[i][j] = multiplicand * matrix[i][j];
		}
	}
}

void vector_list_multiplication(double multiplicand, int* dimensions, int num_of_dimensions, double*** nabla_w, double*** nabla_b){
	if(dimensions == NULL || nabla_w == NULL || nabla_b == NULL){
		return;
	}
	if(num_of_dimensions < 2){
		return;
	}
	for(int k = 0; k < num_of_dimensions-1; k++){
		if(nabla_w[k] == NULL || nabla_b[k] == NULL){
			continue;
		}
		vector_multiplication_matrix(multiplicand, nabla_w[k], dimensions[k+1], dimensions[k]);
		vector_multiplication_matrix(multiplicand, nabla_b[k], dimensions[k+1], 1);
	}
}

void reset_nabla(double*** nabla_w, double*** nabla_b, int* size_array, int num_layers){
	if(nabla_w == NULL || nabla_b == NULL || size_array == NULL){
		return;
	}
	if(num_layers < 2){
		return;
	}
	for(int i = 0; i < num_layers-1; i++){
		if(nabla_w[i] == NULL || nabla_b[i] == NULL){
			continue;
		}
		for(int j = 0; j < size_array[i+1]; j++){
			if(nabla_b[i][j] == NULL){
				continue;
			}
			nabla_b[i][j][0] = 0;
			if(nabla_w[i][j] == NULL){
				continue;
			}
			for(int k = 0; k < size_array[i]; k++){
				nabla_w[i][j][k] = 0;
			}
		}
	}
}

void sgd(Network net, int epochs, double*** training_data, double*** labels, int num_samples, int batch_size, double learning_rate){
	if(training_data == NULL || labels == NULL){
		return;
	}
	if(epochs <= 0 || num_samples <= 0 || batch_size <= 0){
		return;
	}
	if(learning_rate <= 0){
		return;
	}
	
	print_weights(net, 0);
	
	for(int i = 0; i < epochs; i++){
		shuffle_data(training_data, labels, num_samples);
		
		double*** nabla_w = initialize_nabla(net.neurons_per_layer, net.num_of_layers, false);
		double*** nabla_b = initialize_nabla(net.neurons_per_layer, net.num_of_layers, true);
		
		if(nabla_w == NULL || nabla_b == NULL){
			break;
		}
		
		for(int j = 0; j < num_samples; j++){
			if(training_data[j] == NULL || labels[j] == NULL){
				continue;
			}
			
			backprop(net, training_data[j], labels[j]);
			nabla_addition(net, nabla_w, nabla_b);
			
			if((j+1) % batch_size == 0 || j == num_samples-1){
				double mult = -1 * learning_rate / batch_size;
				vector_list_multiplication(mult, net.neurons_per_layer, net.num_of_layers, nabla_w, nabla_b);
				param_tuning(net, nabla_w, nabla_b);
				reset_nabla(nabla_w, nabla_b, net.neurons_per_layer, net.num_of_layers);
			}
		}
		
		if((i+1) % 1000 == 0 || i == epochs - 1){
			print_weights(net, i+1);
		}
		
		for(int k = 0; k < net.num_of_layers-1; k++){
			if(nabla_w[k] != NULL){
				free_memory_twoD(nabla_w[k], net.neurons_per_layer[k+1], net.neurons_per_layer[k]);
			}
			if(nabla_b[k] != NULL){
				free_memory_twoD(nabla_b[k], net.neurons_per_layer[k+1], 1);
			}
		}
		
		free(nabla_w);
		free(nabla_b);
	}
}


// Append this to your existing code, replacing the main() function

// Helper function to create a simplified digit pattern
void create_digit_pattern(double** image, int digit, int variation) {
    // Initialize all pixels to 0 (black)
    for(int i = 0; i < 784; i++) {
        image[i][0] = 0.0;
    }
    
    // Base patterns with variations
    int offset_x = (variation % 3) - 1;  // -1, 0, 1
    int offset_y = ((variation / 3) % 3) - 1;
    
    switch(digit) {
        case 0: // Draw a circle-ish pattern
            for(int row = 8; row < 20; row++) {
                for(int col = 8; col < 20; col++) {
                    int dist_sq = (row-14+offset_y)*(row-14+offset_y) + (col-14+offset_x)*(col-14+offset_x);
                    if(dist_sq > 20 && dist_sq < 50) {
                        image[row * 28 + col][0] = 0.8;
                    }
                }
            }
            break;
        case 1: // Vertical line
            for(int row = 5; row < 23; row++) {
                image[row * 28 + (14+offset_x)][0] = 0.8;
                image[row * 28 + (13+offset_x)][0] = 0.6;
            }
            break;
        case 2: // Z-shaped pattern
            for(int col = 8; col < 20; col++) {
                image[(8+offset_y) * 28 + col][0] = 0.8;
                image[(15+offset_y) * 28 + col][0] = 0.8;
                image[(22+offset_y) * 28 + col][0] = 0.8;
            }
            for(int row = 8; row < 22; row++) {
                int col = 8 + (20 - row);
                if(col >= 8 && col < 20) {
                    image[row * 28 + col][0] = 0.8;
                }
            }
            break;
        case 3: // Two horizontal bars on right
            for(int col = 12; col < 20; col++) {
                image[(8+offset_y) * 28 + col][0] = 0.8;
                image[(15+offset_y) * 28 + col][0] = 0.8;
                image[(22+offset_y) * 28 + col][0] = 0.8;
            }
            for(int row = 8; row < 22; row++) {
                image[row * 28 + (19+offset_x)][0] = 0.8;
            }
            break;
        case 4: // Like a chair
            for(int row = 5; row < 15; row++) {
                image[row * 28 + (10+offset_x)][0] = 0.8;
            }
            for(int col = 10; col < 18; col++) {
                image[(14+offset_y) * 28 + col][0] = 0.8;
            }
            for(int row = 5; row < 23; row++) {
                image[row * 28 + (17+offset_x)][0] = 0.8;
            }
            break;
        case 5: // S-shaped
            for(int col = 8; col < 20; col++) {
                image[(7+offset_y) * 28 + col][0] = 0.8;
                image[(14+offset_y) * 28 + col][0] = 0.8;
                image[(21+offset_y) * 28 + col][0] = 0.8;
            }
            for(int row = 7; row < 14; row++) {
                image[row * 28 + (8+offset_x)][0] = 0.8;
            }
            for(int row = 14; row < 21; row++) {
                image[row * 28 + (19+offset_x)][0] = 0.8;
            }
            break;
        case 6: // Circle with line on left
            for(int row = 10; row < 20; row++) {
                for(int col = 10; col < 18; col++) {
                    int dist_sq = (row-15+offset_y)*(row-15+offset_y) + (col-14+offset_x)*(col-14+offset_x);
                    if(dist_sq > 10 && dist_sq < 30) {
                        image[row * 28 + col][0] = 0.8;
                    }
                }
            }
            for(int row = 5; row < 15; row++) {
                image[row * 28 + (10+offset_x)][0] = 0.8;
            }
            break;
        case 7: // Diagonal from top
            for(int col = 8; col < 20; col++) {
                image[(7+offset_y) * 28 + col][0] = 0.8;
            }
            for(int row = 7; row < 23; row++) {
                int col = 19 - (row - 7) / 2 + offset_x;
                if(col >= 8 && col < 20) {
                    image[row * 28 + col][0] = 0.8;
                }
            }
            break;
        case 8: // Two circles stacked
            for(int row = 6; row < 14; row++) {
                for(int col = 10; col < 18; col++) {
                    int dist_sq = (row-10+offset_y)*(row-10+offset_y) + (col-14+offset_x)*(col-14+offset_x);
                    if(dist_sq > 5 && dist_sq < 20) {
                        image[row * 28 + col][0] = 0.8;
                    }
                }
            }
            for(int row = 14; row < 22; row++) {
                for(int col = 10; col < 18; col++) {
                    int dist_sq = (row-18+offset_y)*(row-18+offset_y) + (col-14+offset_x)*(col-14+offset_x);
                    if(dist_sq > 5 && dist_sq < 20) {
                        image[row * 28 + col][0] = 0.8;
                    }
                }
            }
            break;
        case 9: // Circle with line on right
            for(int row = 8; row < 18; row++) {
                for(int col = 10; col < 18; col++) {
                    int dist_sq = (row-13+offset_y)*(row-13+offset_y) + (col-14+offset_x)*(col-14+offset_x);
                    if(dist_sq > 10 && dist_sq < 30) {
                        image[row * 28 + col][0] = 0.8;
                    }
                }
            }
            for(int row = 13; row < 23; row++) {
                image[row * 28 + (17+offset_x)][0] = 0.8;
            }
            break;
    }
}

// Helper to create one-hot encoded label
void create_one_hot_label(double** label, int digit) {
    for(int i = 0; i < 10; i++) {
        label[i][0] = 0.0;
    }
    label[digit][0] = 1.0;
}

void network_file(const char* filename_weights, const char* filename_bias, Network net){ // The filename has to be in the binary format, name of the file can be test.bin
	int num_of_layers = net.num_of_layers-1;
	FILE* file_weights = fopen(filename_weights, "wb");
	FILE* file_bias = fopen(filename_bias, "wb");

	for(int k =0; k<num_of_layers; k++){
		for(int i =0; i<net.neurons_per_layer[k+1]; i++){
			fwrite(&net.biases[k][i][0], sizeof(double), 1, file_bias);
			for(int j =0; j<net.neurons_per_layer[k]; j++){
				fwrite(&net.weights[k][i][j], sizeof(double), 1, file_weights);
			}
		}
	}
}

				
int main(){
    printf("========================================================\n");
    printf("Neural Network Training on MNIST-like Handwritten Digits\n");
    printf("========================================================\n\n");

    // Create dataset with UNEVEN distribution per digit (more realistic)
    // samples_per_digit: how many samples of each digit to generate
    int samples_per_digit[] = {7, 5, 8, 6, 5, 9, 4, 7, 6, 8}; // Uneven!
    int train_per_digit[] = {5, 3, 6, 4, 3, 6, 2, 5, 4, 5};    // Training split
    
    int total_samples = 0;
    int train_samples = 0;
    for(int i = 0; i < 10; i++) {
        total_samples += samples_per_digit[i];
        train_samples += train_per_digit[i];
    }
    int test_samples = total_samples - train_samples;
    
    int input_size = 784;  // 28x28 pixels
    int output_size = 10;  // 10 digits (0-9)

    double*** all_data = (double***)malloc(sizeof(double**) * total_samples);
    double*** all_labels = (double***)malloc(sizeof(double**) * total_samples);

    printf("Generating %d total samples (UNEVEN distribution)...\n", total_samples);
    printf("Distribution per digit: [");
    for(int i = 0; i < 10; i++) {
        printf("%d", samples_per_digit[i]);
        if(i < 9) printf(", ");
    }
    printf("]\n\n");
    
    // Create samples with uneven distribution
    int idx = 0;
    for(int digit = 0; digit < 10; digit++) {
        for(int sample = 0; sample < samples_per_digit[digit]; sample++) {
            
            all_data[idx] = create_matrix(input_size, 1);
            all_labels[idx] = create_matrix(output_size, 1);
            
            create_digit_pattern(all_data[idx], digit, sample);
            create_one_hot_label(all_labels[idx], digit);
            
            // Add some noise to make samples different
            srand(time(NULL) + idx * 13);
            for(int i = 0; i < 784; i++) {
                if(rand() % 100 < 5) { // 5% noise
                    all_data[idx][i][0] += ((double)rand() / RAND_MAX) * 0.3 - 0.15;
                    if(all_data[idx][i][0] < 0.0) all_data[idx][i][0] = 0.0;
                    if(all_data[idx][i][0] > 1.0) all_data[idx][i][0] = 1.0;
                }
            }
            idx++;
        }
    }

    // Split into train and test sets (uneven split per digit)
    double*** training_data = (double***)malloc(sizeof(double**) * train_samples);
    double*** train_labels = (double***)malloc(sizeof(double**) * train_samples);
    double*** test_data = (double***)malloc(sizeof(double**) * test_samples);
    double*** test_labels = (double***)malloc(sizeof(double**) * test_samples);

    int train_idx = 0;
    int test_idx = 0;
    int data_idx = 0;
    
    for(int digit = 0; digit < 10; digit++) {
        // First N samples go to training (N = train_per_digit[digit])
        for(int i = 0; i < train_per_digit[digit]; i++) {
            training_data[train_idx] = all_data[data_idx];
            train_labels[train_idx] = all_labels[data_idx];
            train_idx++;
            data_idx++;
        }
        // Remaining samples go to testing
        for(int i = train_per_digit[digit]; i < samples_per_digit[digit]; i++) {
            test_data[test_idx] = all_data[data_idx];
            test_labels[test_idx] = all_labels[data_idx];
            test_idx++;
            data_idx++;
        }
    }

    printf("Dataset Split (UNEVEN per digit):\n");
    printf("  Digit | Total | Train | Test\n");
    printf("  ------|-------|-------|-----\n");
    for(int i = 0; i < 10; i++) {
        printf("    %d   |   %d   |   %d   |  %d\n", 
               i, samples_per_digit[i], train_per_digit[i], 
               samples_per_digit[i] - train_per_digit[i]);
    }
    printf("  ------|-------|-------|-----\n");
    printf("  Total |  %2d   |  %2d   | %2d\n\n", total_samples, train_samples, test_samples);

    // Network architecture: 784 inputs -> 128 hidden -> 64 hidden -> 10 output
    int num_layers = 4;
    int neurons_per_layer[] = {784, 128, 64, 10};

    printf("Network Architecture: ");
    for(int i = 0; i < num_layers; i++) {
        printf("%d", neurons_per_layer[i]);
        if(i < num_layers - 1) printf(" -> ");
    }
    printf("\n");
    printf("Total parameters: %d weights + biases\n", 
           784*128 + 128*64 + 64*10 + 128 + 64 + 10);
    printf("\n");

    Network net = initialize_network(num_layers, neurons_per_layer);

    // Training parameters
    int epochs = 5000;
    int batch_size = 10;
    double learning_rate = 0.1;

    printf("Training Parameters:\n");
    printf("  Epochs: %d\n", epochs);
    printf("  Batch size: %d\n", batch_size);
    printf("  Learning rate: %.3f\n\n", learning_rate);

    printf("Training started...\n");
    printf("(Progress will be shown every 1000 epochs)\n\n");
    
    sgd(net, epochs, training_data, train_labels, train_samples, batch_size, learning_rate);
    
    printf("\nTraining completed!\n\n");

    // Test on TRAINING set
    printf("========================================================\n");
    printf("Testing on TRAINING SET (30 samples)\n");
    printf("========================================================\n\n");

    double** test_input = create_matrix(784, 1);
    int train_correct = 0;
    
    int train_digit_correct[10] = {0};
    int train_digit_total[10] = {0};

    for(int i = 0; i < train_samples; i++) {
        for(int j = 0; j < 784; j++) {
            test_input[j][0] = training_data[i][j][0];
        }

        feed_forward(net, test_input);

        int predicted_class = 0;
        double max_activation = net.activations[net.num_of_layers-1][0][0];
        for(int j = 1; j < 10; j++) {
            if(net.activations[net.num_of_layers-1][j][0] > max_activation) {
                max_activation = net.activations[net.num_of_layers-1][j][0];
                predicted_class = j;
            }
        }

        int expected_class = 0;
        for(int j = 0; j < 10; j++) {
            if(train_labels[i][j][0] > 0.5) {
                expected_class = j;
                break;
            }
        }

        train_digit_total[expected_class]++;
        
        bool is_correct = (predicted_class == expected_class);
        if(is_correct) {
            train_correct++;
            train_digit_correct[expected_class]++;
        }

        printf("Train %2d: Expected=%d, Predicted=%d, Conf=%.4f %s\n",
               i + 1, expected_class, predicted_class, max_activation,
               is_correct ? "✓" : "✗");
    }

    printf("\nTraining Set Accuracy: %d/%d = %.2f%%\n", 
           train_correct, train_samples, 
           (train_correct * 100.0) / train_samples);

    printf("\nPer-Digit Training Accuracy:\n");
    for(int i = 0; i < 10; i++) {
        if(train_digit_total[i] > 0) {
            printf("  Digit %d: %d/%d = %.1f%%\n", 
                   i, train_digit_correct[i], train_digit_total[i],
                   (train_digit_correct[i] * 100.0) / train_digit_total[i]);
        }
    }

    // Test on TEST set (unseen data)
    printf("\n========================================================\n");
    printf("Testing on TEST SET (20 samples - UNSEEN DATA)\n");
    printf("========================================================\n\n");

    int test_correct = 0;
    int test_digit_correct[10] = {0};
    int test_digit_total[10] = {0};

    for(int i = 0; i < test_samples; i++) {
        for(int j = 0; j < 784; j++) {
            test_input[j][0] = test_data[i][j][0];
        }

        feed_forward(net, test_input);

        int predicted_class = 0;
        double max_activation = net.activations[net.num_of_layers-1][0][0];
        for(int j = 1; j < 10; j++) {
            if(net.activations[net.num_of_layers-1][j][0] > max_activation) {
                max_activation = net.activations[net.num_of_layers-1][j][0];
                predicted_class = j;
            }
        }

        int expected_class = 0;
        for(int j = 0; j < 10; j++) {
            if(test_labels[i][j][0] > 0.5) {
                expected_class = j;
                break;
            }
        }

        test_digit_total[expected_class]++;
        
        bool is_correct = (predicted_class == expected_class);
        if(is_correct) {
            test_correct++;
            test_digit_correct[expected_class]++;
        }

        printf("Test %2d: Expected=%d, Predicted=%d, Conf=%.4f %s\n",
               i + 1, expected_class, predicted_class, max_activation,
               is_correct ? "✓" : "✗");
    }

    printf("\n========================================================\n");
    printf("FINAL RESULTS\n");
    printf("========================================================\n\n");
    
    printf("Training Set Accuracy: %d/%d = %.2f%%\n", 
           train_correct, train_samples, 
           (train_correct * 100.0) / train_samples);
    
    printf("Test Set Accuracy:     %d/%d = %.2f%% ← GENERALIZATION\n\n", 
           test_correct, test_samples, 
           (test_correct * 100.0) / test_samples);
    
    printf("Per-Digit Test Accuracy:\n");
    for(int i = 0; i < 10; i++) {
        printf("  Digit %d: %d/%d = %.1f%%\n", 
               i, test_digit_correct[i], test_digit_total[i],
               (test_digit_correct[i] * 100.0) / test_digit_total[i]);
    }

   network_file("weights.bin", "bias.bin", net);
    printf("\n========================================================\n");

    // Cleanup
    free_memory_twoD(test_input, 784, 1);

    for(int i = 0; i < total_samples; i++){
        free_memory_twoD(all_data[i], input_size, 1);
        free_memory_twoD(all_labels[i], output_size, 1);
    }
    free(all_data);
    free(all_labels);
    free(training_data);
    free(train_labels);
    free(test_data);
    free(test_labels);

    for(int i = 0; i < net.num_of_layers - 1; i++){
        free_memory_twoD(net.weights[i], net.neurons_per_layer[i+1], net.neurons_per_layer[i]);
        free_memory_twoD(net.biases[i], net.neurons_per_layer[i+1], 1);
        free_memory_twoD(net.z_s[i], net.neurons_per_layer[i+1], 1);
        free_memory_twoD(net.delta_nabla_w[i], net.neurons_per_layer[i+1], net.neurons_per_layer[i]);
        free_memory_twoD(net.delta_nabla_b[i], net.neurons_per_layer[i+1], 1);
    }

    for(int i = 0; i < net.num_of_layers; i++){
        free_memory_twoD(net.activations[i], net.neurons_per_layer[i], 1);
    }


    free(net.weights);
    free(net.biases);
    free(net.activations);
    free(net.z_s);
    free(net.delta_nabla_w);
    free(net.delta_nabla_b);
    free(net.neurons_per_layer);

    return 0;
}
