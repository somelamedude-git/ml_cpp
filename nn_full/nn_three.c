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



int main(){
    // XOR problem setup with 4 discrete input-output pairs
    int num_samples = 4;
    int input_size = 2;
    int output_size = 1;

    // Allocate training data and labels
    double*** training_data = (double***)malloc(sizeof(double**) * num_samples);
    double*** labels = (double***)malloc(sizeof(double**) * num_samples);

    // Initialize XOR data
    training_data[0] = create_matrix(input_size, 1);
    training_data[0][0][0] = 0.0;
    training_data[0][1][0] = 0.0;
    labels[0] = create_matrix(output_size, 1);
    labels[0][0][0] = 0.0;

    training_data[1] = create_matrix(input_size, 1);
    training_data[1][0][0] = 0.0;
    training_data[1][1][0] = 1.0;
    labels[1] = create_matrix(output_size, 1);
    labels[1][0][0] = 1.0;

    training_data[2] = create_matrix(input_size, 1);
    training_data[2][0][0] = 1.0;
    training_data[2][1][0] = 0.0;
    labels[2] = create_matrix(output_size, 1);
    labels[2][0][0] = 1.0;

    training_data[3] = create_matrix(input_size, 1);
    training_data[3][0][0] = 1.0;
    training_data[3][1][0] = 1.0;
    labels[3] = create_matrix(output_size, 1);
    labels[3][0][0] = 0.0;

    // Network architecture: 2 inputs -> 2 hidden -> 1 output
    int num_layers = 3;
    // A smaller hidden layer is sufficient and more efficient for XOR.
    int neurons_per_layer[] = {2, 2, 1};

    printf("Initializing Neural Network for XOR problem...\n");
    printf("Architecture: %d-%d-%d\n", neurons_per_layer[0], neurons_per_layer[1], neurons_per_layer[2]);
    printf("================================================\n\n");

    Network net = initialize_network(num_layers, neurons_per_layer);

    // Training parameters
    // Reduce epochs; with a good learning rate, 100k is often more than enough.
    // Use a slightly lower, more stable learning rate.
    int epochs = 100000;
    int batch_size = 4; // Full-batch gradient descent is suitable for this tiny dataset
    double learning_rate = 0.05;

    printf("Training parameters:\n");
    printf("  Epochs: %d\n", epochs);
    printf("  Batch size: %d\n", batch_size);
    printf("  Learning rate: %.2f\n\n", learning_rate);

    printf("Training started...\n");
    sgd(net, epochs, training_data, labels, num_samples, batch_size, learning_rate);
    printf("Training completed!\n\n");

    // Test the network
    printf("================================================\n");
    printf("Testing XOR Gate:\n");
    printf("================================================\n\n");

    int max_layer_size = neurons_per_layer[0];
    for(int i = 1; i < num_layers; i++){
        if(neurons_per_layer[i] > max_layer_size){
            max_layer_size = neurons_per_layer[i];
        }
    }
    double** test_input = create_matrix(max_layer_size, 1);
    int correct_predictions = 0;

    for(int i = 0; i < num_samples; i++){
        test_input[0][0] = training_data[i][0][0];
        test_input[1][0] = training_data[i][1][0];

        feed_forward(net, test_input);

        double predicted = net.activations[net.num_of_layers-1][0][0];
        double expected = labels[i][0][0];

        int predicted_class = (predicted >= 0.5) ? 1 : 0;
        int expected_class = (expected >= 0.5) ? 1 : 0;

        if(predicted_class == expected_class){
            correct_predictions++;
        }

        printf("Input: [%.0f, %.0f]  |  Expected: %.0f  |  Predicted: %.4f  |  Class: %d  |  %s\n",
               training_data[i][0][0],
               training_data[i][1][0],
               expected,
               predicted,
               predicted_class,
               (predicted_class == expected_class) ? "✓ CORRECT" : "✗ WRONG");
    }

    printf("\n================================================\n");
    printf("Results:\n");
    printf("================================================\n");
    printf("Correct predictions: %d/%d\n", correct_predictions, num_samples);
    printf("Accuracy: %.2f%%\n", (correct_predictions * 100.0) / num_samples);
    printf("================================================\n");

    free_memory_twoD(test_input, max_layer_size, 1);

    for(int i = 0; i < num_samples; i++){
        free_memory_twoD(training_data[i], input_size, 1);
        free_memory_twoD(labels[i], output_size, 1);
    }
    free(training_data);
    free(labels);

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

