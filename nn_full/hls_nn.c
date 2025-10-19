#include <stdio.h>

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
	double inputs[LAYER_ONE][1];
	double outputs[LAYER_FOUR][1]; // Optional to fill
	double output_predicted[LAYER_FOUR][1];
	double accuracy; // Optional
} Testing;


double sigmoid(double x){
        double y = 1.0/(1.0 + exp(-x));
        return y;
}


void sigmoid_list(double x[LAYER_ONE][1], int list_size, double output[LAYER_ONE][1]){
        for(int i =0; i<list_size; i++){
		#pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=49

                output[i][0] = sigmoid(x[i][0]);
        }
}

void matrix_multiplication(double matrix_one[LAYER_ONE][LAYER_ONE], double matrix_two[LAYER_ONE][1], double product_matrix[LAYER_ONE][1], int row_one,
                int col_one, int row_two, int col_two){

        for(int i =0; i<row_one; i++){
		#pragma HLS PIPELINE
		#pragma HLS UNROLL factor=49
                for(int j =0; j<col_two; j++){
                        for(int k =0; k<col_one; k++){
                                product_matrix[i][k] += matrix_one[i][j]*matrix_two[j][k];
                        }
                }
        }
}

void matrix_addition(double matrix_one[LAYER_ONE][1], double matrix_two[LAYER_ONE][1], double addition_matrix[LAYER_ONE][1], int rows, int cols){
        for(int i =0; i<rows; i++){
		#pragma HLS PIPELINE
		#pragma HLS UNROLL factor=49
                for(int j =0; j<cols; j++){
                        addition_matrix[i][j] = matrix_one[i][j] + matrix_two[i][j];
                }
        }
}


void feed_forward(Network* net, Testing* data){
       int size_of_input = net->neurons_per_layer[0];
       for(int i = 0; i < size_of_input; i++){
		#pragma HLS PIPELINE
		#pragma HLS UNROLL factor=49
               net->activations[0][i][0] = data->inputs[i][0];
        }

	double product_matrix[LAYER_ONE][1];
	double addition_matrix[LAYER_ONE][1];
	double activated_output[LAYER ONE][1];

	 for(int i =0; i<net->num_of_layers-1; i++){
		#pragma HLS PIPELINE
		 int weight_rows = net->neurons_per_layer[i+1];
		 int weight_cols = net->neurons_per_layer[i];

		 matrix_multiplication(net->weights[i], net->activations[i], product_matrix, weight_rows, weight_cols, net->neurons_per_layer[i], 1);
		 matrix_addition(net->biases[i], product_matrix, addition_matrix, weight_rows, 1);

		 for(int j =0; j<weight_rows; j++){
			#pragma HLS PIPELINE
			 #pragma HLS UNROLL factor=16

			 net->z_s[i][j][0] = addition_matrix[j][0];
		 }

		 sigmoid_list(net->z_s[i], weight_rows, activated_output);

		 for(int j =0; j<weight_rows; j++){
			#pragma HLS PIPELINE
			 #pragma HLS UNROLL factor=16
			 net->activations[i+1][j][0] = activated_output[j][0];
		 }

		}
}






