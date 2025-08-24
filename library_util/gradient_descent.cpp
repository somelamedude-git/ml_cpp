#include <iostream>
#include <random>
#include "gradient_descent.h"

using namespace std;

//The first three functions are just incase somebody wants to test on random numbers first, with linear form for now

/*int produce_random(){ // This is just producing random numbers, what order we need each element in, depends
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0,99); // This will just produce the integers
    return dis(gen);
}

int** produce_random_arr(int size){ // perform reshape
    int** rand_arr = new int*[size]; // We will now assign addresses :)
    for(int i =0; i<size; i++){
        rand_arr[i] = new int[1];
        rand_arr[i][0] = produce_random(); // now we save actual addresses
    }
    return rand_arr;
}

int** produce_random_arr_matrix(int rows, int cols){
    int** matrix = new int[rows];
    for(int i =0; i<rows; i++){
        matrix[i] = produce_random_arr(cols);
    }
    return matrix;
}

int* flatten_array(int** matrix, int rows, int cols){
    int* flattened_array = new int[rows*cols];
    for(int i=0; i<rows; i++){
        for(int j =0; j<cols; j++){
            flattened_array[i*cols + j] = matrix[i][j];
        }
    }
    return flattened_array;
}

// I need the cycle to complete

unsigned short int* matrix_mult(int rows, int cols, int** x_matrix, int** w_matrix){ // We still have rows and cols here, weight*x chalega, that's why this model for now
    unsigned short int* flat_x = flatten_array(x_matrix, rows, cols);
    unsigned short int* flat_w = flatten_array(w_matrix, cols, 1);
    unsigned short int* y_matrix = new unsigned short int[rows]();
    int k = -1;

    for(int i =0; i<cols*rows; i++){
        if(i%cols==0) k++;
        y_matrix[k] += flat_x[i]*flat_w[i%cols];
    }

    delete[] flat_w;
    delete[] flat_x;
    return y_matrix;
}

// int compute_gradient(int** x_matrix, int** w_matrix, int** y_matrix, int rows, int cols){
//     unsigned short int* y_matrix = matrix_mul
// }


int* produce_defined_output(int* resultant_mult_matrix, int* constant_matrix, int size){ // m is slope, c is the intercept :)
    //both should have same size as cols did
    int* resultant_matrix = new int[size];
    for(int i =0; i<size; i++){
        resultant_matrix[i] = constant_matrix[i] + resultant_mult_matrix[i];
    }
    return resultant_matrix;
}*/

void compute_mean_error(float expected_output[], float output[], int size, float* mean_error){
	*mean_error = 0.0f;

	for(int i =0; i<size; i++){
		float diff = expected_output[i]-output[i];
		*mean_error += (float)(diff*diff);
	}
	*mean_error /= (float)size;
}

int main() {
    // Example arrays
    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[]   = {0.9f, 2.1f, 2.5f, 4.2f};
    int size = 4;

    float mean_error;
    compute_mean_error(expected, output, size, &mean_error);

    cout << "Mean Squared Error = " << mean_error << endl;

    return 0;
}
