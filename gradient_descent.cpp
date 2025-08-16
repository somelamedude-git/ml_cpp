#include <iostream>
#include <random>

using namespace std;

//The first three functions are just incase somebody wants to test on random numbers first, with linear form for now

int produce_random(){ // This is just producing random numbers, what order we need each element in, depends
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

int** matrix_mult(int rows, int cols, int** x_matrix, int** w_matrix){ // We still have rows and cols here, weight*x chalega, that's why this model for now
    // 1*cols, should be the dimension of w, flat arr
    for(int i =0; i<rows; i++){
        for(int j=0; i<cols; j++){

        }
    }
}

int* produce_defined_output(int m, int c, int* input_arr, int size){ // m is slope, c is the intercept :)
    int* y_axis = new int[size];

    for(int i = 0; i<size; i++){
        y_axis[i] = m*input_arr[i] + c;
    }
    return y_axis;
}

int main(){
   
}