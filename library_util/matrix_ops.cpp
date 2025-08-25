#include <iostream>
#include <stdexcept>
#include "gradient_descent.h"
#include "matrix_ops.h"

#define MAX_COL_ONE 100
#define MAX_COL_TWO 100

using namespace std;

// unsigned short int** calculate_square_of_matrix(unsigned short int** matrix, int rows, int cols){

// }

// original matrix is rows*cols, which means i*j, you get the point, transpose will be cols* rows, which means i 
// will be cols and j will be rows
unsigned short int** transpose(unsigned short int** matrix, int rows, int cols){
    unsigned short int** transposed_matrix = new unsigned short int*[cols]; // we are going to be insreting shit like 
    // [], [] to it, so alloting cols makes sense

    for(int i =0; i<cols; i++){
        transposed_matrix[i] = new unsigned short int[rows];
    }

    for(int i =0; i<rows; i++){
        for(int j =0; j<cols; j++){
            transposed_matrix[j][i] = matrix[i][j];
        }
    }

    return transposed_matrix;
}

// put the input in such a way such that matrix_one*matrix_two
// strassen's algorithm here will create unnecessary overhead
void matrix_multiplication(
    float matrix_one[][MAX_COL_ONE],
    float matrix_two[][MAX_COL_TWO],
    float product_matrix[][MAX_COL_TWO],
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

void convert_twoD(float single_D_mat[], int size, float two_D_mat[1][MAX_COL_ONE]){
	for(int j =0; j<size; j++){
		two_D_mat[0][j] = single_D_mat[j];
	}
}


unsigned short int** create_matrix(unsigned short int* arr, int size_of_arr, int rows, int cols){
    if(size_of_arr != (rows*cols)){
        throw runtime_error("Number of elements don't correspond to the dimension");
    }

   unsigned short int** matrix = new unsigned short int*[rows];
    for(int i =0; i<rows; i++){
        matrix[i] = new unsigned short int[cols];
    }

    for(int i =0; i<rows; i++){
        for(int j =0; j<cols; j++){
            matrix[i][j] = arr[i*cols + j];
        }
    }

    return matrix;
}

