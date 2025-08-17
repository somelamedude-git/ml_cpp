#include <iostream>
#include <stdexcept>
#include "gradient_descent.h"

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
int** matrix_multiplication(unsigned short int** matrix_one, unsigned short int** matrix_two, int col_one, int col_two, int row_one, int row_two){
    if(col_one != row_two){
        throw runtime_error("col_one must be equal to row_two");
    }

    int** product_matrix = new int*[row_one]; // dimension of new matrix will be row_one*col_two

    for(int i =0; i<row_one; i++){
        product_matrix[i] = new int[col_two]();
    }

    for(int i =0; i<row_one; i++){
        for(int k=0; k<col_two; k++){
            for(int j=0; j<row_two; j++){
                product_matrix[i][k] += (matrix_one[i][j]*matrix_two[j][]);
            }
        }
    }

    return product_matrix;
}

int main(){

}