#include <iostream>
#include "gradient_descent.h"

unsigned short int** calculate_square_of_matrix(unsigned short int** matrix, int rows, int cols){

}

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