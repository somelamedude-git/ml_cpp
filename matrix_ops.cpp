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
                product_matrix[i][k] += (matrix_one[i][j]*matrix_two[j][k]);
            }
        }
    }

    return product_matrix;
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

int main(){
      unsigned short  int arr[] = {0,1,2,3,4,5,6,7,8};  
    int rows = 3, cols = 3;

    try {
        unsigned short int** matrix = create_matrix(arr, 9, rows, cols);

        cout << "Matrix:" << endl;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                cout << matrix[i][j] << " ";
            }
            cout << endl;
        }

        for(int i = 0; i < rows; i++){
            delete[] matrix[i];
        }
        delete[] matrix;
    }
    catch (const runtime_error& e){
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
