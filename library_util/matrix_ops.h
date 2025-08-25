#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#define MAX_COL_ONE 100
#define MAX_COL_TWO 100 

unsigned short int** transpose(unsigned short int** matrix, int rows, int cols);
void matrix_multiplication(float matrix_one[][MAX_COL_ONE], float matrix_two[][MAX_COL_TWO],
		float product_matrix[][MAX_COL_TWO], int row_one, int col_one, int row_two, int col_two);

void convert_twoD(float single_D_mat[], int size, float two_D_mat[1][MAX_COL_ONE]);
#endif
