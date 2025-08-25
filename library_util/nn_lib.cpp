#include <iostream>
#include "math_expr.h"
#include "matrix_ops.h"

#define MAX_COL_ONE 100

using namespace std;

class ActivationFunctions{
    private:
        double weight;
        double bias;
        double x_axis;

    public:
    ActivationFunctions(double weight, double bias){
        this->weight = weight;
        this->bias = bias;
        this->x_axis = 0;
    }

    void compute_x_axis(double input){
        this->x_axis= (input*this->weight)+this->bias;
    }

    double softplus(){ // To find activation
        MathExpansions m1(this->x_axis);
        double softplus_val = MathExpansions(1 + m1.eExpand()).logarithm();

	return softplus_val;
    }

    float[] layer_calc_forwardprop(int size_next, int size_current, float weights[][MAX_COL_ONE], float biases[], float activation[], float resultant[1][MAX_COL_ONE]){
	   // Weights are going to have dimensions [size_current][size_next]
	   // biases are going to have dimensions [size_next]
	   // activation -> [size_current]

	    float two_D_mat[1][MAX_COL_ONE];
	    convert_twoD(activation, size_current, two_D_mat);
	    float product_matrix[1][size_next];

	    matrix_multiplication(two_D_mat, weights, product_matrix, 1, size_current, size_next, size_current);

	    for(int j =0; j<size_next; j++){
		    product_matrix[0][j] = product_matrix[0][j]+biases[j];
		    resultant[0][j] = softplus(product_matrix[0][j]);
	    }
    }






};
