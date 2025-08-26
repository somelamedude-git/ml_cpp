#include <iostream>
#include "math_expr.h"
#include "matrix_ops.h"
#include <cmath>

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

    double softplus(double x) {
    if (x > 0) {
        return x + log1p(exp(-x));
    } else {
        return log1p(exp(x));
    }
}


     void layer_calc_forwardprop(int size_next, int size_current, float weights[][MAX_COL_ONE], float biases[], float activation[], float resultant[1][MAX_COL_ONE]){
	   // Weights are going to have dimensions [size_current][size_next]
	   // biases are going to have dimensions [size_next]
	   // activation -> [size_current]

	    float two_D_mat[1][MAX_COL_ONE];
	    convert_twoD(activation, size_current, two_D_mat);
	    float product_matrix[1][MAX_COL_ONE];

	    matrix_multiplication(two_D_mat, weights, product_matrix, 1, size_current, size_next, size_current);

	    for(int j =0; j<size_next; j++){
		    product_matrix[0][j] = product_matrix[0][j]+biases[j];
		    resultant[0][j] = softplus(product_matrix[0][j]);
	    }
    }

};


int main(){
    ActivationFunctions af(1.0, 0.0);

    // Example: 2 input neurons -> 3 output neurons
    int size_current = 2;
    int size_next = 3;

    float weights[MAX_COL_ONE][MAX_COL_ONE] = {0};
    float biases[MAX_COL_ONE] = {0};
    float activation[MAX_COL_ONE] = {0};
    float resultant[1][MAX_COL_ONE] = {0};

    // Input activations
    activation[0] = 1.0;
    activation[1] = 2.0;

    // Weights: shape (2, 3)
    weights[0][0] = 0.5; weights[0][1] = -1.0; weights[0][2] = 2.0;
    weights[1][0] = 1.5; weights[1][1] = 0.3;  weights[1][2] = -0.7;

    // Biases
    biases[0] = 0.1;
    biases[1] = -0.2;
    biases[2] = 0.05;

    // Run forward propagation
    af.layer_calc_forwardprop(size_next, size_current, weights, biases, activation, resultant);

    cout << "Resultant activations after forward prop:" << endl;
    for(int j=0; j<size_next; j++){
        cout << "Neuron " << j << ": " << resultant[0][j] << endl;
    }

    return 0;
}
