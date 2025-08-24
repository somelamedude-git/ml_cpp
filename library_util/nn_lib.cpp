#include <iostream>
#include "math_expr.h"

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

    double softplus(){
        MathExpansions m1(this->x_axis);
        double softplus_val = MathExpansions(1 + m1.eExpand()).logarithm();

	return softplus_val;
    }
};
