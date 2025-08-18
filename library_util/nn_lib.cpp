#include <iostream>
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

    double compute_x_axis(double input){
        return (input*this->weight)+this->bias;
    }

    double relu_function()
}