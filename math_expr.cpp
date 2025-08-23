#include <iostream>

// This library because basys3 doesn't have inbuilt math thingd

class MathExpansions{
    private:
        double x;
    public:
        MathExpansions(double x){
            this->x = x;
        }

	double logarithm(){
		double output = (x-1) - ((x-1)*(x-1))/2 + ((x-1)*(x-1)*(x-1))/3 - ((x-1)*(x-1)*(x-1)*(x-1))/4;
		return output;
	}

	double eExpand(){
		double output = 1 + x + (x*x)/2 + (x*x*x)/6 + (x*x*x*x)/24 + (x*x*x*x*x)/120;
		return output;
	}
}
