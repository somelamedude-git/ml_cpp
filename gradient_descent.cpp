#include <iostream>
#include <random>

using namespace std;

//The first three functions are just incase somebody wants to test on random numbers first, with linear form for now

float produce_random(){ // This is just producing random numbers, what order we need each element in, depends
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0,99); // This will just produce the integers
    return dis(gen);
}

int* produce_random_arr(int size){
    int* rand_arr = new int[size];
    for(int i =0; i<size; i++){
        rand_arr[i] = produce_random();
    }
    return rand_arr;
}

int* produce_defined_output(int m, int c, int* input_arr, int size){ // m is slope, c is the intercept :)
    int* y_axis = new int[size];

    for(int i = 0; i<size; i++){
        y_axis[i] = m*input_arr[i] + c;
    }

}

int main(){
    int* rand_arr = produce_random_arr(5);
    for(int i =0; i<5; i++){
        cout<<rand_arr[i]<<' ';
    }
}