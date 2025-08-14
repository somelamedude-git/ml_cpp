#include <iostream>
#include <random>

using namespace std;

float produce_random(){
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

int main(){
    int* rand_arr = produce_random_arr(5);
    for(int i =0; i<5; i++){
        cout<<rand_arr[i]<<' ';
    }
}