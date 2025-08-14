#include <iostream>
#include <random>

using namespace std;

float produce_random(){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0,99); // This will just produce the integers
    return dis(gen);
}

int main(){
    cout<<produce_random()<<'\n';
}