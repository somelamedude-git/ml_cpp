#include <stdio.h>

void create_test_digit(Testing* test_data, int digit_type) {
    for(int i = 0; i < LAYER_ONE; i++) {
        test_data->inputs[i][0] = 0.0;
    }

    switch(digit_type) {
        case 0: 
            for(int row = 8; row < 20; row++) {
                for(int col = 8; col < 20; col++) {
                    int dist_sq = (row-14)*(row-14) + (col-14)*(col-14);
                    if(dist_sq > 20 && dist_sq < 50) {
                        test_data->inputs[row * 28 + col][0] = 0.8;
                    }
                }
            }
            break;

        case 1: 
            for(int row = 5; row < 23; row++) {
                test_data->inputs[row * 28 + 14][0] = 0.8;
                test_data->inputs[row * 28 + 13][0] = 0.6;
            }
            break;

        case 2: 
            for(int col = 8; col < 20; col++) {
                test_data->inputs[8 * 28 + col][0] = 0.8;
                test_data->inputs[15 * 28 + col][0] = 0.8;
                test_data->inputs[22 * 28 + col][0] = 0.8;
            }
            for(int row = 8; row < 22; row++) {
                int col = 8 + (20 - row);
                if(col >= 8 && col < 20) {
                    test_data->inputs[row * 28 + col][0] = 0.8;
                }
            }
            break;

        default:
            printf("Warning: Unknown digit type %d\n", digit_type);
            break;
    }
}

