/*** CONTAINS ALL TESTS FOR THE FULLY CONNECTED LAYER */

#include <stdio.h>
#include <stdlib.h>

#include "fully_conn_layer.h"
#include "activation_layer.h"

int main() {
    FILE* output_file = fopen("fully_conn_layer_tests.txt", "w");

    ReadVectFmt* expected_rvf = (ReadVectFmt*) malloc(sizeof(ReadVectFmt));
    ReadVectFmt* actual_rvf = (ReadVectFmt*) malloc(sizeof(ReadVectFmt));

    /** FULLY CONNECTED LAYER TESTS **/

    /* WEIGHTED SUM TEST AND OUTPUT GENERATION */
    float input_vect[] = {-1000000000, 0, 1};

    float opt_weights_1[] = {-1, 1, 2};
    float opt_weights_2[] = {2, -2, 3};

    float* opt_weights[] = {opt_weights_1, opt_weights_2};

    FullConnLayer fcl; fcl.build = fcl_builder; fcl.exec = fcl_exec; fcl.destroy = fcl_destroyer;
    fcl.build(&fcl.neurons, &fcl.num_neurons, input_vect, opt_weights, 2, 3);

    float expected_wstog[] = {1000000002, -1999999997};
    expected_rvf->vect = expected_wstog;
    expected_rvf->num_rows = 1;
    expected_rvf->num_cols = 2;
    expected_rvf->num_layers = 1;

    float* actual_wstog = fcl.exec(input_vect, fcl.neurons, fcl.num_neurons);
    actual_rvf->vect = actual_wstog;
    actual_rvf->num_rows = 1;
    actual_rvf->num_cols = 2;
    actual_rvf->num_layers = 1;

    disp_test_results("FULLY CONNECTED LAYER TESTS", "WEIGHTED SUM TEST AND OUTPUT GENERATION", 
                      (void*) expected_rvf, (void*) actual_rvf, 0, output_file);

    /* ACTIVATION LAYER OUTPUTS */

    ActLayer al = {(void*) expected_wstog, 1, 2, 1, leaky_relu, gen_output_img, gen_3d_output_img};

    float expected_alo[] = {1000000002, -19999999.97};
    expected_rvf->vect = expected_alo;
    expected_rvf->num_rows = 1;
    expected_rvf->num_cols = 2;
    expected_rvf->num_layers = 1;

    float* actual_alo = al.gen_output_img(al.img, al.num_rows, al.num_cols, al.num_channels, al.activation_function);
    actual_rvf->vect = actual_alo;
    actual_rvf->num_rows = 1;
    actual_rvf->num_cols = 2;
    actual_rvf->num_layers = 1;

    disp_test_results("FULLY CONNECTED LAYER TESTS", "ACTIVATION LAYER OUTPUTS ", 
                      (void*) expected_rvf, (void*) actual_rvf, 0, output_file);

    fclose(output_file);
}