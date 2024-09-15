#include <stdio.h>
#include <assert.h>
#include "img_proc_layer.h"

int main() {
    FILE* output_file = fopen("convo_layer_tests.txt", "w");

    /*** DEFINING IMAGE MATRICES OF VARYING DIMENSIONS FOR TESTS ***/
    float img_zero_by_zero[] = {}; // A 0x0 matrix
    float img_one_by_one[] = {3}; // A 1x1 matrix
    float img_one_by_dimen[] = {-2, 1, -1, 2, 3}; // A 1xn or nx1 matrix, where n is dimension magnitude
    float img_odd_by_odd[] = {-2, 1, -1, 2, 3, 0, -3, 1, 4, 3, 0, -4, 2, 1, -1, 3, 2, -1, 1, 0, 4, -4, 1, 2, 3}; // An oddxodd (5x5) matrix
    float img_even_by_even[] = {-2, 1, -1, 2, 0, -3, 1, 4, 0, -4, 2, 1, 3, 2, -1, 1}; // An evenxeven (4x4) matrix
    float img_r_by_c[] = {-2, 1, -1, 2, 0, -3, 1, 4, 0, -4, 2, 1}; // An rxc matrix, where r != c

    /*
    0 0  0  0 0 0
    0 0  0  0 0 0
    0 -2 1 -1 2 0
    0 0 -3  1 4 0
    0 0 -4  2 1 0
    0 0  0  0 0 0
    0 0  0  0 0 0

    -2  1
    0  -3

    */

    
    /*** DEFINING KERNEL MATRICES OF VARYING DIMENSIONS FOR TESTS ***/
    float kern_zero_by_zero[] = {}; // A 0x0 matrix
    float kern_one_by_one[] = {1}; // A 1x1 matrix
    float kern_one_by_dimen[] = {-2, 1, -1}; // A 1xn or nx1 matrix, where n is dimension magnitude
    float kern_odd_by_odd[] = {-2, 1, -1, 0, -3, 1, 0, -4, 2}; // An oddxodd (3x3) matrix
    float kern_even_by_even[] = {-2, 1, 0, -3}; // An evenxeven (2x2) matrix
    float kern_r_by_c[] = {-2, 1, -1, 0, -3, 1}; // An rxc matrix, where r != c


    /*** SUP_PRODUCT_SUMMATION TESTS ***/

    float actual_result;
    float expected_result;


    /** 0x0 kernel tests **/
    expected_result = 2;

    actual_result = sup_product_summation(kern_zero_by_zero, img_zero_by_zero, 0, 0, 0, 0, 0, 2);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times 0x0 img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 0;
    actual_result = sup_product_summation(kern_zero_by_zero, img_one_by_one, 0, 0, 1, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times 1x1 img", (void*) &expected_result,(void*) &actual_result,
                      1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_one_by_dimen, 0, 0, 5, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times 1xn img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_one_by_dimen, 0, 0, 1, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times nx1 img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_odd_by_odd, 0, 0, 5, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times oddxodd img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_even_by_even, 0, 0, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times evenxeve  img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_r_by_c, 0, 0, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times rxc (3x4) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 3;
    actual_result = sup_product_summation(kern_zero_by_zero, img_r_by_c, 0, 0, 3, 0, 0, 3);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times rxc (4x3) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);


    /** 1x1 kernel tests **/

    expected_result = 3;
    actual_result = sup_product_summation(kern_one_by_one, img_one_by_one, 1, 1, 1, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times 1x1 img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = -1;
    actual_result = sup_product_summation(kern_one_by_one, img_one_by_dimen, 1, 1, 5, 0, 0, 1);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times 1xn img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = -3;
    actual_result = sup_product_summation(kern_one_by_one, img_one_by_dimen, 1, 1, 1, 0, 0, -1);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times nx1 img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_odd_by_odd, 1, 1, 5, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times oddxodd img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_even_by_even, 1, 1, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times evenxeven img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_r_by_c, 1, 1, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times rxc (3x4) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_r_by_c, 1, 1, 3, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times rxc (4x3) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);


    /** 1xn kernel tests **/

    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_one_by_dimen, 1, 3, 5, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times 1xn img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_odd_by_odd, 1, 3, 5, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times oddxodd img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_even_by_even, 1, 3, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times evenxeven img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 10;
    actual_result = sup_product_summation(kern_one_by_dimen, img_r_by_c, 1, 3, 4, 0, 0, 4);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times rxc (3x4) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 2;
    actual_result = sup_product_summation(kern_one_by_dimen, img_r_by_c, 1, 3, 3, 0, 0, -4);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times rxc (4x3) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);


    /** nx1 kernel tests **/

    /* nx1 kern times nx1 img */
    expected_result = 4;
    actual_result = sup_product_summation(kern_one_by_dimen, img_one_by_dimen, 3, 1, 1, 0, 0, -2);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times nx1 img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    /* nx1 kern times oddxodd img */
    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_odd_by_odd, 3, 1, 5, 0, 0, 2);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times oddxodd img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    /* nx1 kern times evenxeven img */
    expected_result = 4;
    actual_result = sup_product_summation(kern_one_by_dimen, img_even_by_even, 3, 1, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times evenxeven img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    /* nx1 kern times rxc (3x4) img */
    expected_result = 7;
    actual_result = sup_product_summation(kern_one_by_dimen, img_r_by_c, 3, 1, 4, 0, 0, 3);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times rxc (3x4) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    /* nx1 kern times rxc (4x3) img */
    expected_result = 1;
    actual_result = sup_product_summation(kern_one_by_dimen, img_r_by_c, 3, 1, 3, 0, 0, -4);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times rxc (4x3) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);


    /** oddxodd kernel tests **/

    expected_result = 36;
    actual_result = sup_product_summation(kern_odd_by_odd, img_odd_by_odd, 3, 3, 5, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "oddxodd kern times oddxodd img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 36;
    actual_result = sup_product_summation(kern_odd_by_odd, img_even_by_even, 3, 3, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "oddxodd kern times evenxeven img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 38;
    actual_result = sup_product_summation(kern_odd_by_odd, img_r_by_c, 3, 3, 4, 0, 0, 2);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "oddxodd kern times rxc (3x4) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = -16;
    actual_result = sup_product_summation(kern_odd_by_odd, img_r_by_c, 3, 3, 3, 0, 0, -3);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "oddxodd kern times rxc (4x3) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);


    /** evenxeven kernel tests **/

    expected_result = 14;
    actual_result = sup_product_summation(kern_even_by_even, img_odd_by_odd, 2, 2, 5, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "evenxeven kern times oddxodd img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 14;
    actual_result = sup_product_summation(kern_even_by_even, img_even_by_even, 2, 2, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "evenxeven kern times evenxeven img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 16;
    actual_result = sup_product_summation(kern_even_by_even, img_r_by_c, 2, 2, 4, 0, 0, 2);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "evenxeven kern times rxc (3x4) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 7;
    actual_result = sup_product_summation(kern_even_by_even, img_r_by_c, 2, 2, 3, 0, 0, 2);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "evenxeven kern times rxc (4x3) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    /** rxc (2x3) kernel tests **/

    expected_result = 16;
    actual_result = sup_product_summation(kern_r_by_c, img_odd_by_odd, 2, 3, 5, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (2x3) kern times oddxodd img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 13;
    actual_result = sup_product_summation(kern_r_by_c, img_even_by_even, 2, 3, 4, 0, 0, -3);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (2x3) kern times evenxeven img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 17;
    actual_result = sup_product_summation(kern_r_by_c, img_r_by_c, 2, 3, 4, 0, 0, 1);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (2x3) kern times rxc (3x4) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 3;
    actual_result = sup_product_summation(kern_r_by_c, img_r_by_c, 2, 3, 3, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (2x3) kern times rxc (4x3) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    /** rxc (3x2) kernel tests **/

    expected_result = 2;
    actual_result = sup_product_summation(kern_r_by_c, img_odd_by_odd, 3, 2, 5, 0, 0, 1);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (3x2) kern times oddxodd img", (void*) &expected_result,(void*) &actual_result,
                      1, output_file);

    expected_result = 1;
    actual_result = sup_product_summation(kern_r_by_c, img_even_by_even, 3, 2, 4, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (3x2) kern times evenxeven img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 0;
    actual_result = sup_product_summation(kern_r_by_c, img_r_by_c, 3, 2, 4, 0, 0, -1);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (3x2) kern times rxc (3x4) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);

    expected_result = 4;
    actual_result = sup_product_summation(kern_r_by_c, img_r_by_c, 3, 2, 3, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (3x2) kern times rxc (4x3) img", (void*) &expected_result, (void*) &actual_result,
                      1, output_file);


    /*** CONVOLUTIONAL LAYER BUILDER TESTS ***/

    /**  **/



    /*** CONVOLUTION EXECUTION TESTS ***/
    ReadVectFmt* expected_rvf = (ReadVectFmt*) malloc(sizeof(ReadVectFmt));
    ReadVectFmt* actual_rvf = (ReadVectFmt*) malloc(sizeof(ReadVectFmt));

    /** IMG_WITH_PADDING TESTS (2D) **/
    float** img_2d = (float**) malloc(sizeof(float*));
    RowColTuple padding;

    /* NO PADDING */
    padding.rows = 0;
    padding.cols = 0;

    *img_2d = img_even_by_even;

    float expected_no_padding_nn[] = {-2, 1, -1, 2, 0, -3, 1, 4, 0, -4, 2, 1, 3, 2, -1, 1};
    expected_rvf->vect = expected_no_padding_nn;
    expected_rvf->num_rows = 4;
    expected_rvf->num_cols = 4;
    expected_rvf->num_layers = 1;

    float** actual_no_padding_nn = img_with_padding(img_2d, 4, 4, 1, padding);
    actual_rvf->vect = *actual_no_padding_nn;
    actual_rvf->num_rows = 4;
    actual_rvf->num_cols = 4;
    actual_rvf->num_layers = 1;

    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "NXN IMAGE NO PADDING", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_no_padding_nn, 1);

    *img_2d = img_r_by_c;

    float expected_no_padding_rc[] = {-2, 1, -1, 2, 0, -3, 1, 4, 0, -4, 2, 1};
    expected_rvf->vect = expected_no_padding_rc;
    expected_rvf->num_rows = 3;
    expected_rvf->num_cols = 4;

    float** actual_no_padding_rc = img_with_padding(img_2d, 3, 4, 1, padding);
    actual_rvf->vect = *actual_no_padding_rc;
    actual_rvf->num_rows = 3;
    actual_rvf->num_cols = 4;

    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "RXC IMAGE NO PADDING", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_no_padding_rc, 1);
    

    /* SAME ROW AND COLUMN DIMENSION PADDING */
    padding.rows = 1;
    padding.cols = 1;

    *img_2d = img_even_by_even;

    float expected_samerc_padding_nn[] = {0, 0, 0, 0, 0, 0, 
                                          0, -2, 1, -1, 2, 0,
                                          0, 0, -3, 1, 4, 0, 
                                          0, 0, -4, 2, 1, 0,
                                          0, 3, 2, -1, 1, 0,
                                          0, 0, 0, 0, 0, 0};
    expected_rvf->vect = expected_samerc_padding_nn;
    expected_rvf->num_rows = 6;
    expected_rvf->num_cols = 6;

    float** actual_samerc_padding_nn = img_with_padding(img_2d, 4, 4, 1, padding);
    actual_rvf->vect = *actual_samerc_padding_nn;
    actual_rvf->num_rows = 6;
    actual_rvf->num_cols = 6;
    
    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "NXN IMAGE SAME ROW COL PADDING", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_samerc_padding_nn, 1);

    *img_2d = img_r_by_c;

    float expected_samerc_padding_rc[] = {0, 0, 0, 0, 0, 0,
                                          0, -2, 1, -1, 2, 0,
                                          0, 0, -3, 1, 4, 0,
                                          0, 0, -4, 2, 1, 0,
                                          0, 0, 0, 0, 0, 0};
    expected_rvf->vect = expected_samerc_padding_rc;
    expected_rvf->num_rows = 5;
    expected_rvf->num_cols = 6;

    float** actual_samerc_padding_rc = img_with_padding(img_2d, 3, 4, 1, padding);
    actual_rvf->vect = *actual_samerc_padding_rc;
    actual_rvf->num_rows = 5;
    actual_rvf->num_cols = 6;
    
    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "RXC IMAGE SAME ROW COL PADDING", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_samerc_padding_rc, 1);

    /* ARBITRARY ROW AND COLUMN DIMENSION PADDING */
    padding.rows = 2;
    padding.cols = 3;

    *img_2d = img_even_by_even;
    float expected_arbitrary_padding_nn[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, -2, 1, -1, 2, 0, 0, 0,
                                             0, 0, 0, 0, -3, 1, 4, 0, 0, 0,
                                             0, 0, 0, 0, -4, 2, 1, 0, 0, 0,
                                             0, 0, 0, 3, 2, -1, 1, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expected_rvf->vect = expected_arbitrary_padding_nn;
    expected_rvf->num_rows = 8;
    expected_rvf->num_cols = 10;
    
    float** actual_arbitrary_padding_nn = img_with_padding(img_2d, 4, 4, 1, padding);
    actual_rvf->vect = *actual_arbitrary_padding_nn;
    actual_rvf->num_rows = 8;
    actual_rvf->num_cols = 10;

    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "NXN IMAGE ARBITRARY PADDING", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_arbitrary_padding_nn, 1);

    *img_2d = img_r_by_c;

    float expected_arbitrary_padding_rc[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, -2, 1, -1, 2, 0, 0, 0,
                                             0, 0, 0, 0, -3, 1, 4, 0, 0, 0,
                                             0, 0, 0, 0, -4, 2, 1, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expected_rvf->vect = expected_arbitrary_padding_rc;
    expected_rvf->num_rows = 7;
    expected_rvf->num_cols = 10;

    float** actual_arbitrary_padding_rc = img_with_padding(img_2d, 3, 4, 1, padding);
    actual_rvf->vect = *actual_arbitrary_padding_rc;
    actual_rvf->num_rows = 7;
    actual_rvf->num_cols = 10;

    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "RXC IMAGE ARBITRARY PADDING", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_arbitrary_padding_rc, 1);

    free(img_2d);


    /** CALC_OUTPUT_DIMEN_SIZE TESTS **/

    /* KERNEL DIMEN SIZE AND IMG DIMEN SIZE ARE 1 */
    expected_result = 1;
    actual_result = calc_output_dimen_size(1, 1, 0, 1);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", "KERNEL DIMEN SIZE AND IMG DIMEN SIZE ARE 1", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE AND IMG DIMEN SIZE ARE THE SAME */
    expected_result = 1;
    actual_result = calc_output_dimen_size(3, 3, 0, 1);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", "KERNEL DIMEN SIZE AND IMG DIMEN SIZE ARE 1", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (NO PADDING, STRIDE OF 1) */
    expected_result = 5;
    actual_result = calc_output_dimen_size(5, 1, 0, 1);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (NO PADDING, STRIDE OF 1)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (NO PADDING, REASONABLE STRIDE GREATER THAN 1) */
    expected_result = 3;
    actual_result = calc_output_dimen_size(5, 1, 0, 2);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (NO PADDING, REASONABLE STRIDE GREATER THAN 1)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (NO PADDING, TOO LARGE STRIDE) */
    expected_result = 1;
    actual_result = calc_output_dimen_size(5, 1, 0, 6);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (NO PADDING, TOO LARGE STRIDE)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (PADDING, STRIDE OF 1) */
    expected_result = 9;
    actual_result = calc_output_dimen_size(5, 1, 2, 1);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (PADDING, STRIDE OF 1)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (PADDING, REASONABLE STRIDE GREATER THAN 1) */
    expected_result = 5;
    actual_result = calc_output_dimen_size(5, 1, 2, 2);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (PADDING, REASONABLE STRIDE GREATER THAN 1)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (PADDING, TOO LARGE STRIDE) */
    expected_result = 1;
    actual_result = calc_output_dimen_size(5, 1, 2, 15);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS 1 WHILE IMG DIMEN SIZE IS GREATER THAN 1 (PADDING, TOO LARGE STRIDE)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (NO PADDING, STRIDE OF 1) */
    expected_result = 4;
    actual_result = calc_output_dimen_size(5, 2, 0, 1);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (NO PADDING, STRIDE OF 1)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (NO PADDING, REASONABLE STRIDE GREATER THAN 1) */
    expected_result = 2;
    actual_result = calc_output_dimen_size(5, 2, 0, 2);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (NO PADDING, REASONABLE STRIDE GREATER THAN 1)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (NO PADDING, TOO LARGE STRIDE) */
    expected_result = 1;
    actual_result = calc_output_dimen_size(5, 2, 0, 15);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (NO PADDING, TOO LARGE STRIDE)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (PADDING, STRIDE OF 1) */
    expected_result = 8;
    actual_result = calc_output_dimen_size(5, 2, 2, 1);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (PADDING, STRIDE OF 1)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (PADDING, REASONABLE STRIDE GREATER THAN 1) */
    expected_result = 4;
    actual_result = calc_output_dimen_size(5, 2, 2, 2);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (PADDING, REASONABLE STRIDE GREATER THAN 1)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);

    /* KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (PADDING, TOO LARGER STRIDE) */
    expected_result = 1;
    actual_result = calc_output_dimen_size(5, 2, 2, 15);
    disp_test_results("CALC_OUTPUT_DIMEN_SIZE TESTS", 
                      "KERNEL DIMEN SIZE IS GREATER THAN 1 WHILE DIMEN SIZE IS GREATER THAN 1 (PADDING, TOO LARGER STRIDE)", 
                      (void*) &expected_result, (void*) &actual_result, 1, output_file);


    /** CONVOLUTION TESTS **/

    float** img = (float**) malloc(sizeof(float*));

    padding.rows = 0;
    padding.cols = 0;

    RowColTuple stride = {1, 1};

    /* USING ODD BY ODD KERNEL ON EVEN BY EVEN IMAGE */
    *img = img_even_by_even;

    ImgProcLayer convl; convl.build = builder; convl.exec = exec; convl.destroy = destroyer;
    convl.build(CONVOLUTIONAL, &convl.kernels, &convl.kernels3d, &convl.num_kernels, &convl.num_channels, 1, 1, 3, 3, 0, kern_odd_by_odd);
    
    float expected_output_obo[] = {36, -10, 0, 4};
    expected_rvf->vect = expected_output_obo;
    expected_rvf->num_rows = 2;
    expected_rvf->num_cols = 2;

    float** actual_output_obo = convl.exec(convl.kernels, convl.kernels3d, convl.num_kernels, convl.num_channels, 3, 3, img, 4, 4, 
                                            padding, stride);
    actual_rvf->vect = *actual_output_obo;
    actual_rvf->num_rows = 2;
    actual_rvf->num_cols = 2;

    disp_test_results("CONVOLUTION TESTS", 
                      "USING ODD BY ODD KERNEL ON EVEN BY EVEN IMAGE", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_output_obo, 1);

    convl.destroy(&convl.kernels, &convl.kernels3d, convl.num_kernels);

    /* USING EVEN BY EVEN KERNEL ON EVEN BY EVEN IMAGE */
    convl.build(CONVOLUTIONAL, &convl.kernels, &convl.kernels3d, &convl.num_kernels, &convl.num_channels, 1, 1, 2, 2, 0, kern_even_by_even);
    
    float expected_output_ebe[] = {14, -6, -8, 9, 1, -1, -10, 13, -6};
    expected_rvf->vect = expected_output_ebe;
    expected_rvf->num_rows = 3;
    expected_rvf->num_cols = 3;

    float** actual_output_ebe = convl.exec(convl.kernels, convl.kernels3d, convl.num_kernels, convl.num_channels, 2, 2, img, 4, 4, 
                                            padding, stride);
    actual_rvf->vect = *actual_output_ebe;
    actual_rvf->num_rows = 3;
    actual_rvf->num_cols = 3;

    disp_test_results("CONVOLUTION TESTS", 
                      "USING EVEN BY EVEN KERNEL ON EVEN BY EVEN IMAGE", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_output_ebe, 1);

    convl.destroy(&convl.kernels, &convl.kernels3d, convl.num_kernels);

    /* USING EVEN BY ODD KERNEL ON EVEN BY EVEN IMAGE */
    convl.build(CONVOLUTIONAL, &convl.kernels, &convl.kernels3d, &convl.num_kernels, &convl.num_channels, 1, 1, 2, 3, 1, kern_r_by_c);
    
    float expected_output_ebo[] = {17, -3, 11, -1, -12, 14};
    expected_rvf->vect = expected_output_ebo;
    expected_rvf->num_rows = 3;
    expected_rvf->num_cols = 2;

    float** actual_output_ebo = convl.exec(convl.kernels, convl.kernels3d, convl.num_kernels, convl.num_channels, 2, 3, img, 4, 4, 
                                            padding, stride);
    actual_rvf->vect = *actual_output_ebo;
    actual_rvf->num_rows = 3;
    actual_rvf->num_cols = 2;

    disp_test_results("CONVOLUTION TESTS", 
                      "USING EVEN BY ODD KERNEL ON EVEN BY EVEN IMAGE", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_output_ebo, 1);

    convl.destroy(&convl.kernels, &convl.kernels3d, convl.num_kernels);

    /* USING ODD BY EVEN KERNEL ON EVEN BY EVEN IMAGE */
    convl.build(CONVOLUTIONAL, &convl.kernels, &convl.kernels3d, &convl.num_kernels, &convl.num_channels, 1, 1, 3, 2, -2, kern_r_by_c);
    
    float expected_output_obe[] = {-1, 12, -4, -12, 2, 2};
    expected_rvf->vect = expected_output_obe;
    expected_rvf->num_rows = 2;
    expected_rvf->num_cols = 3;

    float** actual_output_obe = convl.exec(convl.kernels, convl.kernels3d, convl.num_kernels, convl.num_channels, 3, 2, img, 4, 4, 
                                            padding, stride);
    actual_rvf->vect = *actual_output_obe;
    actual_rvf->num_rows = 2;
    actual_rvf->num_cols = 3;

    disp_test_results("CONVOLUTION TESTS", 
                      "USING ODD BY EVEN KERNEL ON EVEN BY EVEN IMAGE", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_output_obe, 1);

    convl.destroy(&convl.kernels, &convl.kernels3d, convl.num_kernels);

    /* NO PADDING AND STRIDE 1 */
    *img = img_r_by_c;
    convl.build(CONVOLUTIONAL, &convl.kernels, &convl.kernels3d, &convl.num_kernels, &convl.num_channels, 1, 1, 2, 3, 3, kern_r_by_c);

    float expected_output_nps1[] = {19, -1, 13, 1};
    expected_rvf->vect = expected_output_nps1;
    expected_rvf->num_rows = 2;
    expected_rvf->num_cols = 2;

    float** actual_output_nps1 = convl.exec(convl.kernels, convl.kernels3d, convl.num_kernels, convl.num_channels, 2, 3, img, 3, 4, 
                                            padding, stride);
    actual_rvf->vect = *actual_output_nps1;
    actual_rvf->num_rows = 2;
    actual_rvf->num_cols = 2;

    disp_test_results("CONVOLUTION TESTS", 
                      "NO PADDING AND STRIDE 1", (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_output_nps1, 1);

    convl.destroy(&convl.kernels, &convl.kernels3d, convl.num_kernels);

    /* NO PADDING AND REASONABLE COL STRIDE AND UNREASONABLE ROW STRIDE */
    convl.build(CONVOLUTIONAL, &convl.kernels, &convl.kernels3d, &convl.num_kernels, &convl.num_channels, 1, 1, 2, 3, -1, kern_r_by_c);

    stride.rows = 2;
    stride.cols = 2;

    float expected_output_nprcsurs[] = {2, -4};
    expected_rvf->vect = expected_output_nprcsurs;
    expected_rvf->num_rows = 2;
    expected_rvf->num_cols = 1;

    float** actual_output_nprcsurs = convl.exec(convl.kernels, convl.kernels3d, convl.num_kernels, convl.num_channels, 2, 3, img, 4, 3, 
                                                padding, stride);
    actual_rvf->vect = *actual_output_nprcsurs;
    actual_rvf->num_rows = 2;
    actual_rvf->num_cols = 1;

    disp_test_results("CONVOLUTION TESTS", 
                      "NO PADDING AND REASONABLE ROW STRIDE AND UNREASONABLE COLUMN STRIDE", 
                      (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_output_nprcsurs, 1);

    convl.destroy(&convl.kernels, &convl.kernels3d, convl.num_kernels);

    /* ARBITRARY PADDING AND ARBITRARY STRIDES */
    convl.build(CONVOLUTIONAL, &convl.kernels, &convl.kernels3d, &convl.num_kernels, &convl.num_channels, 1, 1, 2, 2, 0, kern_even_by_even);

    padding.rows = 2;
    padding.cols = 1;

    stride.rows = 2;
    stride.cols = 3;

    float expected_output_apas[] = {0, 0, -2, -8, 0, -3};
    expected_rvf->vect = expected_output_apas;
    expected_rvf->num_rows = 3;
    expected_rvf->num_cols = 2;

    float** actual_output_apas = convl.exec(convl.kernels, convl.kernels3d, convl.num_kernels, convl.num_channels, 2, 2, img, 3, 4, 
                                             padding, stride);
    actual_rvf->vect = *actual_output_apas;
    actual_rvf->num_rows = 3;
    actual_rvf->num_cols = 2;

    disp_test_results("CONVOLUTION TESTS", 
                      "ARBITRARY PADDING AND ARBITRARY STRIDES", 
                      (void*) expected_rvf, (void*) actual_rvf, 0, output_file);
    free_img_rsrcs(actual_output_apas, 1);

    convl.destroy(&convl.kernels, &convl.kernels3d, convl.num_kernels);

    free(expected_rvf);
    free(actual_rvf);

    free(img);

    fclose(output_file);
}