#include <stdio.h>
#include <assert.h>
#include "convo_layer.h"

int main() {
    FILE* output_file = fopen("convo_layer_tests.txt", "w");

    /*** DEFINING IMAGE MATRICES OF VARYING DIMENSIONS FOR TESTS ***/
    float img_zero_by_zero[] = {}; // A 0x0 matrix
    float img_one_by_one[] = {3}; // A 1x1 matrix
    float img_one_by_dimen[] = {-2, 1, -1, 2, 3}; // A 1xn or nx1 matrix, where n is dimension magnitude
    float img_odd_by_odd[] = {-2, 1, -1, 2, 3, 0, -3, 1, 4, 3, 0, -4, 2, 1, -1, 3, 2, -1, 1, 0, 4, -4, 1, 2, 3}; // An oddxodd (5x5) matrix
    float img_even_by_even[] = {-2, 1, -1, 2, 0, -3, 1, 4, 0, -4, 2, 1, 3, 2, -1, 1}; // An evenxeven (4x4) matrix
    float img_r_by_c[] = {-2, 1, -1, 2, 0, -3, 1, 4, 0, -4, 2, 1}; // An rxc matrix, where r != c

    
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
    expected_result = 0;

    actual_result = sup_product_summation(kern_zero_by_zero, img_zero_by_zero, 0, 0, 0, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times 0x0 img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_one_by_one, 0, 0, 1, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times 1x1 img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_one_by_dimen, 0, 0, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times 1xn img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_one_by_dimen, 0, 0, 1, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times nx1 img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_odd_by_odd, 0, 0, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times oddxodd img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_even_by_even, 0, 0, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times evenxeve  img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_r_by_c, 0, 0, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times rxc (3x4) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    actual_result = sup_product_summation(kern_zero_by_zero, img_r_by_c, 0, 0, 3, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "0x0 kern times rxc (4x3) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);


    /** 1x1 kernel tests **/

    expected_result = 3;
    actual_result = sup_product_summation(kern_one_by_one, img_one_by_one, 1, 1, 1, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times 1x1 img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_one_by_dimen, 1, 1, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times 1xn img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_one_by_dimen, 1, 1, 1, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times nx1 img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_odd_by_odd, 1, 1, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times oddxodd img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_even_by_even, 1, 1, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times evenxeven img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_r_by_c, 1, 1, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times rxc (3x4) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = -2;
    actual_result = sup_product_summation(kern_one_by_one, img_r_by_c, 1, 1, 3, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1x1 kern times rxc (4x3) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);


    /** 1xn kernel tests **/

    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_one_by_dimen, 1, 3, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times 1xn img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_odd_by_odd, 1, 3, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times oddxodd img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_even_by_even, 1, 3, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times evenxeven img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_r_by_c, 1, 3, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times rxc (3x4) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_r_by_c, 1, 3, 3, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "1xn kern times rxc (4x3) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);


    /** nx1 kernel tests **/

    /* nx1 kern times nx1 img */
    expected_result = 6;
    actual_result = sup_product_summation(kern_one_by_dimen, img_one_by_dimen, 3, 1, 1, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times nx1 img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    /* nx1 kern times oddxodd img */
    expected_result = 4;
    actual_result = sup_product_summation(kern_one_by_dimen, img_odd_by_odd, 3, 1, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times oddxodd img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    /* nx1 kern times evenxeven img */
    expected_result = 4;
    actual_result = sup_product_summation(kern_one_by_dimen, img_even_by_even, 3, 1, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times evenxeven img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    /* nx1 kern times rxc (3x4) img */
    expected_result = 4;
    actual_result = sup_product_summation(kern_one_by_dimen, img_r_by_c, 3, 1, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times rxc (3x4) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    /* nx1 kern times rxc (4x3) img */
    expected_result = 5;
    actual_result = sup_product_summation(kern_one_by_dimen, img_r_by_c, 3, 1, 3, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "nx1 kern times rxc (4x3) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);


    /** oddxodd kernel tests **/

    expected_result = 36;
    actual_result = sup_product_summation(kern_odd_by_odd, img_odd_by_odd, 3, 3, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "oddxodd kern times oddxodd img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 36;
    actual_result = sup_product_summation(kern_odd_by_odd, img_even_by_even, 3, 3, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "oddxodd kern times evenxeven img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 36;
    actual_result = sup_product_summation(kern_odd_by_odd, img_r_by_c, 3, 3, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "oddxodd kern times rxc (3x4) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = -13;
    actual_result = sup_product_summation(kern_odd_by_odd, img_r_by_c, 3, 3, 3, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "oddxodd kern times rxc (4x3) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);


    /** evenxeven kernel tests **/

    expected_result = 14;
    actual_result = sup_product_summation(kern_even_by_even, img_odd_by_odd, 2, 2, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "evenxeven kern times oddxodd img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 14;
    actual_result = sup_product_summation(kern_even_by_even, img_even_by_even, 2, 2, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "evenxeven kern times evenxeven img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 14;
    actual_result = sup_product_summation(kern_even_by_even, img_r_by_c, 2, 2, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "evenxeven kern times rxc (3x4) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 5;
    actual_result = sup_product_summation(kern_even_by_even, img_r_by_c, 2, 2, 3, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "evenxeven kern times rxc (4x3) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    /** rxc (2x3) kernel tests **/

    expected_result = 16;
    actual_result = sup_product_summation(kern_r_by_c, img_odd_by_odd, 2, 3, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (2x3) kern times oddxodd img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 16;
    actual_result = sup_product_summation(kern_r_by_c, img_even_by_even, 2, 3, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (2x3) kern times evenxeven img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 16;
    actual_result = sup_product_summation(kern_r_by_c, img_r_by_c, 2, 3, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (2x3) kern times rxc (3x4) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 3;
    actual_result = sup_product_summation(kern_r_by_c, img_r_by_c, 2, 3, 3, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (2x3) kern times rxc (4x3) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    /** rxc (3x2) kernel tests **/

    expected_result = 1;
    actual_result = sup_product_summation(kern_r_by_c, img_odd_by_odd, 3, 2, 5, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (3x2) kern times oddxodd img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 1;
    actual_result = sup_product_summation(kern_r_by_c, img_even_by_even, 3, 2, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (3x2) kern times evenxeven img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 1;
    actual_result = sup_product_summation(kern_r_by_c, img_r_by_c, 3, 2, 4, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (3x2) kern times rxc (3x4) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);

    expected_result = 4;
    actual_result = sup_product_summation(kern_r_by_c, img_r_by_c, 3, 2, 3, 0, 0);
    disp_test_results("SUP_PRODUCT_SUMMATION TEST", "rxc (3x2) kern times rxc (4x3) img", expected_result, actual_result,
                      (expected_result == actual_result), 1, output_file);



    /*** CONVOLUTION EXECUTION TESTS ***/

    /** IMG_WITH_PADDING TESTS (2D) **/
    float** img_2d = (float**) malloc(sizeof(float*));
    RowColTuple padding;

    /* NO PADDING */
    padding.rows = 0;
    padding.cols = 0;

    *img_2d = img_even_by_even;
    float expected_no_padding_nn[] = {-2, 1, -1, 2, 0, -3, 1, 4, 0, -4, 2, 1, 3, 2, -1, 1};
    float* actual_no_padding_nn = *(img_with_padding(img_2d, 4, 4, 1, padding));
    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "NXN IMAGE NO PADDING", expected_result, actual_result, 
                      (mats_equal(expected_no_padding_nn, actual_no_padding_nn, 4, 4)), 0, output_file);

    *img_2d = img_r_by_c;
    float expected_no_padding_rc[] = {-2, 1, -1, 2, 0, -3, 1, 4, 0, -4, 2, 1};
    float* actual_no_padding_rc = *(img_with_padding(img_2d, 3, 4, 1, padding));
    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "RXC IMAGE NO PADDING", expected_result, actual_result, 
                      (mats_equal(expected_no_padding_rc, actual_no_padding_rc, 3, 4)), 0, output_file);

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
    float* actual_samerc_padding_nn = *(img_with_padding(img_2d, 4, 4, 1, padding));
    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "NXN IMAGE SAME ROW COL PADDING", expected_result, actual_result, 
                      (mats_equal(expected_samerc_padding_nn, actual_samerc_padding_nn, 6, 6)), 0, output_file);

    *img_2d = img_r_by_c;
    float expected_samerc_padding_rc[] = {0, 0, 0, 0, 0, 0,
                                          0, -2, 1, -1, 2, 0,
                                          0, 0, -3, 1, 4, 0,
                                          0, 0, -4, 2, 1, 0,
                                          0, 0, 0, 0, 0, 0};
    float* actual_samerc_padding_rc = *(img_with_padding(img_2d, 3, 4, 1, padding));
    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "RXC IMAGE SAME ROW COL PADDING", expected_result, actual_result, 
                      (mats_equal(expected_samerc_padding_rc, actual_samerc_padding_rc, 5, 6)), 0, output_file);

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
    float* actual_arbitrary_padding_nn = *(img_with_padding(img_2d, 4, 4, 1, padding));
    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "NXN IMAGE ARBITRARY PADDING", expected_result, actual_result, 
                      (mats_equal(expected_arbitrary_padding_nn, actual_arbitrary_padding_nn, 8, 10)), 0, output_file);

    *img_2d = img_r_by_c;
    float expected_arbitrary_padding_rc[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, -2, 1, -1, 2, 0, 0, 0,
                                             0, 0, 0, 0, -3, 1, 4, 0, 0, 0,
                                             0, 0, 0, 0, -4, 2, 1, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float* actual_arbitrary_padding_rc = *(img_with_padding(img_2d, 3, 4, 1, padding));
    disp_test_results("IMG_WITH_PADDING TESTS (2D)", "RXC IMAGE ARBITRARY PADDING", expected_result, actual_result, 
                      (mats_equal(expected_arbitrary_padding_rc, actual_arbitrary_padding_rc, 7, 10)), 0, output_file);


    /** CONVOLUTION TESTS **/

    /**/
}