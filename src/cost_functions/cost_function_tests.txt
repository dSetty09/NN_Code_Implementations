/** CONTAINS TESTS FOR COST FUNCTIONS AND THEIR DERIVATIVES */

#include <stdio.h>
#include <stdlib.h>

#include "common_definitions.h"
#include "cost_functions.h"

int main() {
    FILE* output_file = fopen("cost_function_tests.txt", "w");

    float expected;
    float actual;

    /** CROSS ENTROPY LOSS TESTS **/

    /* STANDARD DISTRIBUTIONS */
    float p_sd[] = {0.029489, 0.217895, 0.080159, 0.592299, 0.080159};
    float q_sd[] = {0.623472, 0.00959, 0.234224, 0.123124, 0.00959};

    expected = 2.75595603036;
    actual = cross_entropy(q_sd, p_sd, 5, NO_DERIV);

    disp_test_results("CROSS ENTROPY LOSS TESTS", "STANDARD DISTRIBUTIONS", (void*) &expected, (void*) &actual, TRUE, output_file);

    /* DISTRIBUTION WHERE ONE YIELDED HAS PROBABILITY 0 */

    float p_0[] = {0.06, 0.16, 0.78};
    float q_0[] = {0, 0.25, 0.75};

    expected = 2.51852569799;
    actual = cross_entropy(q_0, p_0, 3, NO_DERIV);

    disp_test_results("CROSS ENTROPY LOSS TESTS", "DISTRIBUTION WHERE ONE YIELDED HAS PROBABILITY 0", (void*) &expected, (void*) &actual, TRUE, output_file);

    /* DISTRIBUTION WHERE CLASSIFIES COMPLETELY WRONG */

    float p_d0[] = {0.83, 1E-14, 0.17};
    float q_d0[] = {0.00, 0.00, 1.00};

    expected = 28.6671844078;
    actual = cross_entropy(q_d0, p_d0, 3, NO_DERIV);

    disp_test_results("CROSS ENTROPY LOSS TESTS", "DISTRIBUTION WHERE CLASSIFIES COMPLETELY WRONG", (void*) &expected, (void*) &actual, TRUE, output_file);

    /* DERIVIATIVE WITH RESPECT TO 0 AT INDEX WITH LARGE TRUE VALUE */
    expected = -829999946924032.000000;
    actual = cross_entropy(q_d0, p_d0, 3, 0);

    disp_test_results("CROSS ENTROPY LOSS TESTS", "DERIVIATIVE WITH RESPECT TO 0 AT INDEX WITH LARGE TRUE VALUE", (void*) &expected, (void*) &actual, TRUE, output_file);

    /* DERIVATIVE WITH RESPECT TO 0 AT INDEX WTIH SMALL TRUE VALUE */
    expected = -10;
    actual = cross_entropy(q_d0, p_d0, 3, 1);

    disp_test_results("CROSS ENTROPY LOSS TESTS", "DERIVATIVE WITH RESPECT TO 0 AT INDEX WTIH SMALL TRUE VALUE", (void*) &expected, (void*) &actual, TRUE, output_file);
 
    fclose(output_file);
}