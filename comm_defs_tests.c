#include <stdio.h>
#include <assert.h>
#include "common_definitions.h"

int main() {
    float matrix[] = {9, 4, 8, 5, 7, 1, 12, 11, 14, 18, 6, 31, 22, 44, 59, 16, 21, 25, 27, 28};
    int numrows = 4;
    int numcols = 5;
    

    /* MATVAL TESTS */

    assert(mat_val(matrix, numcols, 2, 2) == 22); // Testing access in matrix interior
    
    // Testing corner accesses
    assert(mat_val(matrix, numcols, 0, 0) == 9);
    assert(mat_val(matrix, numcols, 0, 4) == 7);
    assert(mat_val(matrix, numcols, 3, 0) == 16);
    assert(mat_val(matrix, numcols, 3, 4) == 28);

    // Testing top, left, right, bottom row accesses
    assert(mat_val(matrix, numcols, 0, 2) == 8);
    assert(mat_val(matrix, numcols, 1, 4) == 18);
    assert(mat_val(matrix, numcols, 2, 0) == 6);
    assert(mat_val(matrix, numcols, 3, 1) == 21);


    /* SETMATVAL TESTS */

    // Testing access in matrix interior
    set_mat_val(matrix, numcols, 2, 2, 99);
    assert(mat_val(matrix, numcols, 2, 2) == 99);

    // Testing corner accesses
    set_mat_val(matrix, numcols, 0, 0, 123);
    set_mat_val(matrix, numcols, 0, 4, 321);
    set_mat_val(matrix, numcols, 3, 0, 456);
    set_mat_val(matrix, numcols, 3, 4, 654);

    assert(mat_val(matrix, numcols, 0, 0) == 123);
    assert(mat_val(matrix, numcols, 0, 4) == 321);
    assert(mat_val(matrix, numcols, 3, 0) == 456);
    assert(mat_val(matrix, numcols, 3, 4) == 654);

    // Testing corner accesses
    set_mat_val(matrix, numcols, 0, 2, 789);
    set_mat_val(matrix, numcols, 1, 4, 987);
    set_mat_val(matrix, numcols, 2, 0, 159);
    set_mat_val(matrix, numcols, 3, 1, 951);

    assert(mat_val(matrix, numcols, 0, 2) == 789);
    assert(mat_val(matrix, numcols, 1, 4) == 987);
    assert(mat_val(matrix, numcols, 2, 0) == 159);
    assert(mat_val(matrix, numcols, 3, 1) == 951);
}
