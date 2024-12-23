#include "../../include/neural_net_ops/neural_net_ops.h"

float round_to_place(float val, float place) {
    float multiple = powf(10, place);
    return round(val * multiple) / multiple;
}

int are_equal(float val1, float val2) {
    if (fabsf(val1 - val2) < FLT_EPSILON) {
        return 1;
    }

    return 0;
}

float arr_max(float* arr, int num) {
    float max = arr[0];

    for (int i = 0; i < num; ++i) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }

    return max;
}

float flt_safe_exp(float x) {
    if (x < MIN_FLT_EXP) return FLT_EPSILON;
    if (x > MAX_FLT_EXP) return FLT_MAX;
    return expf(x);
}

float flt_safe_log(float x) {
    if (x < MIN_FLT_LOG_ARG) return logf(MIN_FLT_LOG_ARG);
    if (x > MAX_FLT_LOG_ARG) return logf(MAX_FLT_LOG_ARG);
    return logf(x);
}

float flt_safe_square(float x) {
    if (x < MIN_FLT_SQR_ARG) return powf(MIN_FLT_SQR_ARG, 2);
    if (x > MAX_FLT_SQR_ARG) return powf(MAX_FLT_SQR_ARG, 2);
    return powf(x, 2);
}

float random_num(float min, float max, float precision) {
    srand(time(NULL)); // set random number generator seed based on current time

    int multiple = powf(10, precision);

    int temp_min = (int) ceilf((min * multiple));
    int temp_max = (int) floorf(max * multiple) + 1;

    int shift_val = -temp_min;

    if (min < 0) {
        temp_min += shift_val;
        temp_max += shift_val;
    }

    int temp_ret = (rand() + temp_min) % temp_max;
    
    if (min < 0) {
        temp_ret -= shift_val;
    }

    float ret = ((float) temp_ret) / ((float) multiple);
    return ret;
}

float mat_val(float* mat, int ncols, int i, int j) {
    return mat[ncols * i + j];
}

void set_mat_val(float* mat, int ncols, int i, int j, float new_val) {
    mat[ncols * i + j] = new_val;
}

int mats_equal(float* mat1, float* mat2, int nrows, int ncols) {
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            float first_val = round_to_place(mat_val(mat1, ncols, i, j), 5);
            float second_val = round_to_place(mat_val(mat2, ncols, i, j), 5);

            if (first_val != second_val) {
                return 0;
            }
        }
    }

    return 1;
}

float* flatten_img(void* img, int num_rows, int num_cols, int num_channels) {
    if (num_channels == 1) {
        return (float*) img;
    }

    float* flattened_set = (float*) malloc(sizeof(float) * num_channels * num_rows * num_cols);
    int entries_per_img_channel = num_rows * num_cols;

    for (int ch = 0; ch < num_channels; ++ch) {
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                flattened_set[entries_per_img_channel * ch + num_cols * r + c] = 
                    mat_val(((float**) img)[ch], num_cols, r, c);
            }
        }
    }

    return flattened_set;
}

float min_vect_elem(void* vect, int num_rows, int num_cols, int num_layers) {
    float** vect3d = NULL;
    float* vect2d = NULL;

    if (num_layers > 1) {
        vect3d = (float**) vect;
    } else {
        vect2d = (float*) vect;
    }

    float min = (vect3d) ? mat_val(vect3d[0], num_rows, 0, 0) : mat_val(vect2d, num_cols, 0, 0);

    if (vect2d) {
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                float curr_val = mat_val(vect2d, num_cols, r, c);
                if (curr_val < min) min = curr_val;
            }
        }

        return min;
    }

    for (int l = 0; l < num_layers; ++l) {
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                float curr_val = mat_val(vect3d[l], num_cols, r, c);
                if (curr_val < min) min = curr_val;
            }
        }
    }

    return min;
}

void print_vector(void* vect, int num_rows, int num_cols, int num_layers,
                  const char* vect_name, int vect_name_len, FILE* output_file) {

    int vect_offset = vect_name_len + 3;
    int vect_centr_index = num_rows / 2;

    int elem_offset; elem_offset = (min_vect_elem(vect, num_rows, num_cols, num_layers) < 0) ? 1 : 0;

    for (int r = 0; r < num_rows; ++r) {
        if (r == vect_centr_index) fprintf(output_file, "%s = ", vect_name);
        else for (int i = 0; i < vect_offset; ++i) fprintf(output_file, " ");

        for (int l = 0; l < num_layers; ++l) {
            fprintf(output_file, "[");

            for (int c = 0; c < num_cols; ++c) {
                float entry_to_print; 

                if (num_layers > 1) {
                    entry_to_print = mat_val(((float**) vect)[l], num_cols, r, c);
                } else {
                    entry_to_print = mat_val((float*) vect, num_cols, r, c);
                }

                for (int i = 0; i < elem_offset; ++i) fprintf(output_file, " ");
                fprintf(output_file, "%f", entry_to_print);
                
                if (c < num_cols - 1) fprintf(output_file, ", ");
            }

            fprintf(output_file, "]");
            
            if (l == num_layers - 1) {
                fprintf(output_file, "\n");
            } else {
                if (r == vect_centr_index) fprintf(output_file, " , ");
                else fprintf(output_file, "   ");
            }
        }
    }
}

void disp_test_results(const char* testing_function, const char* conducting_test, void* expected, void* actual, int testing_value, FILE* output_file) {
    fprintf(output_file, "-----------------------------------------------------\n");
    fprintf(output_file, "%s | %s\n", testing_function, conducting_test);

    if (testing_value) {
        fprintf(output_file, "Expected Result: %f\n", *((float*) expected));
        fprintf(output_file, "Actual Result: %f\n", *((float*) actual));
        assert(round_to_place(*((float*) expected), 5) == round_to_place(*((float*) actual), 5));
    } else {
        ReadVectFmt* expected_rvf = (ReadVectFmt*) expected;
        print_vector(expected_rvf->vect, expected_rvf->num_rows, expected_rvf->num_cols, expected_rvf->num_layers,
                     "Expected", 8, output_file);

        fprintf(output_file, "\n");

        ReadVectFmt* actual_rvf = (ReadVectFmt*) actual;
        print_vector(actual_rvf->vect, actual_rvf->num_rows, actual_rvf->num_cols, actual_rvf->num_layers,
                     "Actual", 6, output_file);

        assert(mats_equal((float*) expected_rvf->vect, (float*) actual_rvf->vect, 
                          expected_rvf->num_rows, expected_rvf->num_cols));
    }

    fprintf(output_file, "Test passed.\n");
    fprintf(output_file, "-----------------------------------------------------\n");
}
