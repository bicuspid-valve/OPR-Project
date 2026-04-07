/*
 * _fast_core.c — C extension for hot inner loops in the OPR game simulator.
 *
 * Provides four functions:
 *   c_greedy_move(col, row, gc, gr, budget, occupancy, exclusion,
 *                 enemy_positions, n_enemies, cols, rows,
 *                 is_charge, flying, already_adjacent)
 *       -> (new_col, new_row)
 *
 *   c_find_kite_point(cx, cy, tcx, tcy, enemy_centres, n_enemies,
 *                     budget, weapon_range, has_wr)
 *       -> (best_col, best_row)
 *
 *   c_min_dists_sq(a_positions, n_a, t_positions, n_t)
 *       -> list[int]
 *
 *   c_encode_distances(cx, cy, targets, n_targets, inv_diag)
 *       -> list[float]
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

/* 8 directions: dc, dr, cost*1000 (use int to avoid FP in comparisons) */
static const int DIRS_DC[8] = { 1, -1,  0,  0,  1,  1, -1, -1 };
static const int DIRS_DR[8] = { 0,  0,  1, -1,  1, -1,  1, -1 };
/* cost * 1000 for integer comparison against budget*1000 */
static const int DIRS_COST_MILLI[8] = { 1000, 1000, 1000, 1000, 1414, 1414, 1414, 1414 };
static const double DIRS_COST[8] = { 1.0, 1.0, 1.0, 1.0, 1.41421356, 1.41421356, 1.41421356, 1.41421356 };


/* -----------------------------------------------------------------------
 * c_greedy_move
 * ----------------------------------------------------------------------- */
static PyObject* py_greedy_move(PyObject* self, PyObject* args) {
    int col, row, gc, gr;
    double budget;
    Py_buffer occ_buf, excl_buf, enemy_buf;
    int n_enemies, cols, rows;
    int is_charge, flying, already_adjacent;

    if (!PyArg_ParseTuple(args, "iiiidy*y*y*iiiiii",
            &col, &row, &gc, &gr, &budget,
            &occ_buf, &excl_buf, &enemy_buf,
            &n_enemies, &cols, &rows,
            &is_charge, &flying, &already_adjacent))
        return NULL;

    const unsigned char *occupancy = (const unsigned char *)occ_buf.buf;
    const unsigned char *exclusion = (const unsigned char *)excl_buf.buf;
    /* enemy_positions: flat array of (col, row) pairs as int32 */
    const int *enemies = (const int *)enemy_buf.buf;

    int check_exclusion = !is_charge && !already_adjacent;
    int no_enemies = (n_enemies == 0);

    int dc0 = col - gc;
    int dr0 = row - gr;
    int best_dist = dc0 * dc0 + dr0 * dr0;

    if (best_dist == 0) {
        PyBuffer_Release(&occ_buf);
        PyBuffer_Release(&excl_buf);
        PyBuffer_Release(&enemy_buf);
        return Py_BuildValue("(ii)", col, row);
    }

    double remaining = budget;

    while (remaining >= 1.0) {
        int best_nc = -1, best_nr = -1;
        int best_next_dist = best_dist;
        double best_cost = 0.0;

        for (int d = 0; d < 8; d++) {
            double cost = DIRS_COST[d];
            if (cost > remaining + 0.01)
                continue;
            int nc = col + DIRS_DC[d];
            int nr = row + DIRS_DR[d];
            /* bounds check */
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows)
                continue;
            /* enemy position check (unless flying) */
            if (!flying && !no_enemies) {
                int blocked = 0;
                for (int e = 0; e < n_enemies; e++) {
                    if (enemies[e * 2] == nc && enemies[e * 2 + 1] == nr) {
                        blocked = 1;
                        break;
                    }
                }
                if (blocked) continue;
            }
            /* exclusion zone check */
            if (check_exclusion && exclusion[nr * cols + nc])
                continue;
            /* distance to goal */
            int dgc = nc - gc;
            int dgr = nr - gr;
            int dd = dgc * dgc + dgr * dgr;
            if (dd < best_next_dist) {
                best_nc = nc;
                best_nr = nr;
                best_next_dist = dd;
                best_cost = cost;
            }
        }

        if (best_nc < 0)
            break;

        col = best_nc;
        row = best_nr;
        remaining -= best_cost;
        best_dist = best_next_dist;

        if (best_dist == 0)
            break;
    }

    /* Final position must not be occupied */
    int start_col = col, start_row = row;
    /* We need the original start to return if stuck — but we don't have it.
       The Python wrapper handles the "return start" fallback for the outer
       occupied-square check. We just check occupancy here. */
    if (occupancy[row * cols + col]) {
        /* Try adjacent free square closest to goal */
        int best_alt_c = -1, best_alt_r = -1;
        int best_alt_dist = 999999;
        for (int d = 0; d < 8; d++) {
            int nc = col + DIRS_DC[d];
            int nr = row + DIRS_DR[d];
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows)
                continue;
            if (occupancy[nr * cols + nc])
                continue;
            /* Check not an enemy position */
            int is_enemy = 0;
            for (int e = 0; e < n_enemies; e++) {
                if (enemies[e * 2] == nc && enemies[e * 2 + 1] == nr) {
                    is_enemy = 1;
                    break;
                }
            }
            if (is_enemy) continue;
            int dgc = nc - gc;
            int dgr = nr - gr;
            int dd = dgc * dgc + dgr * dgr;
            if (dd < best_alt_dist) {
                best_alt_c = nc;
                best_alt_r = nr;
                best_alt_dist = dd;
            }
        }
        if (best_alt_c >= 0) {
            col = best_alt_c;
            row = best_alt_r;
        }
        /* else: caller handles fallback to original start */
    }

    PyBuffer_Release(&occ_buf);
    PyBuffer_Release(&excl_buf);
    PyBuffer_Release(&enemy_buf);
    return Py_BuildValue("(ii)", col, row);
}


/* -----------------------------------------------------------------------
 * c_find_kite_point
 * ----------------------------------------------------------------------- */
static PyObject* py_find_kite_point(PyObject* self, PyObject* args) {
    double cx, cy, tcx, tcy;
    Py_buffer enemy_buf;
    int n_enemies;
    double move_budget, weapon_range;
    int has_wr;  /* boolean: is weapon_range constraint active? */

    if (!PyArg_ParseTuple(args, "ddddy*iddi",
            &cx, &cy, &tcx, &tcy,
            &enemy_buf, &n_enemies,
            &move_budget, &weapon_range, &has_wr))
        return NULL;

    const double *enemy_centres = (const double *)enemy_buf.buf;

    int icx = (int)round(cx);
    int icy = (int)round(cy);
    int budget_int = (int)ceil(move_budget);
    double budget_sq = move_budget * move_budget;
    double wr_sq = weapon_range * weapon_range;

    int best_pc = icx, best_pr = icy;
    double best_min_enemy_dist_sq = -1.0;

    for (int dc = -budget_int; dc <= budget_int; dc++) {
        for (int dr = -budget_int; dr <= budget_int; dr++) {
            int pc = icx + dc;
            int pr = icy + dr;
            /* Within move budget? */
            if (dc * dc + dr * dr > budget_sq)
                continue;
            /* Weapon range constraint */
            if (has_wr) {
                double tdx = pc - tcx;
                double tdy = pr - tcy;
                if (tdx * tdx + tdy * tdy > wr_sq)
                    continue;
            }
            /* Minimum distance to any enemy */
            double min_enemy_dsq = 1e18;
            for (int e = 0; e < n_enemies; e++) {
                double edx = pc - enemy_centres[e * 2];
                double edy = pr - enemy_centres[e * 2 + 1];
                double dsq = edx * edx + edy * edy;
                if (dsq < min_enemy_dsq)
                    min_enemy_dsq = dsq;
            }
            if (min_enemy_dsq > best_min_enemy_dist_sq) {
                best_min_enemy_dist_sq = min_enemy_dsq;
                best_pc = pc;
                best_pr = pr;
            }
        }
    }

    PyBuffer_Release(&enemy_buf);
    return Py_BuildValue("(ii)", best_pc, best_pr);
}


/* -----------------------------------------------------------------------
 * c_min_dists_sq
 * ----------------------------------------------------------------------- */
static PyObject* py_min_dists_sq(PyObject* self, PyObject* args) {
    Py_buffer a_buf, t_buf;
    int n_a, n_t;

    if (!PyArg_ParseTuple(args, "y*iy*i", &a_buf, &n_a, &t_buf, &n_t))
        return NULL;

    const int *a_pos = (const int *)a_buf.buf;  /* flat: [c0,r0, c1,r1, ...] */
    const int *t_pos = (const int *)t_buf.buf;

    PyObject *result = PyList_New(n_a);
    if (!result) {
        PyBuffer_Release(&a_buf);
        PyBuffer_Release(&t_buf);
        return NULL;
    }

    for (int i = 0; i < n_a; i++) {
        int ac = a_pos[i * 2];
        int ar = a_pos[i * 2 + 1];
        int best = 999999;
        for (int j = 0; j < n_t; j++) {
            int dc = ac - t_pos[j * 2];
            int dr = ar - t_pos[j * 2 + 1];
            int d = dc * dc + dr * dr;
            if (d < best)
                best = d;
        }
        PyList_SET_ITEM(result, i, PyLong_FromLong(best));
    }

    PyBuffer_Release(&a_buf);
    PyBuffer_Release(&t_buf);
    return result;
}


/* -----------------------------------------------------------------------
 * c_encode_distances
 * ----------------------------------------------------------------------- */
static PyObject* py_encode_distances(PyObject* self, PyObject* args) {
    double cx, cy, inv_diag;
    Py_buffer targets_buf;
    int n_targets;

    if (!PyArg_ParseTuple(args, "ddy*id", &cx, &cy, &targets_buf, &n_targets, &inv_diag))
        return NULL;

    const double *targets = (const double *)targets_buf.buf; /* flat: [x0,y0, x1,y1, ...] */

    PyObject *result = PyList_New(n_targets);
    if (!result) {
        PyBuffer_Release(&targets_buf);
        return NULL;
    }

    for (int i = 0; i < n_targets; i++) {
        double dx = cx - targets[i * 2];
        double dy = cy - targets[i * 2 + 1];
        double d = sqrt(dx * dx + dy * dy) * inv_diag;
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(d));
    }

    PyBuffer_Release(&targets_buf);
    return result;
}


/* -----------------------------------------------------------------------
 * c_pathfind_move — Dijkstra-based pathfinding (replaces greedy)
 *
 * Explores all reachable cells within the movement budget via Dijkstra,
 * then returns the reachable cell closest to the goal that is not
 * occupied or enemy-held.  Unlike the greedy approach, this can route
 * around friendly models and other obstacles that require a temporary
 * detour away from the goal.
 *
 * Same signature as c_greedy_move for drop-in replacement.
 * ----------------------------------------------------------------------- */

#define MAX_CELLS (72 * 48)   /* 3 456 — matches COLS * ROWS */
#define HEAP_CAP  (MAX_CELLS * 4)

typedef struct { int cost_milli; int idx; } HEntry;

/* Sift-up (used after push) */
static void heap_sift_up(HEntry *h, int i) {
    HEntry val = h[i];
    while (i > 0) {
        int parent = (i - 1) >> 1;
        if (h[parent].cost_milli <= val.cost_milli) break;
        h[i] = h[parent];
        i = parent;
    }
    h[i] = val;
}

/* Sift-down (used after pop) */
static void heap_sift_down(HEntry *h, int size, int i) {
    HEntry val = h[i];
    while (1) {
        int left = 2 * i + 1;
        if (left >= size) break;
        int right = left + 1;
        int smallest = left;
        if (right < size && h[right].cost_milli < h[left].cost_milli)
            smallest = right;
        if (val.cost_milli <= h[smallest].cost_milli) break;
        h[i] = h[smallest];
        i = smallest;
    }
    h[i] = val;
}

static PyObject* py_pathfind_move(PyObject* self, PyObject* args) {
    int start_col, start_row, gc, gr;
    double budget;
    Py_buffer occ_buf, excl_buf, enemy_buf;
    int n_enemies, cols, rows;
    int is_charge, flying, already_adjacent;

    if (!PyArg_ParseTuple(args, "iiiidy*y*y*iiiiii",
            &start_col, &start_row, &gc, &gr, &budget,
            &occ_buf, &excl_buf, &enemy_buf,
            &n_enemies, &cols, &rows,
            &is_charge, &flying, &already_adjacent))
        return NULL;

    const unsigned char *occupancy = (const unsigned char *)occ_buf.buf;
    const unsigned char *exclusion = (const unsigned char *)excl_buf.buf;
    const int *enemies = (const int *)enemy_buf.buf;

    int check_exclusion = !is_charge && !already_adjacent;
    int no_enemies = (n_enemies == 0);
    int total_cells = cols * rows;

    int start_dc = start_col - gc;
    int start_dr = start_row - gr;
    int start_dist_sq = start_dc * start_dc + start_dr * start_dr;

    if (start_dist_sq == 0) {
        PyBuffer_Release(&occ_buf);
        PyBuffer_Release(&excl_buf);
        PyBuffer_Release(&enemy_buf);
        return Py_BuildValue("(ii)", start_col, start_row);
    }

    int budget_milli = (int)(budget * 1000 + 0.5);

    /* Dijkstra cost array — static to avoid large stack allocation */
    static int dist_arr[MAX_CELLS];
    for (int i = 0; i < total_cells; i++)
        dist_arr[i] = 999999999;

    int start_idx = start_row * cols + start_col;
    dist_arr[start_idx] = 0;

    /* Binary min-heap */
    static HEntry heap[HEAP_CAP];
    int heap_size = 0;

    /* Push start */
    heap[heap_size++] = (HEntry){0, start_idx};

    while (heap_size > 0) {
        /* Pop min */
        HEntry cur = heap[0];
        heap[0] = heap[--heap_size];
        if (heap_size > 0) heap_sift_down(heap, heap_size, 0);

        if (cur.cost_milli > dist_arr[cur.idx])
            continue;

        int cc = cur.idx % cols;
        int cr = cur.idx / cols;

        for (int d = 0; d < 8; d++) {
            int nc = cc + DIRS_DC[d];
            int nr = cr + DIRS_DR[d];
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows)
                continue;

            int new_cost = cur.cost_milli + DIRS_COST_MILLI[d];
            if (new_cost > budget_milli + 10)
                continue;

            /* Enemy position check (unless flying) */
            if (!flying && !no_enemies) {
                int blocked = 0;
                for (int e = 0; e < n_enemies; e++) {
                    if (enemies[e * 2] == nc && enemies[e * 2 + 1] == nr) {
                        blocked = 1;
                        break;
                    }
                }
                if (blocked) continue;
            }

            /* Exclusion zone check */
            if (check_exclusion && exclusion[nr * cols + nc])
                continue;

            int nidx = nr * cols + nc;
            if (new_cost < dist_arr[nidx]) {
                dist_arr[nidx] = new_cost;
                if (heap_size < HEAP_CAP) {
                    heap[heap_size] = (HEntry){new_cost, nidx};
                    heap_sift_up(heap, heap_size);
                    heap_size++;
                }
            }
        }
    }

    /* Find best reachable, non-occupied cell closest to goal */
    int best_col = start_col, best_row = start_row;
    int best_dist_sq = start_dist_sq;

    for (int idx = 0; idx < total_cells; idx++) {
        if (dist_arr[idx] >= 999999999)
            continue;
        int cc = idx % cols;
        int cr = idx / cols;

        if (occupancy[idx])
            continue;

        if (!no_enemies) {
            int is_enemy = 0;
            for (int e = 0; e < n_enemies; e++) {
                if (enemies[e * 2] == cc && enemies[e * 2 + 1] == cr) {
                    is_enemy = 1;
                    break;
                }
            }
            if (is_enemy) continue;
        }

        int dgc = cc - gc;
        int dgr = cr - gr;
        int d = dgc * dgc + dgr * dgr;
        if (d < best_dist_sq) {
            best_dist_sq = d;
            best_col = cc;
            best_row = cr;
        }
    }

    PyBuffer_Release(&occ_buf);
    PyBuffer_Release(&excl_buf);
    PyBuffer_Release(&enemy_buf);
    return Py_BuildValue("(ii)", best_col, best_row);
}


/* -----------------------------------------------------------------------
 * c_dijkstra_reachable_set — Return all reachable cells within budget
 *
 * Reuses the Dijkstra loop from py_pathfind_move.  After exploration,
 * iterates over all cells with dist < INF and applies occupancy/exclusion
 * filters.  Returns a flat int32 buffer [c0,r0, c1,r1, ...].
 * ----------------------------------------------------------------------- */

static PyObject* py_dijkstra_reachable_set(PyObject* self, PyObject* args) {
    int start_col, start_row;
    double budget;
    Py_buffer occ_buf, excl_buf, enemy_buf;
    int n_enemies, cols, rows;
    int is_charge, flying, already_adjacent;

    if (!PyArg_ParseTuple(args, "iidy*y*y*iiiiii",
            &start_col, &start_row, &budget,
            &occ_buf, &excl_buf, &enemy_buf,
            &n_enemies, &cols, &rows,
            &is_charge, &flying, &already_adjacent))
        return NULL;

    const unsigned char *occupancy = (const unsigned char *)occ_buf.buf;
    const unsigned char *exclusion = (const unsigned char *)excl_buf.buf;
    const int *enemies = (const int *)enemy_buf.buf;

    int check_exclusion = !is_charge && !already_adjacent;
    int no_enemies = (n_enemies == 0);
    int total_cells = cols * rows;

    int budget_milli = (int)(budget * 1000 + 0.5);

    /* Dijkstra cost array */
    static int drs_dist_arr[MAX_CELLS];
    for (int i = 0; i < total_cells; i++)
        drs_dist_arr[i] = 999999999;

    int start_idx = start_row * cols + start_col;
    drs_dist_arr[start_idx] = 0;

    /* Binary min-heap */
    static HEntry drs_heap[HEAP_CAP];
    int heap_size = 0;

    drs_heap[heap_size++] = (HEntry){0, start_idx};

    while (heap_size > 0) {
        HEntry cur = drs_heap[0];
        drs_heap[0] = drs_heap[--heap_size];
        if (heap_size > 0) heap_sift_down(drs_heap, heap_size, 0);

        if (cur.cost_milli > drs_dist_arr[cur.idx])
            continue;

        int cc = cur.idx % cols;
        int cr = cur.idx / cols;

        for (int d = 0; d < 8; d++) {
            int nc = cc + DIRS_DC[d];
            int nr = cr + DIRS_DR[d];
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows)
                continue;

            int new_cost = cur.cost_milli + DIRS_COST_MILLI[d];
            if (new_cost > budget_milli + 10)
                continue;

            /* Enemy position check (unless flying) */
            if (!flying && !no_enemies) {
                int blocked = 0;
                for (int e = 0; e < n_enemies; e++) {
                    if (enemies[e * 2] == nc && enemies[e * 2 + 1] == nr) {
                        blocked = 1;
                        break;
                    }
                }
                if (blocked) continue;
            }

            /* Exclusion zone check */
            if (check_exclusion && exclusion[nr * cols + nc])
                continue;

            int nidx = nr * cols + nc;
            if (new_cost < drs_dist_arr[nidx]) {
                drs_dist_arr[nidx] = new_cost;
                if (heap_size < HEAP_CAP) {
                    drs_heap[heap_size] = (HEntry){new_cost, nidx};
                    heap_sift_up(drs_heap, heap_size);
                    heap_size++;
                }
            }
        }
    }

    /* Collect all reachable, non-occupied cells into result buffer.
     * Pre-allocate for worst case (all cells reachable). */
    static int result_buf[MAX_CELLS * 2];
    int n_results = 0;

    for (int idx = 0; idx < total_cells; idx++) {
        if (drs_dist_arr[idx] >= 999999999)
            continue;

        int cc = idx % cols;
        int cr = idx / cols;

        /* Skip occupied cells (but the start cell is always allowed — handled by caller) */
        if (occupancy[idx])
            continue;

        /* Skip enemy positions */
        if (!no_enemies) {
            int is_enemy = 0;
            for (int e = 0; e < n_enemies; e++) {
                if (enemies[e * 2] == cc && enemies[e * 2 + 1] == cr) {
                    is_enemy = 1;
                    break;
                }
            }
            if (is_enemy) continue;
        }

        result_buf[n_results * 2] = cc;
        result_buf[n_results * 2 + 1] = cr;
        n_results++;
    }

    PyBuffer_Release(&occ_buf);
    PyBuffer_Release(&excl_buf);
    PyBuffer_Release(&enemy_buf);

    /* Return as bytes object (caller reshapes via numpy) */
    return PyBytes_FromStringAndSize(
        (const char *)result_buf,
        (Py_ssize_t)(n_results * 2 * sizeof(int)));
}


/* -----------------------------------------------------------------------
 * Module definition
 * ----------------------------------------------------------------------- */
static PyMethodDef FastCoreMethods[] = {
    {"c_greedy_move", py_greedy_move, METH_VARARGS,
     "Fast greedy pathfinding for one model toward a goal (legacy)."},
    {"c_pathfind_move", py_pathfind_move, METH_VARARGS,
     "Dijkstra-based pathfinding for one model toward a goal."},
    {"c_find_kite_point", py_find_kite_point, METH_VARARGS,
     "Find best kite position maximising distance to nearest enemy."},
    {"c_min_dists_sq", py_min_dists_sq, METH_VARARGS,
     "Min squared distance from each attacker to nearest target."},
    {"c_encode_distances", py_encode_distances, METH_VARARGS,
     "Compute normalised Euclidean distances from a point to N targets."},
    {"c_dijkstra_reachable_set", py_dijkstra_reachable_set, METH_VARARGS,
     "Return all reachable cells within movement budget via Dijkstra."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastcoremodule = {
    PyModuleDef_HEAD_INIT,
    "_fast_core",
    "C-accelerated hot loops for OPR game simulation.",
    -1,
    FastCoreMethods
};

PyMODINIT_FUNC PyInit__fast_core(void) {
    return PyModule_Create(&fastcoremodule);
}
