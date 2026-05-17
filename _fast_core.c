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
/* Difficult-terrain cap (TERRAIN_SPEC.md §3.2). 6.0" * 1000 = 6000 milli-inches. */
#define DIFFICULT_CAP_MILLI 6000

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
    Py_buffer occ_buf, excl_buf, enemy_buf, imp_buf, diff_buf;
    int n_enemies, cols, rows;
    int is_charge, flying, already_adjacent, strider;

    /* Argument layout (5 ints + 1 double + 5 buffers + 7 ints):
     *   start_col, start_row, gc, gr, budget,
     *   occupancy, exclusion, enemies, impassible, difficult,
     *   n_enemies, cols, rows, is_charge, flying, already_adjacent, strider
     * Format: "iiiidy*y*y*y*y*iiiiiii" */
    if (!PyArg_ParseTuple(args, "iiiidy*y*y*y*y*iiiiiii",
            &start_col, &start_row, &gc, &gr, &budget,
            &occ_buf, &excl_buf, &enemy_buf, &imp_buf, &diff_buf,
            &n_enemies, &cols, &rows,
            &is_charge, &flying, &already_adjacent, &strider))
        return NULL;

    const unsigned char *occupancy = (const unsigned char *)occ_buf.buf;
    const unsigned char *exclusion = (const unsigned char *)excl_buf.buf;
    const int *enemies = (const int *)enemy_buf.buf;
    const unsigned char *impassible = (const unsigned char *)imp_buf.buf;
    const unsigned char *difficult = (const unsigned char *)diff_buf.buf;

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
        PyBuffer_Release(&imp_buf);
        PyBuffer_Release(&diff_buf);
        return Py_BuildValue("(ii)", start_col, start_row);
    }

    int budget_milli = (int)(budget * 1000 + 0.5);
    int cap_when_entered = budget_milli < DIFFICULT_CAP_MILLI ? budget_milli : DIFFICULT_CAP_MILLI;

    /* Dijkstra cost array — two states per cell: 0 = haven't entered difficult,
     * 1 = have entered (sticky). Layout: dist_arr[cell_idx * 2 + entered]. */
    static int dist_arr[MAX_CELLS * 2];
    int total_states = total_cells * 2;
    for (int i = 0; i < total_states; i++)
        dist_arr[i] = 999999999;

    int start_idx = start_row * cols + start_col;
    /* Starting square never counts as "entered" per §3.1, even if it is itself
     * difficult terrain. */
    dist_arr[start_idx * 2] = 0;

    static HEntry heap[HEAP_CAP];
    int heap_size = 0;
    /* Pack (cell_idx, entered_flag) into HEntry.idx as (cell_idx * 2 + entered). */
    heap[heap_size++] = (HEntry){0, start_idx * 2};

    while (heap_size > 0) {
        HEntry cur = heap[0];
        heap[0] = heap[--heap_size];
        if (heap_size > 0) heap_sift_down(heap, heap_size, 0);

        if (cur.cost_milli > dist_arr[cur.idx])
            continue;

        int cidx = cur.idx >> 1;
        int entered = cur.idx & 1;
        int cc = cidx % cols;
        int cr = cidx / cols;

        for (int d = 0; d < 8; d++) {
            int nc = cc + DIRS_DC[d];
            int nr = cr + DIRS_DR[d];
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows)
                continue;

            int nidx = nr * cols + nc;

            /* Impassible blocks unless flying (§3.2/§3.4). */
            if (!flying && impassible[nidx])
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
            if (check_exclusion && exclusion[nidx])
                continue;

            /* Strider ignores difficult terrain; otherwise difficult is sticky. */
            int new_entered = entered;
            if (!strider && difficult[nidx])
                new_entered = 1;

            int new_cost = cur.cost_milli + DIRS_COST_MILLI[d];
            int cap = new_entered ? cap_when_entered : budget_milli;
            if (new_cost > cap + 10)
                continue;

            int new_state_idx = nidx * 2 + new_entered;
            if (new_cost < dist_arr[new_state_idx]) {
                dist_arr[new_state_idx] = new_cost;
                if (heap_size < HEAP_CAP) {
                    heap[heap_size] = (HEntry){new_cost, new_state_idx};
                    heap_sift_up(heap, heap_size);
                    heap_size++;
                }
            }
        }
    }

    /* Find best reachable, non-occupied, non-impassible cell closest to goal.
     * Destination forbidden in impassible terrain even for flying units (§3.3). */
    int best_col = start_col, best_row = start_row;
    int best_dist_sq = start_dist_sq;

    for (int idx = 0; idx < total_cells; idx++) {
        int s0 = dist_arr[idx * 2];
        int s1 = dist_arr[idx * 2 + 1];
        if (s0 >= 999999999 && s1 >= 999999999)
            continue;
        int cc = idx % cols;
        int cr = idx / cols;

        if (occupancy[idx])
            continue;
        if (impassible[idx])
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
    PyBuffer_Release(&imp_buf);
    PyBuffer_Release(&diff_buf);
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
    Py_buffer occ_buf, excl_buf, enemy_buf, imp_buf, diff_buf;
    int n_enemies, cols, rows;
    int is_charge, flying, already_adjacent, strider;

    if (!PyArg_ParseTuple(args, "iidy*y*y*y*y*iiiiiii",
            &start_col, &start_row, &budget,
            &occ_buf, &excl_buf, &enemy_buf, &imp_buf, &diff_buf,
            &n_enemies, &cols, &rows,
            &is_charge, &flying, &already_adjacent, &strider))
        return NULL;

    const unsigned char *occupancy = (const unsigned char *)occ_buf.buf;
    const unsigned char *exclusion = (const unsigned char *)excl_buf.buf;
    const int *enemies = (const int *)enemy_buf.buf;
    const unsigned char *impassible = (const unsigned char *)imp_buf.buf;
    const unsigned char *difficult = (const unsigned char *)diff_buf.buf;

    int check_exclusion = !is_charge && !already_adjacent;
    int no_enemies = (n_enemies == 0);
    int total_cells = cols * rows;

    int budget_milli = (int)(budget * 1000 + 0.5);
    int cap_when_entered = budget_milli < DIFFICULT_CAP_MILLI ? budget_milli : DIFFICULT_CAP_MILLI;

    /* Dijkstra cost array — two states per cell, see py_pathfind_move. */
    static int drs_dist_arr[MAX_CELLS * 2];
    int total_states = total_cells * 2;
    for (int i = 0; i < total_states; i++)
        drs_dist_arr[i] = 999999999;

    int start_idx = start_row * cols + start_col;
    drs_dist_arr[start_idx * 2] = 0;

    static HEntry drs_heap[HEAP_CAP];
    int heap_size = 0;
    drs_heap[heap_size++] = (HEntry){0, start_idx * 2};

    while (heap_size > 0) {
        HEntry cur = drs_heap[0];
        drs_heap[0] = drs_heap[--heap_size];
        if (heap_size > 0) heap_sift_down(drs_heap, heap_size, 0);

        if (cur.cost_milli > drs_dist_arr[cur.idx])
            continue;

        int cidx = cur.idx >> 1;
        int entered = cur.idx & 1;
        int cc = cidx % cols;
        int cr = cidx / cols;

        for (int d = 0; d < 8; d++) {
            int nc = cc + DIRS_DC[d];
            int nr = cr + DIRS_DR[d];
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows)
                continue;

            int nidx = nr * cols + nc;
            if (!flying && impassible[nidx])
                continue;
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
            if (check_exclusion && exclusion[nidx])
                continue;

            int new_entered = entered;
            if (!strider && difficult[nidx])
                new_entered = 1;

            int new_cost = cur.cost_milli + DIRS_COST_MILLI[d];
            int cap = new_entered ? cap_when_entered : budget_milli;
            if (new_cost > cap + 10)
                continue;

            int new_state_idx = nidx * 2 + new_entered;
            if (new_cost < drs_dist_arr[new_state_idx]) {
                drs_dist_arr[new_state_idx] = new_cost;
                if (heap_size < HEAP_CAP) {
                    drs_heap[heap_size] = (HEntry){new_cost, new_state_idx};
                    heap_sift_up(drs_heap, heap_size);
                    heap_size++;
                }
            }
        }
    }

    /* Collect all reachable, non-occupied, non-impassible cells.
     * A cell is reachable if either state (entered or not) reached it. */
    static int result_buf[MAX_CELLS * 2];
    int n_results = 0;

    for (int idx = 0; idx < total_cells; idx++) {
        int s0 = drs_dist_arr[idx * 2];
        int s1 = drs_dist_arr[idx * 2 + 1];
        if (s0 >= 999999999 && s1 >= 999999999)
            continue;

        int cc = idx % cols;
        int cr = idx / cols;

        if (occupancy[idx])
            continue;
        if (impassible[idx])
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

        result_buf[n_results * 2] = cc;
        result_buf[n_results * 2 + 1] = cr;
        n_results++;
    }

    PyBuffer_Release(&occ_buf);
    PyBuffer_Release(&excl_buf);
    PyBuffer_Release(&enemy_buf);
    PyBuffer_Release(&imp_buf);
    PyBuffer_Release(&diff_buf);

    return PyBytes_FromStringAndSize(
        (const char *)result_buf,
        (Py_ssize_t)(n_results * 2 * sizeof(int)));
}


/* -----------------------------------------------------------------------
 * c_build_exclusion_grid — precompute a flat grid marking squares within
 * 1" of any enemy model (8-neighborhood). Returns bytearray of size cols*rows.
 * ----------------------------------------------------------------------- */
static PyObject* py_build_exclusion_grid(PyObject* self, PyObject* args) {
    Py_buffer enemy_buf;
    int n_enemies, cols, rows;

    if (!PyArg_ParseTuple(args, "y*iii",
            &enemy_buf, &n_enemies, &cols, &rows))
        return NULL;

    const int *enemies = (const int *)enemy_buf.buf;
    Py_ssize_t total = (Py_ssize_t)cols * (Py_ssize_t)rows;
    PyObject *grid = PyByteArray_FromStringAndSize(NULL, total);
    if (!grid) {
        PyBuffer_Release(&enemy_buf);
        return NULL;
    }
    char *g = PyByteArray_AS_STRING(grid);
    memset(g, 0, (size_t)total);

    for (int i = 0; i < n_enemies; i++) {
        int c = enemies[i * 2];
        int r = enemies[i * 2 + 1];
        int c0 = c - 1, c1 = c + 1;
        int r0 = r - 1, r1 = r + 1;
        if (c0 < 0) c0 = 0;
        if (c1 >= cols) c1 = cols - 1;
        if (r0 < 0) r0 = 0;
        if (r1 >= rows) r1 = rows - 1;
        for (int nr = r0; nr <= r1; nr++) {
            for (int nc = c0; nc <= c1; nc++) {
                if (nc == c && nr == r) continue;
                g[nr * cols + nc] = 1;
            }
        }
    }

    PyBuffer_Release(&enemy_buf);
    return grid;
}


/* -----------------------------------------------------------------------
 * c_compute_post_move_rel — (sin θ, cos θ, normalised_dist) from a
 * post-move (x, y) to each of n_enemies enemy slots. Returns bytes
 * of (n_enemies * 3) float32 values (3 × 10 = 30 for tactical).
 * Matches Python: feats[i*3]=dy/d, feats[i*3+1]=dx/d, feats[i*3+2]=d*inv_diag.
 * ----------------------------------------------------------------------- */
static PyObject* py_compute_post_move_rel(PyObject* self, PyObject* args) {
    double post_x, post_y, inv_diag;
    Py_buffer enemy_buf;
    int n_enemies;

    if (!PyArg_ParseTuple(args, "ddy*id",
            &post_x, &post_y, &enemy_buf, &n_enemies, &inv_diag))
        return NULL;

    const double *ep = (const double *)enemy_buf.buf;
    Py_ssize_t nbytes = (Py_ssize_t)n_enemies * 3 * (Py_ssize_t)sizeof(float);
    PyObject *out = PyBytes_FromStringAndSize(NULL, nbytes);
    if (!out) {
        PyBuffer_Release(&enemy_buf);
        return NULL;
    }
    float *f = (float *)PyBytes_AS_STRING(out);
    memset(f, 0, (size_t)nbytes);

    for (int i = 0; i < n_enemies; i++) {
        double ex = ep[i * 2];
        double ey = ep[i * 2 + 1];
        double dx = ex - post_x;
        double dy = ey - post_y;
        double d = sqrt(dx * dx + dy * dy);
        int base = i * 3;
        if (d < 1e-6) {
            f[base] = 0.0f;
            f[base + 1] = 0.0f;
        } else {
            double inv_d = 1.0 / d;
            f[base] = (float)(dy * inv_d);
            f[base + 1] = (float)(dx * inv_d);
        }
        f[base + 2] = (float)(d * inv_diag);
    }

    PyBuffer_Release(&enemy_buf);
    return out;
}


/* -----------------------------------------------------------------------
 * c_encode_unit_tactical — write 200 float32 features for one unit into
 * a writable float32 buffer at a given offset. Mirror of
 * ml_features._encode_unit_tactical_into. The pure-Python wrapper
 * pre-extracts unit scalars and already-flipped centroids.
 *
 * scalars (15 doubles):
 *   [0]  wound_count (tough or 1)           → buf[0]  (normalised by max_tough)
 *   [1]  models (raw count)                 → buf[1]  (normalised by max_models)
 *   [2]  speed (0 if artillery else rush)   → buf[2]  (normalised by max_speed)
 *   [3]  survival_fraction                  → buf[3]
 *   [4]  points_fraction                    → buf[4]  (unit.points / total_side_points)
 *   [5]  flying (0/1)                       → buf[5]
 *   [6]  artillery (0/1)                    → buf[6]
 *   [7]  fearless (0/1)                     → buf[7]
 *   [8]  fear_gt_0 (0/1)                    → buf[8]
 *   [9]  is_friendly (0/1)                  → buf[9]
 *   [10] cx (already flipped for side B)
 *   [11] cy (already flipped for side B)
 *   [12] advance_distance (cells)
 *   [13] rush_distance (cells)
 *   [14] models_alive (used with models for ranged/melee scale)
 * Passed as 15 doubles via bytes buffer to avoid long argument lists.
 *
 * matchup_scale = models_alive / max(models, 1)   (computed in wrapper)
 *
 * Constants (passed explicitly for compile-time independence):
 *   cols, rows           — board dimensions
 *   max_tough, max_models, max_speed
 *   inv_board_diag
 *   obj_seize_range
 *   dead_sentinel_x, dead_sentinel_y  — positions equal to this are skipped
 *                                        in can_charge (dead slot sentinel)
 *   num_objectives (5), num_opp (10), num_same (10), num_range_thr (7)
 *   feature_dim (200), buf_len (TACTICAL_TOTAL_FEATURES)
 *
 * Layout offsets within the 200-float block are baked in (matching Python):
 *   0-9: scalars, 10-11: pos, 12-26: obj_rel, 27-56: opp_rel, 57-86: same_rel,
 *   87-156: ranged, 157-166: melee, 167-176: opp_post_adv,
 *   177-186: obj_reach, 187-196: can_charge, 197-199: tactical bools (caller)
 * ----------------------------------------------------------------------- */
static PyObject* py_encode_unit_tactical(PyObject* self, PyObject* args) {
    Py_buffer scalars_buf;       /* 14 doubles */
    Py_buffer objectives_buf;    /* 5 × 2 doubles = 10 */
    Py_buffer opp_pos_buf;       /* 10 × 2 doubles = 20 */
    Py_buffer opp_adv_buf;       /* 10 doubles */
    Py_buffer same_pos_buf;      /* 10 × 2 doubles = 20 */
    Py_buffer ranged_buf;        /* 70 float32 */
    Py_buffer melee_buf;         /* 10 float32 */
    Py_buffer buf;               /* writable float32 output */
    int offset;
    double inv_diag, max_tough, max_models, max_speed, obj_seize_range;
    double dead_x, dead_y;
    int cols, rows;

    if (!PyArg_ParseTuple(args, "y*y*y*y*y*y*y*w*idddddddii",
            &scalars_buf,
            &objectives_buf, &opp_pos_buf, &opp_adv_buf, &same_pos_buf,
            &ranged_buf, &melee_buf,
            &buf, &offset,
            &inv_diag, &max_tough, &max_models, &max_speed, &obj_seize_range,
            &dead_x, &dead_y,
            &cols, &rows))
        return NULL;

    const double *S = (const double *)scalars_buf.buf;
    /* Python-side guards "if us.models_alive <= 0: return" before we're called,
     * so the output buffer's zero fill already covers dead-slot units. */

    const double *OBJ = (const double *)objectives_buf.buf;       /* 10 doubles */
    const double *OPP = (const double *)opp_pos_buf.buf;          /* 20 doubles */
    const double *OADV = (const double *)opp_adv_buf.buf;         /* 10 doubles */
    const double *SAME = (const double *)same_pos_buf.buf;        /* 20 doubles */
    const float *RNG = (const float *)ranged_buf.buf;             /* 70 float32 */
    const float *MEL = (const float *)melee_buf.buf;              /* 10 float32 */
    float *F = (float *)buf.buf;

    double wound_count = S[0];
    double models_count = S[1];
    double speed_val = S[2];
    double survival = S[3];
    double points_frac = S[4];
    double flying = S[5];
    double artillery = S[6];
    double fearless = S[7];
    double fear_pos = S[8];
    double is_friendly = S[9];
    double cx = S[10];
    double cy = S[11];
    double advance_dist = S[12];
    double rush_dist = S[13];

    /* Feature offsets (must match Python) */
    const int TOFF_POS = 10;
    const int TOFF_OBJ_REL = 12;
    const int TOFF_OPP_REL = 27;
    const int TOFF_SAME_REL = 57;
    const int TOFF_RANGED = 87;
    const int TOFF_MELEE = 157;
    const int TOFF_OPP_POST_ADV = 167;
    const int TOFF_OBJ_REACH = 177;
    const int TOFF_CAN_CHARGE = 187;
    /* 197-199 are set by the Python caller (tactical bools) */

    const int NUM_OBJ = 5;
    const int NUM_OPP = 10;
    const int NUM_SAME = 10;
    const int NUM_RT = 7;

    float *out = F + offset;

    /* 0-9: scalar features */
    out[0] = (float)(wound_count / max_tough);
    out[1] = (float)(models_count / max_models);
    out[2] = (float)(speed_val / max_speed);
    out[3] = (float)survival;
    out[4] = (float)points_frac;
    out[5] = (float)flying;
    out[6] = (float)artillery;
    out[7] = (float)fearless;
    out[8] = (float)fear_pos;
    out[9] = (float)is_friendly;

    /* 10-11: absolute normalised position.
     * Matches Python: (cx + 0.5) / COLS, (cy + 0.5) / ROWS. */
    out[TOFF_POS] = (float)((cx + 0.5) / (double)cols);
    out[TOFF_POS + 1] = (float)((cy + 0.5) / (double)rows);

    /* 12-26: objectives (sin θ, cos θ, dist) */
    int o = TOFF_OBJ_REL;
    for (int i = 0; i < NUM_OBJ; i++) {
        double ox = OBJ[i * 2];
        double oy = OBJ[i * 2 + 1];
        double dx = ox - cx;
        double dy = oy - cy;
        double d = sqrt(dx * dx + dy * dy);
        if (d < 1e-6) {
            out[o] = 0.0f;
            out[o + 1] = 0.0f;
        } else {
            double inv_d = 1.0 / d;
            out[o] = (float)(dy * inv_d);
            out[o + 1] = (float)(dx * inv_d);
        }
        out[o + 2] = (float)(d * inv_diag);
        o += 3;
    }

    /* 27-56: opposing units (sin θ, cos θ, dist), also compute 167-176 post-adv */
    o = TOFF_OPP_REL;
    int pa = TOFF_OPP_POST_ADV;
    for (int i = 0; i < NUM_OPP; i++) {
        double ox = OPP[i * 2];
        double oy = OPP[i * 2 + 1];
        double dx = ox - cx;
        double dy = oy - cy;
        double d = sqrt(dx * dx + dy * dy);
        if (d < 1e-6) {
            out[o] = 0.0f;
            out[o + 1] = 0.0f;
        } else {
            double inv_d = 1.0 / d;
            out[o] = (float)(dy * inv_d);
            out[o + 1] = (float)(dx * inv_d);
        }
        out[o + 2] = (float)(d * inv_diag);
        o += 3;
        double post = d - OADV[i];
        if (post < 0.0) post = 0.0;
        out[pa + i] = (float)(post * inv_diag);
    }

    /* 57-86: same-side units (sin θ, cos θ, dist) */
    o = TOFF_SAME_REL;
    for (int i = 0; i < NUM_SAME; i++) {
        double sx = SAME[i * 2];
        double sy = SAME[i * 2 + 1];
        double dx = sx - cx;
        double dy = sy - cy;
        double d = sqrt(dx * dx + dy * dy);
        if (d < 1e-6) {
            out[o] = 0.0f;
            out[o + 1] = 0.0f;
        } else {
            double inv_d = 1.0 / d;
            out[o] = (float)(dy * inv_d);
            out[o + 1] = (float)(dx * inv_d);
        }
        out[o + 2] = (float)(d * inv_diag);
        o += 3;
    }

    /* 87-156: ranged matchups × scale, where scale = models_alive / max(models, 1).
     * Python computes this scale regardless of tough, so we pass models_alive
     * as S[14] and reconstruct here. */
    double models_alive_val = S[14];
    double scale = models_alive_val / (models_count > 0 ? models_count : 1.0);
    float scale_f = (float)scale;
    int rs = TOFF_RANGED;
    for (int i = 0; i < NUM_OPP * NUM_RT; i++) {
        out[rs + i] = RNG[i] * scale_f;
    }

    /* 157-166: melee × scale */
    int ms = TOFF_MELEE;
    for (int i = 0; i < NUM_OPP; i++) {
        out[ms + i] = MEL[i] * scale_f;
    }

    /* 177-186: objective reachability (can_advance, can_rush) per obj */
    int orb = TOFF_OBJ_REACH;
    double adv_thr = advance_dist + obj_seize_range;
    double rush_thr = rush_dist + obj_seize_range;
    for (int i = 0; i < NUM_OBJ; i++) {
        double ox = OBJ[i * 2];
        double oy = OBJ[i * 2 + 1];
        double dx = ox - cx;
        double dy = oy - cy;
        double d = sqrt(dx * dx + dy * dy);
        out[orb] = (d <= adv_thr) ? 1.0f : 0.0f;
        out[orb + 1] = (d <= rush_thr) ? 1.0f : 0.0f;
        orb += 2;
    }

    /* 187-196: can_charge per opposing unit
     * Python: skip if (ox, oy) == _DEAD_SENTINEL (35.5, 23.5).
     * Threshold: rush_dist + 2.0, compared via squared distance. */
    int cc = TOFF_CAN_CHARGE;
    double charge_thr = rush_dist + 2.0;
    double charge_thr_sq = charge_thr * charge_thr;
    for (int i = 0; i < NUM_OPP; i++) {
        double ox = OPP[i * 2];
        double oy = OPP[i * 2 + 1];
        if (ox == dead_x && oy == dead_y) continue;
        double dx = ox - cx;
        double dy = oy - cy;
        if (dx * dx + dy * dy < charge_thr_sq) {
            out[cc + i] = 1.0f;
        }
    }

    PyBuffer_Release(&scalars_buf);
    PyBuffer_Release(&objectives_buf);
    PyBuffer_Release(&opp_pos_buf);
    PyBuffer_Release(&opp_adv_buf);
    PyBuffer_Release(&same_pos_buf);
    PyBuffer_Release(&ranged_buf);
    PyBuffer_Release(&melee_buf);
    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
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
    {"c_build_exclusion_grid", py_build_exclusion_grid, METH_VARARGS,
     "Build a flat bytearray marking cells within 1\" of any enemy model."},
    {"c_compute_post_move_rel", py_compute_post_move_rel, METH_VARARGS,
     "(sin θ, cos θ, normalised_dist) from a post-move position to each enemy slot."},
    {"c_encode_unit_tactical", py_encode_unit_tactical, METH_VARARGS,
     "Write one unit's 200 tactical features into a float32 output buffer."},
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
