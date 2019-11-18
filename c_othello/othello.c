#include "othello.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void othello_init(othello_t *o)
{
        o->disks[PLAYER_BLACK] = 0;
        o->disks[PLAYER_WHITE] = 0;

        othello_set_cell_state(o, 3, 4, CELL_BLACK);
        othello_set_cell_state(o, 4, 3, CELL_BLACK);
        othello_set_cell_state(o, 3, 3, CELL_WHITE);
        othello_set_cell_state(o, 4, 4, CELL_WHITE);
}

cell_state_t othello_cell_state(const othello_t *o, int row, int col)
{
        uint64_t mask = 1ULL << (row * 8 + col);

        assert(row >= 0 && row <= 7);
        assert(col >= 0 && col <= 7);

        if (o->disks[PLAYER_BLACK] & mask) {
                return CELL_BLACK;
        }
        if (o->disks[PLAYER_WHITE] & mask) {
                return CELL_WHITE;
        }
        return CELL_EMPTY;
}

void othello_set_cell_state(othello_t *o, int row, int col, cell_state_t s)
{
        uint64_t mask = 1ULL << (row * 8 + col);

        assert(row >= 0 && row <= 7);
        assert(col >= 0 && col <= 7);

        o->disks[PLAYER_BLACK] &= ~mask;
        o->disks[PLAYER_WHITE] &= ~mask;

        if (s == CELL_BLACK) {
                o->disks[PLAYER_BLACK] |= mask;
        } else if (s == CELL_WHITE) {
                o->disks[PLAYER_WHITE] |= mask;
        }
}

static int popcount(uint64_t x)
{
#ifdef __GNUC__
        return __builtin_popcountll(x);
#else
        int n = 0;

        while (x) {
                x &= x - 1;
                n++;
        }

        return n;
#endif
}

int othello_score(const othello_t *o, player_t p)
{
        return popcount(o->disks[p]);
}

#define NUM_DIRS 8

/* Shift disks in direction dir. */
static uint64_t shift(uint64_t disks, int dir)
{
        /* Note: the directions refer to how we shift the bits, not the
           positions on the board (where the least significant bit is
           the top-left corner). */

        static const uint64_t MASKS[] = {
                0x7F7F7F7F7F7F7F7FULL, /* Right. */
                0x007F7F7F7F7F7F7FULL, /* Down-right. */
                0xFFFFFFFFFFFFFFFFULL, /* Down. */
                0x00FEFEFEFEFEFEFEULL, /* Down-left. */
                0xFEFEFEFEFEFEFEFEULL, /* Left. */
                0xFEFEFEFEFEFEFE00ULL, /* Up-left. */
                0xFFFFFFFFFFFFFFFFULL, /* Up. */
                0x7F7F7F7F7F7F7F00ULL  /* Up-right. */
        };
        static const uint64_t LSHIFTS[] = {
                0, /* Right. */
                0, /* Down-right. */
                0, /* Down. */
                0, /* Down-left. */
                1, /* Left. */
                9, /* Up-left. */
                8, /* Up. */
                7  /* Up-right. */
        };
        static const uint64_t RSHIFTS[] = {
                1, /* Right. */
                9, /* Down-right. */
                8, /* Down. */
                7, /* Down-left. */
                0, /* Left. */
                0, /* Up-left. */
                0, /* Up. */
                0  /* Up-right. */
        };

        assert(dir >= 0 && dir < NUM_DIRS);

        if (dir < NUM_DIRS / 2) {
                assert(LSHIFTS[dir] == 0 && "Shifting right.");
                return (disks >> RSHIFTS[dir]) & MASKS[dir];
        } else {
                assert(RSHIFTS[dir] == 0 && "Shifting left.");
                return (disks << LSHIFTS[dir]) & MASKS[dir];
        }
}

static uint64_t generate_moves(uint64_t my_disks, uint64_t opp_disks)
{
        int dir;
        uint64_t x;
        uint64_t empty_cells = ~(my_disks | opp_disks);
        uint64_t legal_moves = 0;

        assert((my_disks & opp_disks) == 0 && "Disk sets should be disjoint.");

        for (dir = 0; dir < NUM_DIRS; dir++) {
                /* Get opponent disks adjacent to my disks in direction dir. */
                x = shift(my_disks, dir) & opp_disks;

                /* Add opponent disks adjacent to those, and so on. */
                x |= shift(x, dir) & opp_disks;
                x |= shift(x, dir) & opp_disks;
                x |= shift(x, dir) & opp_disks;
                x |= shift(x, dir) & opp_disks;
                x |= shift(x, dir) & opp_disks;

                /* Empty cells adjacent to those are valid moves. */
                legal_moves |= shift(x, dir) & empty_cells;
        }

        return legal_moves;
}

bool othello_has_valid_move(const othello_t *o, player_t p)
{
        return generate_moves(o->disks[p], o->disks[p ^ 1]) != 0;
}

bool othello_is_valid_move(const othello_t *o, player_t p, int row, int col)
{
        uint64_t mask = 1ULL << (row * 8 + col);

        assert(row >= 0 && row <= 7);
        assert(col >= 0 && col <= 7);

        return (generate_moves(o->disks[p], o->disks[p ^ 1]) & mask) != 0;
}

static void resolve_move(uint64_t *my_disks, uint64_t *opp_disks, int board_idx)
{
        int dir;
        uint64_t x, bounding_disk;
        uint64_t new_disk = 1ULL << board_idx;
        uint64_t captured_disks = 0;

        assert(board_idx < 64 && "Move must be within the board.");
        assert((*my_disks & *opp_disks) == 0 && "Disk sets must be disjoint.");
        assert(!((*my_disks | *opp_disks) & new_disk) && "Target not empty!");

        *my_disks |= new_disk;

        for (dir = 0; dir < NUM_DIRS; dir++) {
                /* Find opponent disk adjacent to the new disk. */
                x = shift(new_disk, dir) & *opp_disks;

                /* Add any adjacent opponent disk to that one, and so on. */
                x |= shift(x, dir) & *opp_disks;
                x |= shift(x, dir) & *opp_disks;
                x |= shift(x, dir) & *opp_disks;
                x |= shift(x, dir) & *opp_disks;
                x |= shift(x, dir) & *opp_disks;

                /* Determine whether the disks were captured. */
                bounding_disk = shift(x, dir) & *my_disks;
                captured_disks |= (bounding_disk ? x : 0);
        }

        assert(captured_disks && "A valid move must capture disks.");

        *my_disks ^= captured_disks;
        *opp_disks ^= captured_disks;

        assert(!(*my_disks & *opp_disks) && "The sets must still be disjoint.");
}

void othello_make_move(othello_t *o, player_t p, int row, int col)
{
        assert(othello_is_valid_move(o, p, row, col));

        resolve_move(&o->disks[p], &o->disks[p ^ 1], row * 8 + col);
}

static void frontier_disks(uint64_t my_disks, uint64_t opp_disks,
                           uint64_t *my_frontier, uint64_t *opp_frontier)
{
        uint64_t empty_cells = ~(my_disks | opp_disks);
        uint64_t x;
        int dir;

        *my_frontier = 0;
        *opp_frontier = 0;

        for (dir = 0; dir < NUM_DIRS; dir++) {
                /* Check cells adjacent to empty cells. */
                x = shift(empty_cells, dir);
                *my_frontier |= x & my_disks;
                *opp_frontier |= x & opp_disks;
        }
}

#define WIN_BONUS (1 << 20)

static int eval(uint64_t my_disks, uint64_t opp_disks,
                uint64_t my_moves, uint64_t opp_moves)
{
        static const uint64_t CORNER_MASK = 0x8100000000000081ULL;

        int my_disk_count, opp_disk_count;
        uint64_t my_corners, opp_corners;
        uint64_t my_frontier, opp_frontier;
        int score = 0;

        if (!my_moves && !opp_moves) {
                /* Terminal state. */
                my_disk_count = popcount(my_disks);
                opp_disk_count = popcount(opp_disks);
                return (my_disk_count - opp_disk_count) * WIN_BONUS;
        }

        my_corners = my_disks & CORNER_MASK;
        opp_corners = opp_disks & CORNER_MASK;

        frontier_disks(my_disks, opp_disks, &my_frontier, &opp_frontier);

        /* Optimize for corners, mobility and few frontier disks. */
        score += (popcount(my_corners) - popcount(opp_corners)) * 16;
        score += (popcount(my_moves) - popcount(opp_moves)) * 2;
        score += (popcount(my_frontier) - popcount(opp_frontier)) * -1;

        assert(abs(score) < WIN_BONUS);

        return score;
}

int othello_eval(const othello_t *o, player_t p)
{
        uint64_t my_disks, opp_disks, my_moves, opp_moves;

        my_disks = o->disks[p];
        opp_disks = o->disks[p ^ 1];
        my_moves = generate_moves(my_disks, opp_disks);
        opp_moves = generate_moves(opp_disks, my_disks);

        return eval(my_disks, opp_disks, my_moves, opp_moves);
}

static int negamax(uint64_t my_disks, uint64_t opp_disks, int max_depth,
                   int alpha, int beta, int *best_move, int *eval_count)
{
        uint64_t my_moves, opp_moves;
        uint64_t my_new_disks, opp_new_disks;
        int i, s, best;

        /* Generate moves. */
        my_moves = generate_moves(my_disks, opp_disks);
        opp_moves = generate_moves(opp_disks, my_disks);

        if (!my_moves && opp_moves) {
                /* Null move. */
                return -negamax(opp_disks, my_disks, max_depth, -beta, -alpha,
                                best_move, eval_count);
        }

        if (max_depth == 0 || (!my_moves && !opp_moves)) {
                /* Maximum depth or terminal state reached. */
                ++*eval_count;
                return eval(my_disks, opp_disks, my_moves, opp_moves);
        }

        /* Find the best move. */
        assert(alpha < beta);
        best = -INT_MAX;
        for (i = 0; i < 64; i++) {
                if (!(my_moves & (1ULL << i))) {
                        continue;
                }
                my_new_disks = my_disks;
                opp_new_disks = opp_disks;
                resolve_move(&my_new_disks, &opp_new_disks, i);

                s = -negamax(opp_new_disks, my_new_disks,
                             max_depth - 1, -beta, -alpha, NULL,
                             eval_count);

                if (s > best) {
                        best = s;
                        if (best_move) {
                                *best_move = i;
                        }
                        alpha = s > alpha ? s : alpha;

                        if (alpha >= beta) {
                                break;
                        }
                }
        }

        return best;
}

int othello_negamax(const othello_t *o, player_t p, int depth)
{
        int best_move, eval_count;

        return negamax(o->disks[p], o->disks[p ^ 1], depth, -INT_MAX, INT_MAX,
                       &best_move, &eval_count);
}

static int iterative_negamax(uint64_t my_disks, uint64_t opp_disks,
                             int start_depth, int eval_budget)
{
        int depth, best_move, eval_count, s;

        assert(start_depth > 0 && "At least one move must be explored.");

        eval_count = 0;
        best_move = -1;
        for (depth = start_depth; eval_count < eval_budget; depth++) {
                s = negamax(my_disks, opp_disks, depth, -INT_MAX, INT_MAX,
                            &best_move, &eval_count);
                if (s >= WIN_BONUS || -s >= WIN_BONUS) {
                        break;
                }
        }

        assert(best_move != -1 && "No move found?");

        return best_move;
}

int othello_iterative_negamax(const othello_t *o, player_t p, int budget)
{
        return iterative_negamax(o->disks[p], o->disks[p ^ 1], 1, budget);
}

void othello_compute_move(const othello_t *o, player_t p, int *row, int *col)
{
        int move_idx;

        static const int START_DEPTH = 8;
        static const int EVAL_BUDGET = 500000;

        assert(othello_has_valid_move(o, p));

        move_idx = iterative_negamax(o->disks[p], o->disks[p ^ 1],
                                     START_DEPTH, EVAL_BUDGET);

        *row = move_idx / 8;
        *col = move_idx % 8;
}

void othello_compute_random_move(const othello_t *o, player_t p,
                                 int *row, int *col)
{
        uint64_t moves;
        int i;

        moves = generate_moves(o->disks[p], o->disks[p ^ 1]);

        do {
                i = rand() % 64;
        } while (!(moves & (1ULL << i)));

        *row = i / 8;
        *col = i % 8;
}

void othello_to_string(const othello_t *o, char *s)
{
        int row, col;
        char c;

        s += sprintf(s, " abcdefgh \n");
        for (row = 0; row < 8; row++) {
                s += sprintf(s, "%d", row + 1);
                for (col = 0; col < 8; col++) {
                        switch (othello_cell_state(o, row, col)) {
                        case CELL_BLACK:
                                c = 'x';
                                break;
                        case CELL_WHITE:
                                c = 'o';
                                break;
                        default:
                                c = '.';
                                break;
                        }
                        *s++ = c;
                }
                s += sprintf(s, "%d\n", row + 1);
        }
        sprintf(s, " abcdefgh \n");
}

void othello_from_string(const char *s, othello_t *o)
{
        size_t len, i;
        int j;

        o->disks[PLAYER_BLACK] = 0;
        o->disks[PLAYER_WHITE] = 0;

        len = strlen(s);
        j = 0;
        for (i = 0; i < len; i++) {
                switch (s[i]) {
                case '.':
                        /* Blank cell. */
                        j++;
                        break;
                case 'x':
                        /* Black disk. */
                        othello_set_cell_state(o, j / 8, j % 8, CELL_BLACK);
                        j++;
                        break;
                case 'o':
                        /* White disk. */
                        othello_set_cell_state(o, j / 8, j % 8, CELL_WHITE);
                        j++;
                        break;
                default:
                        break;
                }
        }

        assert(j == 64 && "Exactly 64 cells must be provided.");
}
