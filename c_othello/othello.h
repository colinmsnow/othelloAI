#ifndef OTHELLO_H
#define OTHELLO_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
        uint64_t disks[2];
} othello_t;

typedef enum {
        CELL_BLACK = 0,
        CELL_WHITE = 1,
        CELL_EMPTY = 2
} cell_state_t;

typedef enum {
        PLAYER_BLACK = 0,
        PLAYER_WHITE = 1
} player_t;


/* Note: rows and columns are zero-indexed, i.e. between 0 and 7 inclusive. */

/* Initialize game state. */
void othello_init(othello_t *o);

/* Get the state of a cell. */
cell_state_t othello_cell_state(const othello_t *o, int row, int col);

/* Set the state of a cell. */
void othello_set_cell_state(othello_t *o, int row, int col, cell_state_t s);

/* Get a player's score. */
int othello_score(const othello_t *o, player_t p);

/* Return true if the player can make a valid move. */
bool othello_has_valid_move(const othello_t *o, player_t p);

/* Return true if the move is valid. */
bool othello_is_valid_move(const othello_t *o, player_t p, int row, int col);

/* Make a move. */
void othello_make_move(othello_t *o, player_t p, int row, int col);

/* Compute a good move for player p. */
void othello_compute_move(const othello_t *o, player_t p, int *row, int *col);



/* Utilities for testing, benchmarking, etc. */
void othello_to_string(const othello_t *o, char *s);
void othello_from_string(const char *s, othello_t *o);
void othello_compute_random_move(const othello_t *o, player_t p,
                                 int *row, int *col);
int othello_eval(const othello_t *o, player_t p);
int othello_negamax(const othello_t *o, player_t p, int depth);
int othello_iterative_negamax(const othello_t *o, player_t p, int budget);

#endif
