#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/xpm.h>
#include <sys/select.h>
#include <unistd.h>

#include "othello.h"
#include "icons/othello_icon.xpm"

static othello_t board;
static enum { BLACKS_MOVE, WHITES_MOVE, GAME_OVER } state;

static struct {
        int x, y;      /* Position of the grid relative to window origin. */
        int size;      /* Size (width and height are equal) of the grid. */
        int cell_size; /* Size of a grid cell, not including its border. */
        int sel_row;   /* Currently selected row, or -1 if none. */
        int sel_col;   /* Currently selected column. */
} grid;

static Display *display;
static Window win;
static GC black_gc;
static GC white_gc;
static GC board_gc;     /* For the board background. */
static GC highlight_gc; /* For selected grid cells. */
static XFontStruct *font;
static Atom wm_delete_window;

static void err(const char *msg, ...)
{
        va_list ap;

        va_start(ap, msg);
        fprintf(stderr, "error: ");
        vfprintf(stderr, msg, ap);
        fprintf(stderr, "\n");
        va_end(ap);
        exit(EXIT_FAILURE);
}

#define CELL_GAP 1       /* Cell gap in pixels. */
#define FONT_NAME "9x15" /* Font for labels and status. */
#define MIN_SIZE 300     /* Minimum window size. */
#define INIT_SIZE 450    /* Initial window size. */

static unsigned long alloc_color(uint8_t red, uint8_t green, uint8_t blue)
{
        XColor color;
        Colormap map;

        map = DefaultColormap(display, DefaultScreen(display));
        color.red   = red   * 256;
        color.green = green * 256;
        color.blue  = blue  * 256;

        if (!XAllocColor(display, map, &color)) {
                err("XAllocColor failed");
        }

        return color.pixel;
}

static void init(int argc, char **argv)
{
        XSizeHints *size_hints;
        XWMHints *wm_hints;
        XClassHint *class_hint;
        unsigned long black_color, white_color, grey_color;
        unsigned long board_color, hl_color;
        char *window_name = "Othello";
        XTextProperty window_name_prop;

        /* Connect to the display. */
        if (!(display = XOpenDisplay(NULL))) {
                err("cannot connect to X server %s", XDisplayName(NULL));
        }

        /* Allocate colours. */
        black_color = alloc_color(0x00, 0x00, 0x00);
        white_color = alloc_color(0xFF, 0xFF, 0xFF);
        grey_color  = alloc_color(0xC0, 0xC0, 0xC0);
        board_color = alloc_color(0x00, 0x80, 0x00);
        hl_color    = alloc_color(0x00, 0xAA, 0x00);

        /* Create the window. */
        win = XCreateSimpleWindow(display,
                                  RootWindow(display, DefaultScreen(display)),
                                  0, 0, INIT_SIZE, INIT_SIZE, CELL_GAP,
                                  black_color, grey_color);

        /* Prepare window name property. */
        if (!XStringListToTextProperty(&window_name, 1, &window_name_prop)) {
                err("XStringListToTextProperty failed");
        }

        /* Prepare size hints. */
        if (!(size_hints = XAllocSizeHints())) {
                err("XAllocSizeHints() failed");
        }
        size_hints->flags = PMinSize;
        size_hints->min_width = MIN_SIZE;
        size_hints->min_height = MIN_SIZE;

        /* Prepare window manager hints. */
        if (!(wm_hints = XAllocWMHints())) {
                err("XAllocWMHints() failed");
        }
        wm_hints->initial_state = NormalState;
        wm_hints->input = True;
        wm_hints->flags = StateHint | InputHint;
        if (XpmCreatePixmapFromData(display, win, othello_icon,
                                    &wm_hints->icon_pixmap,
                                    &wm_hints->icon_mask, NULL) == XpmSuccess) {
                wm_hints->flags |= IconPixmapHint | IconMaskHint;
        }

        /* Prepare class hint. */
        if (!(class_hint = XAllocClassHint())) {
                err("XAllocClassHint() failed");
        }
        class_hint->res_name = argv[0];
        class_hint->res_class = window_name;

        /* Set name property, size, wm and class hints for the window. */
        XSetWMProperties(display, win, &window_name_prop, &window_name_prop,
                         argv, argc, size_hints, wm_hints, class_hint);
        XFree(window_name_prop.value);
        XFree(size_hints);
        XFree(wm_hints);
        XFree(class_hint);

        /* Register for events. */
        XSelectInput(display, win, ExposureMask | KeyPressMask |
                        ButtonPressMask | StructureNotifyMask |
                        PointerMotionMask | PointerMotionHintMask);

        wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
        if (!XSetWMProtocols(display, win, &wm_delete_window, 1)) {
                err("XSetWMProtocols failed");
        }

        /* Load the font. */
        if (!(font = XLoadQueryFont(display, FONT_NAME))) {
                err("cannot open %s font", FONT_NAME);
        }

        /* Set up GCs. */
        black_gc = XCreateGC(display, win, 0, NULL);
        XSetForeground(display, black_gc, black_color);
        XSetFont(display, black_gc, font->fid);

        white_gc = XCreateGC(display, win, 0, NULL);
        XSetForeground(display, white_gc, white_color);

        board_gc = XCreateGC(display, win, 0, NULL);
        XSetForeground(display, board_gc, board_color);

        highlight_gc = XCreateGC(display, win, 0, NULL);
        XSetForeground(display, highlight_gc, hl_color);

        /* Show the window. */
        XMapWindow(display, win);
}

static int min(int x, int y)
{
        return x < y ? x : y;
}

static int max(int x, int y) {
        return x > y ? x : y;
}

/* Compute the grid's size and position in the window. */
static void compute_grid_position(int win_width, int win_height)
{
        /* The grid is a 10x10 grid. The 8x8 centre is the Othello
           board, the top row and left column are used for labels, and the
           bottom row for status text. */

        grid.cell_size = (min(win_width, win_height) - 9 * CELL_GAP) / 10;
        grid.size = grid.cell_size * 10 + 9 * CELL_GAP;
        grid.x = win_width / 2 - grid.size / 2;
        grid.y = win_height / 2 - grid.size / 2;
}

/* Check whether the position is over an Othello cell. */
static bool grid_hit_test(int x, int y, int *row, int *col)
{
        *row = (y - grid.y) / (grid.cell_size + CELL_GAP) - 1;
        *col = (x - grid.x) / (grid.cell_size + CELL_GAP) - 1;

        if (*row >= 0 && *row < 8 && *col >= 0 && *col < 8) {
                return true;
        }

        return false;
}

/* Draw an Othello cell and its contents. */
static void draw_othello_cell(int row, int col)
{
        int x, y;
        bool highlight;
        cell_state_t cs;

        x = grid.x + (col + 1) * (grid.cell_size + CELL_GAP);
        y = grid.y + (row + 1) * (grid.cell_size + CELL_GAP);

        highlight = (row == grid.sel_row && col == grid.sel_col &&
                     state == BLACKS_MOVE);

        /* Draw the cell background. */
        XFillRectangle(display, win, highlight ? highlight_gc : board_gc,
                       x, y, grid.cell_size, grid.cell_size);

        if ((cs = othello_cell_state(&board, row, col)) != CELL_EMPTY) {
                /* Draw the disk. */
                XFillArc(display, win, cs == CELL_BLACK ? black_gc : white_gc,
                         x, y, grid.cell_size, grid.cell_size, 0, 360 * 64);
        }
}

/* Draw string s of length len centered at (x,y). */
static void draw_string(const char *s, int len, int x, int y)
{
        int width, height;

        width = XTextWidth(font, s, len);
        height = font->ascent + font->descent;

        XDrawString(display, win, black_gc,
                    x - width / 2, y + height / 2,
                    s, len);
}

/* Draw the grid and its contents. */
static void draw_grid(void)
{
        int row, col, x, y, bs, ws;
        char status[128];

        XClearWindow(display, win);

        /* Draw a background square around the 8x8 centre cells. */
        XFillRectangle(display, win, black_gc,
                       grid.x + grid.cell_size,
                       grid.y + grid.cell_size,
                       8 * grid.cell_size + 9 * CELL_GAP,
                       8 * grid.cell_size + 9 * CELL_GAP);

        /* Draw labels. */
        for (row = 0; row < 8; row++) {
                x = grid.x + grid.cell_size / 2;
                y = grid.y + (row + 1) * (grid.cell_size + CELL_GAP) +
                        grid.cell_size / 2;
                draw_string(&"12345678"[row], 1, x, y);
        }
        for (col = 0; col < 8; col++) {
                x = grid.x + (col + 1) * (grid.cell_size + CELL_GAP) +
                        grid.cell_size / 2;
                y = grid.y + grid.cell_size / 2;
                draw_string(&"ABCDEFGH"[col], 1, x, y);
        }

        /* Draw status text. */
        switch (state) {
        case BLACKS_MOVE:
                sprintf(status, "Human's move.");
                break;
        case WHITES_MOVE:
                sprintf(status, "Computer's move..");
                break;
        case GAME_OVER:
                bs = othello_score(&board, PLAYER_BLACK);
                ws = othello_score(&board, PLAYER_WHITE);
                if (bs > ws) {
                        sprintf(status, "Human wins %d-%d!", bs, ws);
                } else if (ws > bs) {
                        sprintf(status, "Computer wins %d-%d!", ws, bs);
                } else {
                        sprintf(status, "Draw!");
                }
        }
        draw_string(status, strlen(status), grid.x + grid.size / 2,
                    grid.y + grid.size - grid.cell_size / 2);

        /* Draw cells. */
        for (row = 0; row < 8; row++) {
                for (col = 0; col < 8; col++) {
                        draw_othello_cell(row, col);
                }
        }
}

static int white_move_pipe[2]; /* [0] for reading, [1] for writing. */

static void compute_white_move(void)
{
        int x, row, col;
        char c;

        assert(state == WHITES_MOVE);

        /* Compute white's move in a background process. */
        if ((x = fork()) == -1) {
                err("fork() failed: %s", strerror(errno));
        } else if (x != 0) {
                /* Parent process. */
                return;
        }

        /* Child process: compute the move and send it through the pipe. */
        othello_compute_move(&board, PLAYER_WHITE, &row, &col);
        c = (char)(row * 8 + col);
        if (write(white_move_pipe[1], &c, 1) != 1) {
                err("write() failed");
        }

        exit(EXIT_SUCCESS);
}

/* Make a move for the current player and transition the game state. */
static void make_move(int row, int col)
{
        assert(state == BLACKS_MOVE || state == WHITES_MOVE);

        if (state == BLACKS_MOVE) {
                if (!othello_is_valid_move(&board, PLAYER_BLACK, row, col)) {
                        /* Illegal move; ignored. */
                        return;
                }

                othello_make_move(&board, PLAYER_BLACK, row, col);
                state = WHITES_MOVE;
        } else {
                othello_make_move(&board, PLAYER_WHITE, row, col);
                state = BLACKS_MOVE;
        }

        if (!othello_has_valid_move(&board, PLAYER_BLACK) &&
            !othello_has_valid_move(&board, PLAYER_WHITE)) {
                state = GAME_OVER;
        } else if (state == WHITES_MOVE &&
                   !othello_has_valid_move(&board, PLAYER_WHITE)) {
                state = BLACKS_MOVE;
        } else if (state == BLACKS_MOVE &&
                   !othello_has_valid_move(&board, PLAYER_BLACK)) {
                state = WHITES_MOVE;
        }

        if (state == WHITES_MOVE) {
                compute_white_move();
        }
}

static void new_game(void)
{
        othello_init(&board);
        state = BLACKS_MOVE;
}

static void on_mouse_click(void)
{
        if (state == GAME_OVER) {
                new_game();
                return;
        }

        if (state == BLACKS_MOVE && grid.sel_row >= 0) {
                make_move(grid.sel_row, grid.sel_col);
        }
}

static void select_othello_cell(int row, int col)
{
        int old_row = grid.sel_row;
        int old_col = grid.sel_col;

        if (row == old_row && col == old_col) {
                /* This cell is already selected. */
                return;
        }

        grid.sel_row = row;
        grid.sel_col = col;

        if (old_row >= 0) {
                /* Re-draw the previously selected cell. */
                draw_othello_cell(old_row, old_col);
        }

        if (row >= 0) {
                /* Draw the newly selected cell. */
                draw_othello_cell(row, col);
        }
}

static void on_mouse_move(void)
{
        Window root, child;
        int root_x, root_y, win_x, win_y, row, col;
        unsigned mask;

        if (!XQueryPointer(display, win, &root, &child, &root_x, &root_y,
                           &win_x, &win_y, &mask)) {
                return;
        }

        if (grid_hit_test(win_x, win_y, &row, &col)) {
                select_othello_cell(row, col);
        } else {
                select_othello_cell(-1, -1);
        }
}

static void on_key_press(XKeyEvent *xkey, bool *quit, bool *draw)
{
        int row, col;

        row = grid.sel_row;
        col = grid.sel_col;

        switch (XLookupKeysym(xkey, 0)) {
        default:
                return;
        case XK_q:
                *quit = true;
                return;
        case XK_space:
        case XK_Return:
                on_mouse_click();
                *draw = true;
                return;
        case XK_Right: col++; break;
        case XK_Left:  col--; break;
        case XK_Down:  row++; break;
        case XK_Up:    row--; break;
        case XK_a:     col = 0; break;
        case XK_b:     col = 1; break;
        case XK_c:     col = 2; break;
        case XK_d:     col = 3; break;
        case XK_e:     col = 4; break;
        case XK_f:     col = 5; break;
        case XK_g:     col = 6; break;
        case XK_h:     col = 7; break;
        case XK_1:     row = 0; break;
        case XK_2:     row = 1; break;
        case XK_3:     row = 2; break;
        case XK_4:     row = 3; break;
        case XK_5:     row = 4; break;
        case XK_6:     row = 5; break;
        case XK_7:     row = 6; break;
        case XK_8:     row = 7; break;
        }

        select_othello_cell(max(0, min(row, 7)), max(0, min(col, 7)));
}

static void event_loop(void)
{
        int display_fd;
        bool quit, draw;
        fd_set fds;
        XEvent event;
        char c;

        display_fd = XConnectionNumber(display);
        quit = false;
        draw = false;

        while (!quit) {
                if (draw) {
                        draw_grid();
                        draw = false;
                }

                if (XPending(display) == 0) {
                        /* Wait for X event or a white move. */
                        FD_ZERO(&fds);
                        FD_SET(display_fd, &fds);
                        FD_SET(white_move_pipe[0], &fds);
                        if (select(max(display_fd, white_move_pipe[0]) + 1,
                                   &fds, NULL, NULL, NULL) == -1) {
                                err("select() failed: %s", strerror(errno));
                        }

                        if (FD_ISSET(white_move_pipe[0], &fds)) {
                                /* Read white move from the pipe. */
                                if (read(white_move_pipe[0], &c, 1) != 1) {
                                        err("read() failed");
                                }
                                make_move(c / 8, c % 8);
                                draw = true;
                                continue;
                        }
                }

                XNextEvent(display, &event);

                switch (event.type) {
                case ConfigureNotify:
                        /* The window's configuration has changed. */
                        compute_grid_position(event.xconfigure.width,
                                              event.xconfigure.height);
                        break;
                case Expose:
                        /* The window has become visible. */
                        if (event.xexpose.count == 0) {
                                draw = true;
                        }
                        break;
                case MotionNotify:
                        on_mouse_move();
                        break;
                case KeyPress:
                        on_key_press(&event.xkey, &quit, &draw);
                        break;
                case ButtonPress:
                        on_mouse_move();
                        on_mouse_click();
                        draw = true;
                        break;
                case ClientMessage:
                        if (event.xclient.data.l[0] == wm_delete_window) {
                                /* Window closed. */
                                quit = true;
                        }
                        break;
                }
        }
}

int main(int argc, char **argv)
{
        if (pipe(white_move_pipe) != 0) {
                err("pipe() failed: %s\n", strerror(errno));
        }

        grid.sel_row = -1;
        init(argc, argv);
        new_game();

        event_loop();

        XFreeGC(display, black_gc);
        XFreeGC(display, white_gc);
        XFreeGC(display, board_gc);
        XFreeGC(display, highlight_gc);
        XFreeFont(display, font);
        XCloseDisplay(display);

        return 0;
}
