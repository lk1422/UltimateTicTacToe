from typing import List

X_MARK =  1
O_MARK = -1
EMPTY  =  0

class UltimateTicTacToe():

    def __init__(self):
        #The actual board
        self.board = [[[EMPTY for _ in range(3)] for _ in range(3)] for _ in range(9)] 
        #Board to move into
        self.sub_board = -1 
        #Winning States
        self.resolutions = [EMPTY for _ in range(9)] 
        self.turn = X_MARK
        self.game_over = False


    def _check_line(sub_board : List[List[int]], dx : int, dy : int, x : int, y : int):
        """
        Parameters:
            sub_board: tic tac toe board which we are checking
                       for winners.
            x: column we are checking for solutions
            y: row we are checking for solutions
            dx: The direction we are iterating over in the cols
            dy: The direction we are iterating over in the rows
        Returns:
            1, 0, -1:  1 if board in row y column x is won by X, -1 if
                         solved by O, 0 if undecieded.
        """
        assert -1 <= dx and dx <= 1, "Invalid dx"
        assert -1 <= dy and dy <= 1, "Invalid dy"
        assert x >= 0 and x < 3, "Invalid Row."
        assert y >= 0 and y < 3, "Invalid Col."

        positive_dx_dy = dx*2 + x >= 3 or dy*2 + y >= 3
        negative_dx_dy = dx*2 + x < 0 or dy*2 + y < 0
        if (positive_dx_dy or negative_dx_dy): return 0
        X_winner, Y_winner = True, True
        for i in range(3):
            X_winner = X_winner and (sub_board[y + i*dy][x + i*dx] ==  1)
            Y_winner = Y_winner and (sub_board[y + i*dy][x + i*dx] == -1)

        if not (X_winner or Y_winner): return EMPTY
        return 1 if X_winner else -1


    def _check_board_index(sub_board: List[List[int]] , x: int, y : int):
        """
        Parameters:
            sub_board: tic tac toe board which we are checking
                       for winners.
            x: column we are checking for solutions
            y: row we are checking for solutions
        Returns:
            1, 0, -1:  1 if board in row y column x is won by X, -1 if
                         solved by O, 0 if undecieded.
        """
        assert x >= 0 and x < 3, "Invalid Row."
        assert y >= 0 and y < 3, "Invalid Col."

        for dx in range(-1, 2):
            for dy in range(-1, 2):

                if dx == dy and dx == 0:
                    continue

                winner = UltimateTicTacToe._check_line(sub_board, dx, dy, x, y)
                if winner != EMPTY: return winner

        return 0

    def _sub_solved(sub_board : List[List[int]]):
        """
        Parameters:
            sub_board: 3x3 tic tac toe board.
        Returns:
            True: if board[i] contains a
            winner.
            False: No winner on board[i]
        """

        for y in range(3):
            for x in range(3):

                winner = UltimateTicTacToe._check_board_index(sub_board, x, y)
                if winner != 0: return winner

        return 0
    
    def check_state(self):
        """
        Checks to see if the game has ended if so sets the game_over
        flag and returns True else returns False

        Parameters:
            None
        Returns:
            True if game has terminated
            False if not
        """
        board = [[0 for _ in range(3)] for _ in range(3)]
        all_filled = True
        for i in range(9):
            all_filled = all_filled and (self.resolutions[i] != 0)
            board[i//3][i%3] = self.resolutions[i]
        print(board)
        winner = UltimateTicTacToe._sub_solved(board)
        vote = sum(self.resolutions)
        if winner == 0 and all_filled: winner =  sum(vote)/abs(vote)
        self.game_over = (winner!=0)
        return self.game_over


    def place_mark(self, board : int, x : int, y : int):
        """
        Places the mark on the corresponding square, if this mark
        results in a victory this will be marked in the resolutions,
        array, it will also update the sub_board variable to the 
        corresponding sub_board. Must place a square in an 
        unresolved board (no winner).

        Parameters:
            board : tic tac toe board which we are checking
                       for winners.
            x: column where we are placing the mark
            y: row where we are placing the mark
        Returns:
            True if game has terminated
            False if not
        """
        assert 0 <= board and board < 9, "Invlaid Board."
        assert x >= 0 and x < 3, "Invalid Row."
        assert y >= 0 and y < 3, "Invalid Col."
        assert self.resolutions[board] == 0, "Invalid sub_board."

        self.board[board][y][x] = self.turn
        #self.turn = self.turn - 2*self.turn
        self.resolutions[board] = UltimateTicTacToe._sub_solved(self.board[board])
        self.sub_board = -1 if UltimateTicTacToe._sub_solved(self.board[y*3 + x]) else y*3 + x
        return self.check_state()


    def _generate_2d_array(self):
        """
        Generates a 2d array representation of the 3d board. This
        is easier to use when outputing the state of the board.
        
        Parameters:
            None
        Returns:
            arr: a 2d representation of the board
        """
        arr = [[0 for _ in range(9)] for _ in range(9)]
        for i in range(9):
            sub_x = 3 * (i%3) 
            sub_y =  3 *(i // 3)
            for local_y in range(3):
                for local_x in range(3):
                    arr[sub_y + local_y][sub_x + local_x] = self.board[i][local_y][local_x]
        return arr


    def _convert_marker(value):
        """
        Parameters:
            value: X_MARK (1), O_MARK(-1), EMPTY = 0
        Returns:
            None
        """
        assert value == X_MARK or value == O_MARK or value == 0, "Invalid Marker."

        if value == X_MARK: return "X"
        if value == O_MARK: return  "O"
        return "."


    def print_board(self):
        """
        Prints the current state of the board to stdout.
        
        Parameters:
            None
        Returns:
            None
        """
        b = self._generate_2d_array()
        for  y in range(9):
            line = ""
            for x in range(9):
                if x!=0 and x%3 == 0:
                    line += "|\t"
                mark = UltimateTicTacToe._convert_marker(b[y][x])
                line += "| " + mark + " "
            line += "|"
            print(line)
            if (y+1)%3 == 0:
                print()





