from typing import List
import torch

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
        self.winner = 0


    def step(self, q_func, action):
        """
        Preforms the action then returns the subsequent state and
        reward, as well as the terminal flag.

        Parameters:
            action: action to be preformed
            q_func: current estimated state-action function, used for generating
                    adverserial moves.
        Returns:
           next_state: Tensor representing state after preforming action a
           reward: int representing the reward for the action in the state
           game_over: True/False for terminal/non-terminal
        """

        assert 0 <= action and action < 81

        sub_board, sub_x, sub_y = UltimateTicTacToe._action_to_coord(action)
        self.place_mark(sub_board, sub_x, sub_y)
        if self.game_over:
            return self.get_state(), self.winner, self.game_over
        #Generate Adverserial move
        adv_state = self.get_state(adverse=True)
        actions = self.get_legal_actions().to(torch.long)
        a = q_func.generate_action(adv_state, actions)
        sub_board, sub_x, sub_y = UltimateTicTacToe._action_to_coord(a)
        self.place_mark(sub_board, sub_x, sub_y)
        return self.get_state(), self.winner, self.game_over

    
    def _action_to_coord(action: int):
        """
        Converts the 0-80 action to the sub_board, sub_x, sub_y coordinate.

        Parameters:
            action: action to be preformed
        Returns:
            sub_board: 0-8 which sub_board this actions refers to
            sub_x: 0-2 refers to the column of the sub_board
            sub_y: 0-2 refers to the row of the sub_board
        """
        assert 0 <= action and action < 81

        sub_board = action//9
        sub_y = (action - sub_board*9)//3
        sub_x = (action - sub_board*9)%3
        return sub_board, sub_x, sub_y






    def get_legal_actions(self):
        """
        Returns a tensor representing a list of valid moves. IE which squares
        are empty and in the corresponding sub_board.

        Parameters:
            None
        Returns:
           Tensor representing the state
        """

        squares = []
        if self.sub_board != -1:
            base = 9*self.sub_board
            for y in range(3):
                for x in range(3):
                    if self.board[self.sub_board][y][x] == EMPTY:
                        squares.append(base + y*3 + x)
            return torch.Tensor(squares)
        else:
            b = torch.Tensor(self.board).reshape(-1)
            for square in range(b.shape[0]):
                sub_board = square//9
                if b[square] == EMPTY and self.resolutions[sub_board]==0:
                    squares.append(square)
            return torch.Tensor(squares)



    def get_state(self, adverse=False):
        """
        Returns a tensor representation of the game state:
            state[:10] represents which sub board is a valid move state[9]
            is for the option to move anywhere. If adverserial generates
            the state for the opponent by flipping the signs of the board.

        Parameters:
            adverse: boolearn representing whether or not to generate the adverserial state
        Returns:
           Tensor representing the state
        """

        active_board = [0 for _ in range(10)]
        active_board[self.sub_board] = 1
        active_board = torch.Tensor(active_board)
        board_tensor = torch.Tensor(self.board).reshape(-1)
        board_tensor -1*board_tensor if adverse else board_tensor
        state = torch.cat((active_board, board_tensor), dim=0)
        return state



    def reset(self):
        """
        Resets the board back to the original state of the game.

        Parameters:
            None
        Returns:
            None
        """

        #The actual board
        self.board = [[[EMPTY for _ in range(3)] for _ in range(3)] for _ in range(9)] 
        #Board to move into
        self.sub_board = -1 
        #Winning States
        self.resolutions = [EMPTY for _ in range(9)] 
        self.turn = X_MARK
        self.game_over = False
        self.winner = 0


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
             1:  if board[i] has been won by X
            -1:  if board[i] has been won by O
            0.5: if board[i] is filled (TIE)
        """

        for y in range(3):
            for x in range(3):

                winner = UltimateTicTacToe._check_board_index(sub_board, x, y)
                if winner != 0: return winner

        if UltimateTicTacToe._sub_board_filled(sub_board): return 0.5
        return 0


    def _sub_board_filled(sub_board: List[List[int]]):
        """
        Checks to see if the sub_board, board is completely full.

        Parameters:
            sub_board: the sub_board (3x3)
        Returns:
            True if board full
            False if not
        """
        for y in range(3):
            for x in range(3):
                if sub_board[y][x] == 0:
                    return False
        return True


    def _board_filled(self):
        """
        Checks to see if the board is completely full.

        Parameters:
            None
        Returns:
            True if board full
            False if not
        """
        for i in range(9):
            for y in range(3):
                for x in range(3):
                    if self.board[i][y][x] == 0:
                        return False
        return True


    
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
            board[i//3][i%3] = int(self.resolutions[i])
        winner = UltimateTicTacToe._sub_solved(board)
        vote_arr = [int(item) for item in self.resolutions]
        vote = sum(vote_arr)
        vote_win = 0 if vote==0 else vote/abs(vote)
        if winner == 0.5 or all_filled: #Board full
            self.winner = vote_win
            self.game_over = True
            return self.game_over
        else: #Winning pattern or not gamve over
            self.winner = winner
            self.game_over = (winner!=0)
            return self.game_over

    def debug(self, board,x,y):
        self.print_board()
        print(board, x, y)
        print(self.get_legal_actions())
        print(self.sub_board)
        print(self.resolutions)
        return ""


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
        assert self.resolutions[board] == 0, "Invalid sub_board." + str(self.debug(board,x,y))
        assert self.board[board][y][x] == EMPTY

        self.board[board][y][x] = self.turn
        self.turn = self.turn - 2*self.turn
        self.resolutions[board] = UltimateTicTacToe._sub_solved(self.board[board])
        self.sub_board = -1 if UltimateTicTacToe._sub_solved(self.board[y*3 + x]) \
                            else y*3 + x
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


    def debug_play_agent(self, model):
        while not self.game_over:
            b = self.sub_board
            if self.turn == X_MARK:
                action = model.generate_action(self.get_state(), self.get_legal_actions())
                sub_board, sub_x, sub_y = UltimateTicTacToe._action_to_coord(action)
                self.place_mark(sub_board, sub_x, sub_y)
                print("PLAYED:", sub_board, sub_y, sub_x)
            else:
                if self.sub_board == -1:
                    b = int(input("Enter sub board: ").strip())
                row = int(input("Enter row: ").strip())
                col = int(input("Enter col: ").strip())
                self.place_mark(b, col, row)
            self.print_board()
            print(self.resolutions)
        print("Game Over Winner:", self.winner)
