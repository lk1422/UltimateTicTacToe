from ultimate import UltimateTicTacToe

ttt = UltimateTicTacToe()
for i in range(3):
    for j in range(3):
        print(ttt.place_mark(3*i, 2, j))
        ttt.print_board()
print(ttt.resolutions)
        

