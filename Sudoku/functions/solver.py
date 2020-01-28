
# Combine sudoku based on predicted numbers and their location
def combine_sudoku(locations,numbers):

    sudoku = []
    for i in range(0,9):
        sudoku.append([])
        for j in range(0,9):
            sudoku[i].append(0)

    for index,location in enumerate(locations):
        row = location[0]
        col = location[1]
        sudoku[row][col] = numbers[index]

    return sudoku

# Find next empty cell
def find_empty(sudoku):

    for i in range(0,9):
        for j in range(0,9):
            if sudoku[i][j] == 0:
                return (i,j)

    return None


# Check if number is valid to use
def valid(sudoku,number,pos):

    #Check row
    for i in range(len(sudoku[0])):
        if sudoku[pos[0]][i] == number and pos[1] != i:
            return False

    for i in range(len(sudoku[0])):
        if sudoku[i][pos[1]] == number and pos[0] != i:
            return False

    #Check box
    box_x = pos[1] // 3
    box_y = pos[0] //3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if number == sudoku[i][j] and (i,j) != pos:
                return False

    return True

# Call until the sudoku is solved or no solution is found
def solve(sudoku):

    find = find_empty(sudoku)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(sudoku, i, (row, col)):
            sudoku[row][col] = i

            if solve(sudoku):
                return True

            sudoku[row][col] = 0

    return False



if __name__ == '__main__':

    locations = [(0,1),(0,4),(1,5),(1,6),(1,7),(2,0),(2,3),(2,8),(3,2),(3,5),(3,7),(3,8),(4,4),
                 (5,0),(5,1),(5,3),(5,6),(6,0),(6,5),(6,8),(7,1),(7,2),(7,3),(8,4),(8,7)]


    numbers = [2,1,9,3,4,4,5,2,6,2,3,5,9,1,4,8,9,6,5,7,7,1,4,8,2]

    sudoku = combine_sudoku(locations,numbers)

    solve(sudoku)

