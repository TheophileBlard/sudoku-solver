
class SudokuGrid:
    def __init__(self, digits):
        assert(len(digits) == 9)
        for i in range(9):
            assert(len(digits[i]) == 9)

        self.digits = digits

    def __str__(self):
        output = ''
        for i in range(9):
            for j in range(9):
                output += str(self.digits[i][j])
                if j==2 or j==5:
                    output += '|'
                if j==8:
                    output += '\n'
            if i==2 or i==5:
                output += 11*'-'
                output += '\n'
        return output