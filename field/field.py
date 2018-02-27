from point import Point


class Field:

    def __init__(self):
        self.width = None
        self.height = None
        self.cells = None

    def Clone(self):
        """ Create a deep clone of this field.
        """
        ft = Field()
        ft.width = self.width
        ft.height = self.height
        ft.cells = [self.cells[i][:] for i in range(len(self.cells))]
        return ft

    def parse(self, field_input):
        self.cells = [[] for _ in range(self.width)]
        x = 0
        y = 0

        for cell in field_input.split(','):
            self.cells[x].insert(y, cell)
            x += 1

            if x == self.width:
                x = 0
                y += 1

    def count_neighbors(self, x, y):
        count_0 = 0
        count_1 = 0
        coords = [(x-1,y-1), (x-1,y), (x-1,y+1),
                  (x,y-1), (x,y+1),
                  (x+1,y-1), (x+1,y), (x+1,y+1)]
        for coord in coords:
            if coord[0]>=0 and coord[0]<self.width and coord[1]>=0 and coord[1]<self.height:
                if self.cells[coord[0]][coord[1]] == '0':
                    count_0 += 1
                elif self.cells[coord[0]][coord[1]] == '1':
                    count_1 += 1
        return [count_0 + count_1, count_0, count_1]

    def get_cell_mapping(self):
        cell_map = {}

        for x in range(self.width):
            for y in range(self.height):
                cell = self.cells[x][y]
                cell_type_list = cell_map.get(cell, [])
                cell_type_list.append(Point(x, y))
                cell_map[cell] = cell_type_list

        return cell_map

    def pprint(self):
        for j in xrange(self.height):
            line = ""
            for i in xrange(self.width):
                line = line + self.cells[i][j]
            print (line)