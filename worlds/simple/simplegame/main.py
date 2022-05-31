
import random

from pynput.keyboard import Key, Listener

mapsize: int = 200


class Tile:
    def __init__(self):
        self.terrain = "."  # it's grass
        self.contents = []

    def tostring(self) -> str:
        for t in self.terrain:
            return t
        return self.terrain


class Player:
    def __init__(self):
        self.x = int(mapsize / 2)
        self.y = int(mapsize / 2)
        self.satiation = 10
        self.water = 10


def create_world(world: [Tile]):
    print(world)
    for i in range(len(world)):
        for j in range(len(world[0])):
            world[i][j] = Tile()
            if random.random() < 0.2:
                world[i][j].terrain = "T"  # it's a tree
            if random.random() < 0.1:
                world[i][j].terrain = "o"  # it's a rock
            if random.random() < 0.05:
                world[i][j].terrain = "O"  # it's a big rock
            if random.random() < 0.1:
                world[i][j].contents.append("*")  # it's food
            if random.random() < 0.1:
                world[i][j].contents.append("w")  # it's water


def print_local(world, x: int, y: int, size: int):
    for i in range(x - size, x + size + 1):
        for j in range(y - size, y + size + 1):
            if i == x and j == y:
                print(" @ ", end='')  # player should probably be in contents or something
                continue
            print(" " + world[i][j].tostring() + " ", end='')  # don't go off the edge lol
        print("")


def get_input(world, player):
    inp = " "
    def press(key):
        # print(key)
        nonlocal inp
        inp = key.char
    def release(key):
        return False #Returns False to stop the listener
    ##If spacebar is pressed it will stop
    with Listener(
            on_press=press,
            on_release=release) as listener:
        listener.join()

    # print(inp)
    print(inp)
    if inp == 'a':
        player.y -= 1
    if inp == 'e':
        player.y += 1
    if inp == 'o':
        player.x += 1
    if inp == ',':
        player.x -= 1


def update_world(world):
    pass


def run_game():
    world = [ [0]*mapsize for i in range(mapsize)]
    create_world(world)
    player = Player()
    while True:
        # game loop
        print_local(world, player.x, player.y, 3)
        get_input(world, player)
        update_world(world)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_game()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
