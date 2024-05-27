import random


class intergers():
    def __init__(self):
        pass

    def generate(length=int):
        code = ""
        for i in range(0, length):
            code += str(random.randint(0, 9))
        return code
