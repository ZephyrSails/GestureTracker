import thread


class Controller(object):

    def __init__(self):

        self.curPos = None
        self.speed = None
        self.refreshRate = 40

        thread.start_new_thread(self.updateMouse)


    def updateMouse(self):
        while True:
            self.curPos = self.curPos + self.speed

            x, y = map(int, self.curPos)

            move(x, y, 1. / self.refreshRate)

            sleep(self.refreshRate)


    def act(self, pred, pos):

        self.speed =
