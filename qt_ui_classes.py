from random import randint

from PySide2.QtGui import QPixmap, QFont
from PySide2.QtWidgets import QLabel


class DartPositionLabel(QLabel):
    def __init__(self, parent):
        QLabel.__init__(self, parent)

        self.DartPositionLocale = parent
        self.setPixmap(QPixmap(u"Resources/x-circle.svg"))

    @property
    def DartPositionLocale(self):
        # On what Form is the DartPosition located
        return self.__parent

    @DartPositionLocale.setter
    def DartPositionLocale(self, value):
        self.__parent = value

    @property
    def xCor(self):
        # DartPosition's current X-Coordinate
        return self.__xCoordinate

    @xCor.setter
    def xCor(self, value):
        self.__xCoordinate = value

    @property
    def yCor(self):
        # DartPosition's current Y-Coordinate
        return self.__yCoordinate

    @yCor.setter
    def yCor(self, value):
        self.__yCoordinate = value

    @property
    def DartPositionImage(self):
        # DartPosition's current Image
        return self.__DartPositionImage

    @DartPositionImage.setter
    def DartPositionImage(self, value):
        self.__DartPositionImage.setPixmap(QPixmap(value))

    def placeDartPosition(self):
        self.move(self.xCor, self.yCor)
        # I do not have access to your Image so did this instead
        value = "x"
        self.setFont(QFont('Arial', 16))
        self.setStyleSheet("color: magenta")
        self.setText(value)
        #        self.DartPositions[DartPositionId].DartPositionImage = './456.jpg'
        self.show()

    def addDartPositionRandomly(self):
        # Note because you do not currently check to see if
        # the current DartPositionId is not already in your list
        # you may end up just moving them instead of creating
        # them as such I would suggest such a check here and
        # creating another method called moveDartPosition
        self.xCor = randint(1, 450)
        self.yCor = randint(1, 350)
        self.placeDartPosition()

    def addDartPosition(self, x, y):
        self.xCor = x
        self.yCor = y
        self.placeDartPosition()