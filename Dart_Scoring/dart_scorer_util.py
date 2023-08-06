import math
import numpy as np

# radius limits for the different fields on the board
from Dart_Scoring import DartScore

bullsLimit = 10.0
singleBullsLimit = 15.0  # example values
innerTripleLimit = 50.0
outerTripleLimit = 55.0
innerDoubleLimit = 95.0
outerBoardLimit = 100.0

# dictionary that maps the angles to the corresponding fields on the dart board
# every field has a range of 18° and the first field is split in half,
# because the dart board does not start horizontally
listOfFields = {
    (0.0001, 9.000): 6,
    (9.0001, 27.000): 13,
    (27.0001, 45.000): 4,
    (45.0001, 63.000): 18,
    (63.0001, 81.000): 1,
    (81.0001, 99.000): 20,
    (99.0001, 117.000): 5,
    (117.0001, 135.000): 12,
    (135.0001, 153.000): 9,
    (153.0001, 171.000): 14,
    (171.0001, 189.000): 11,
    (189.0001, 207.000): 8,
    (207.0001, 225.000): 16,
    (225.0001, 243.000): 7,
    (243.0001, 261.000): 19,
    (261.0001, 279.000): 3,
    (279.0001, 297.000): 17,
    (297.0001, 315.000): 2,
    (315.0001, 333.000): 15,
    (333.0001, 351.000): 10,
    (351.0001, 360.000): 6,
}

def find_tip_of_dart(pt1, pt2, pt3):
    """
    find the tip of the dart by finding the point with the largest distance to the other two points
    :param pt1:
    :param pt2:
    :param pt3:
    :return:
    """
    dart_point = pt1
    rest_pts = [pt2, pt3]
    dist_1_2 = np.linalg.norm(pt1 - pt2)
    dist_1_3 = np.linalg.norm(pt1 - pt3)
    dist_2_3 = np.linalg.norm(pt2 - pt3)

    if dist_1_2 > dist_1_3 and dist_2_3 > dist_1_3:
        dart_point = pt2
        rest_pts = [pt1, pt3]
    elif dist_1_3 > dist_1_2 and dist_2_3 > dist_1_2:
        dart_point = pt3
        rest_pts = [pt1, pt2]

    dart_point = dart_point.ravel()
    return dart_point, rest_pts


def get_radius_and_angle(centerX, centerY, pointX, pointY):
    """
    get the radius and the angle of the thrown point in relation to the board center
    """
    radius = -1.0  # indicates an error state
    angle = 0.0
    if (centerX >= 0) and (centerY >= 0) and (pointX >= 0) and (pointY >= 0):
        radius = math.sqrt((pointX - centerX) ** 2 + (pointY - centerY) ** 2)

        if pointY < centerY:
            if pointX < centerX:
                angle = math.acos(abs(pointY - centerY) / radius) + np.pi/2
            else:
                angle = math.asin(abs(pointY - centerY) / radius)
        else:
            if pointX > centerX:
                angle = math.acos(abs(pointY - centerY) / radius) + np.pi + np.pi/2
            else:
                angle = math.asin(abs(pointY - centerY) / radius) + np.pi
        angle = angle * (180 / math.pi)  # convert radiant to degrees
    return radius, angle


def evaluate_throw(radius, angle):
    """
    evaluates the value and the multiplier of the field with given radius and angle
    """
    value = -1  # -1 is error state
    multiplier = 1
    # print(bullsLimit, outerBoardLimit)

    if radius >= 0.0:
        if radius < outerBoardLimit:
            # evaluates the value of the field
            for limits in listOfFields:
                if limits[0] <= angle <= limits[1]:
                    value = listOfFields[limits]

            # evaluates the multiplier of the field
            if radius <= bullsLimit:
                return 50, 1  # Bull´s Eye!
            elif radius <= singleBullsLimit:
                return 25, 1  # Single Bull!
            elif radius <= innerTripleLimit:
                multiplier = 1
            elif radius <= outerTripleLimit:
                multiplier = 3
            elif radius <= innerDoubleLimit:
                multiplier = 1
            elif radius <= outerBoardLimit:
                multiplier = 2
        else:
            return 0, 1  # Dart is off the board, no points for player
    else:
        print("Radius is invalid! Point cannot be evaluated!")
        print("Please throw again!")

    return value, multiplier


def get_bottom_point(pt1: np.ndarray, pt2: np.ndarray):
    """
    finds the point on a triangle of given points, where the line from one corner of the triangle to the hypotenuse
    is orthogonal to the hypotenuse
    """
    if pt1.shape == (1, 2) and pt2.shape == (1, 2):
        pt1, pt2 = pt1.ravel(), pt2.ravel()
    if pt1.shape == (2,) and pt2.shape == (2,):
        center = (pt1 + pt2) / 2
        return center.astype(np.int32)
    else:
        print("Points have the wrong shape! Cannot find bottom Point.")


def update_score(dart_score: DartScore.Score, values_of_round, mults_of_round):
    """
    This function evaluates one dart round (3 hits) of a player
    :param dart_score:
    :param values_of_round:
    :param mults_of_round:
    :return:
    """
    points, hit_double = dart_score.calculatePoints(values_of_round[0], mults_of_round[0],
                                                values_of_round[1], mults_of_round[1],
                                                values_of_round[2], mults_of_round[2])
    if points is None or hit_double is None:
        print("Error: Cannot calculate points!")
    else:
        dart_score.pointsScored(points, hit_double)