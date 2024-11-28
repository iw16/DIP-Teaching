'''
Mark points:
    00-16 face around
    17-21 right eyebrow
    22-26 left eyebrow
    27-30 mid nose
    31 32 right nose hole
    33 nose peak
    34 35 left nose hole
    36-41 right eye, clockwise
    42-47 left eye, clockwise
    48-53 upper outer lip, clockwise
    54-59 lower outer lip, clockwise
    60-63 upper inner lip, clockwise
    64-67 lower inner lip, clockwise
'''

import enum
import typing as tp

import numpy as np
from numpy.typing import NDArray

class FaceExpression(enum.Enum):
    CLOSE_EYES  = 0
    EXPAND_EYES = 1
    CLOSE_LIPS  = 2
    SMILE_MOUTH = 3
    SLIM_FACE   = 4

    @staticmethod
    def value_of(description: str) -> tp.Self:
        return {
            'close eyes':  FaceExpression.CLOSE_EYES,
            'expand eyes': FaceExpression.EXPAND_EYES,
            'close lips':  FaceExpression.CLOSE_LIPS,
            'smile mouth': FaceExpression.SMILE_MOUTH,
            'slim face':   FaceExpression.SLIM_FACE,
        }[description.lower()]

def close_eyes(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    out_points[[37, 38]] = out_points[[41, 40]]
    out_points[[43, 44]] = out_points[[47, 46]]
    return out_points

def expand_eyes(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    out_points[[37, 38]] += 0.75 * (out_points[[37, 38]] - out_points[[41, 40]])
    out_points[[43, 44]] += 0.75 * (out_points[[43, 44]] - out_points[[47, 46]])
    return out_points

def close_lips(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: NDArray = (out_points[[63, 62, 61]] - out_points[[65, 66, 67]]).mean(axis=1)
    out_points[[65, 66, 67]] = out_points[[63, 62, 61]]
    out_points[55:58, 0] += 1.0 * diff
    return out_points

def smile_mouth(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: float = (out_points[54, 1] - out_points[48, 1]).item()
    out_points[54] += np.array([-diff, diff]) * 0.1
    out_points[48] += diff * -0.1
    out_points[64] += np.array([-diff, diff]) * 0.05
    out_points[60] += diff * -0.05
    return out_points

def slim_face(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: NDArray = out_points[0:8] - out_points[9:17]
    out_points[0:8]  -= 0.05 * diff
    out_points[9:17] += 0.05 * diff
    return out_points

def transform(
    expr_id: FaceExpression,
) -> tp.Callable[[NDArray], NDArray]:
    return {
        FaceExpression.CLOSE_EYES:  close_eyes,
        FaceExpression.EXPAND_EYES: expand_eyes,
        FaceExpression.CLOSE_LIPS:  close_lips,
        FaceExpression.SMILE_MOUTH: smile_mouth,
        FaceExpression.SLIM_FACE:   slim_face,
    }[expr_id]