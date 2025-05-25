# %% Imports
import math
import numpy as np
# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *
import OpenGL.GL as GL
import  OpenGL.GLUT as GLUT
import OpenGL.GLU as GLU

# %% Constants
PI = np.pi
orbit = 0.2
uRad = 0.1
uniDensMul = 1.0
diskColor = np.array([0.8, 0.9, 1.0])
gridColor = np.array([0.2, 0.2, 0.5])

def pR(p, a):
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    p = np.dot(np.array([[cos_a, -sin_a], [sin_a, cos_a]]), p)

def iCosShutter(x, t1, t2):
    if x < t1:
        return 0.0
    if x > t2:
        return 1.0
    d = 1.0 / (t2 - t1)
    x -= t1
    return x * d - math.sin(2.0 * PI * x * d) / (2.0 * PI)

def diskAngle(posR, diskR):
    if diskR <= 0.0:
        return 0.0
    if posR <= diskR - orbit:
        return -1.0
    div = (orbit * orbit + posR * posR - diskR * diskR) / (2.0 * posR * orbit)
    if abs(div) > 1.0:
        return 0.0
    return math.acos(div)

def screenDiskTrad(p, frameStart, speed):
    pol = np.array([np.linalg.norm(p), math.atan2(p[1], p[0])])
    pol[1] = (pol[1] + object_rot) % (PI * 2) - PI
    da = diskAngle(pol[0], uRad)
    if da == 0.0:
        return 0.0
    if da == -1.0:
        return 1.0

    shut1 = -0.5 / 60.0
    shut2 = 0.5 / 60.0
    obj1 = (pol[1] - da) / speed
    obj2 = (pol[1] + da) / speed

    l = max(shut1, obj1)
    r = min(shut2, obj2)
    return max(0.0, (r - l) / (shut2 - shut1))

def screenDiskCos(p, frameStart, speed):
    pol = np.array([np.linalg.norm(p), math.atan2(p[1], p[0])])
    pol[1] = (pol[1] + object_rot) % (PI * 2) - PI
    da = diskAngle(pol[0], uRad)
    if da == 0.0:
        return 0.0
    if da == -1.0:
        return 1.0

    shut1 = -1.0 / 60.0
    shut2 = 1.0 / 60.0
    obj1 = (pol[1] - da) / speed
    obj2 = (pol[1] + da) / speed

    return iCosShutter(obj2, shut1, shut2) - iCosShutter(obj1, shut1, shut2)

def mainImage(fragColor, fragCoord, iRes, iTime, video_motion_blur,
              object_speed, object_rot):
    ratio = np.array([iRes[0] / iRes[1], 1.0])
    p = fragCoord / iRes - 0.5
    p *= ratio
    speed = max(0.001, object_speed)

    frameN = math.floor(iTime * 60.0)
    frameAge = iTime * 60.0 - frameN
    frameStart = frameN / 60.0
    diskA = object_rot
    diskC = np.array([-orbit, 0.0])
    pR(diskC, object_rot)
    diskAmt = 0.0
    if video_motion_blur == 0:
        diskAmt = 1.0 - smoothstep(uRad, uRad + 1.0 / iRes[1], np.linalg.norm(p - diskC))
    elif video_motion_blur == 1:
        diskAmt = screenDiskTrad(p, frameN / 60.0, speed)
    elif video_motion_blur == 2:
        diskAmt = screenDiskCos(p, frameN / 60.0, speed)
    gridP = abs(np.mod(p, 0.16) - np.array([0.08]))
    gridAmt = 1.0 - smoothstep(0.6 / iRes[0], 1.6 / iRes[0], gridP[0])
    gridAmt = max(gridAmt, 1.0 - smoothstep(0.6 / iRes[1], 1.6 / iRes[1], gridP[1]))
    col = np.zeros(3)
    col = mix(col, gridColor, gridAmt)
    col = mix(col, diskColor, diskAmt)
    fragColor[:] = np.append(col, 1.0)

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def mix(x, y, a):
    return x * (1.0 - a) + y * a

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    iRes = np.array([800, 600])
    iTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0
    video_motion_blur = 1  # Example value
    object_speed = 10.0  # Example value
    object_rot = 0.0  # Example value

    fragColor = np.zeros(4)
    fragCoord = np.array([0.0, 0.0])  # Example value
    mainImage(fragColor, fragCoord, iRes, iTime, video_motion_blur, object_speed, object_rot)

    # The rest of your OpenGL rendering code goes here...

    glutSwapBuffers()

# def main():
#     glutInit()
#     glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
#     glutInitWindowSize(800, 600)
#     glutCreateWindow('Converted Shader')
#     glutDisplayFunc(display)
#     glutIdleFunc(display)
#     glutMainLoop()

# if __name__ == "__main__":
#     main()
