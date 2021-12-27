import numpy as np
import cv2
import yaml
import math

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# Parameters
n_row = 9  # chessboard horizontal corners
n_col = 6  # chessboard vertical corners
n_min_img = 2  # min 3 images to calibrate camera
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
corner_accuracy = (11, 11)
result_file = "./calibration.yaml"

# virtual chessboard append on camera
virtual_chess = cv2.imread('imgs/chessboard.png', cv2.IMREAD_UNCHANGED)
scale_percent = 50  # percent of original size
width = int(virtual_chess.shape[1] * scale_percent / 100)
height = int(virtual_chess.shape[0] * scale_percent / 100)
dim = (width, height)
virt_chess = cv2.resize(virtual_chess, dim, interpolation=cv2.INTER_AREA)
# virt_chess[np.where(np.all(virt_chess[..., :3] == 255, -1))] = 0

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(n_row-1,n_col-1,0)
objp = np.zeros((n_row * n_col, 3), np.float32)
objp[:, :2] = np.mgrid[0:n_row, 0:n_col].T.reshape(-1, 2)

# Intialize camera and window
camera = cv2.VideoCapture(0)  # Supposed to be the only camera
if not camera.isOpened():
    print("Camera not found!")
    quit()
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("Calibration")


# Interaction keyboard
def usage():
    print("Press on displayed window : \n")
    print("[space]     : take picture")
    print("[c]         : compute calibration")
    print("[r]         : reset program")
    print("[ESC]    : quit")


usage()
Initialization = True
pos = [0, 0]
while True:
    if Initialization:
        print("Initialize data structures ..")
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        n_img = 0
        Initialization = False
        tot_error = 0

    # Read from camera and display on windows
    ret, img = camera.read()
    img = cv2.flip(img, 1)

    # img = cvzone.overlayPNG(img, virt_chess, pos)

    cv2.imshow("Calibration", img)
    if not ret:
        print("Cannot read camera frame, exit from program!")
        camera.release()
        cv2.destroyAllWindows()
        break

    # Wait for instruction
    k = cv2.waitKey(50)

    # SPACE pressed to take picture
    if k % 256 == 32:
        print("Adding image for calibration...")
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(imgGray, (n_row, n_col), None)

        # If found => add object points, image points (after refining them)
        if not ret:
            print("Cannot found Chessboard corners!")

        else:
            print("Chessboard corners successfully found.")
            objpoints.append(objp)
            n_img += 1
            corners2 = cv2.cornerSubPix(imgGray, corners, corner_accuracy, (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            imgAugmnt = cv2.drawChessboardCorners(img, (n_row, n_col), corners2, ret)
            cv2.imshow('Calibration', imgAugmnt)
            cv2.waitKey(500)

            # "c" = perform calibration
    elif k % 256 == 99:
        if n_img <= n_min_img:
            print("Only ", n_img, " captured, ", " at least ", n_min_img, " images are needed")

        else:
            print("Computing calibration ...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

            if not ret:
                print("Cannot compute calibration!")

            else:
                print("Camera calibration successfully computed")
                # Compute reprojection errors
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    tot_error += error
                    R = cv2.Rodrigues(rvecs[i])[0]
                    T = np.concatenate((R, tvecs[i]), axis=None)  # Transformation chessboard to camera
                    theta = rotationMatrixToEulerAngles(R) * 180 / np.pi
                    print('α β γ tx ty tz :', theta, np.transpose(tvecs[i]), 'Reproj = ', error)
                print("Camera matrix: ", mtx)
                print("Distortion coeffs: ", dist)
                print("Total reprojection error: ", tot_error)
                print("Mean rerojection error: ", np.mean(error))

                # Saving calibration matrix
                print("Saving camera matrix .. in ", result_file)
                data = {"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()}
                with open(result_file, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)

    # ESC = quit
    elif k % 256 == 27:
        print("Escape hit, closing...")
        camera.release()
        cv2.destroyAllWindows()
        break
    # "r" = reset
    elif k % 256 == 114:
        print("Reset program...")
        Initialization = True
