import cv2
import numpy as np

INPUT_IMG = 'test_images/19_19_full_top.jpg'
OUTPUT_GRAY = 'output_images/gray.jpg'
OUTPUT_EDGES = 'output_images/edges.jpg'
OUTPUT_FINAL = 'output_images/houghlines3.jpg'

# Rotate an image by an angle in degrees(?)
def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def perp( a ):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2,b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    v = denom.astype(float)
    if v == 0:
        raise ValueError
    return (num / v)*db + b1

# Takes a hough line and returns a tuple of points on the line
# each point is a numpy array of two floats
def points_from_hough_line(line):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    a1 = np.array( [float(x0 + 10000.0*(-b)), float(y0 + 10000.0*(a))] )
    b1 = np.array( [float(x0 - 10000.0*(-b)), float(y0 - 10000.0*(a))] )
    return (a1, b1)

# Given two sets of lines, return all intersection points of the two sets of lines
def intersecting_points_from_lines(lines_a, lines_b):
    i = 0
    points = []
    for line in lines_a:
        a1, b1 = points_from_hough_line(line[0])
        i = i + 1
        for line2 in lines_b:
            a2, b2 = points_from_hough_line(line2[0])
            try:
                points += [seg_intersect(a1, b1, a2, b2)]
            except:
                next

    return points

# Takes a preprocessed image and finds the angle to rotate the image such that most
# lines are aligned horizontally
def calc_image_rotation_angle(edges_img):
    lines = cv2.HoughLines(edges_img,1,np.pi/180,240)

    avg_theta = sum([line[0][1] for line in lines])/len(lines)

    lines_a = [l for l in lines if l[0][1] < avg_theta]
    avg_theta_a = sum([line[0][1] for line in lines_a])/len(lines_a)

    lines_b = [l for l in lines if l[0][1] > avg_theta]
    avg_theta_b = sum([line[0][1] for line in lines_b])/len(lines_b)

    print(avg_theta, avg_theta_a, avg_theta_b)

    # Rotate image
    v = (np.pi - avg_theta_b) * (-180.0 / np.pi)
    return v

# Takes an RGB image and prepares an image for detection
def preprocess_img_for_hough(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(OUTPUT_GRAY, gray)
    # _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray,100,150,apertureSize = 3)
    cv2.imwrite(OUTPUT_EDGES, edges)
    return edges

# Takes a preproccessed image and returns the verticle and horizontal lines of the image as a tuple
def get_lines_from_img(edges_img):
    lines = cv2.HoughLines(edges_img, 1, np.pi/180 ,200)
    # print(lines)
    # draw_lines(img, lines)

    lines_a = [l for l in lines if (l[0][1] < np.pi * 0.05 and l[0][1] >= 0.0) or (l[0][1] > np.pi * 0.95 and l[0][1] <= np.pi)]
    lines_b = [l for l in lines if l[0][1] < np.pi * 0.51 and l[0][1] > np.pi * 0.49]
    if len(lines_a) == 0 or len(lines_b) == 0:
        return (None, None)

    avg_theta_a = sum([line[0][1] for line in lines_a])/len(lines_a)
    avg_theta_b = sum([line[0][1] for line in lines_b])/len(lines_b)

    print(avg_theta_a, avg_theta_b)

    return (lines_a, lines_b)

def get_circles_from_img(edges_img):
    circles = cv2.HoughCircles(edges_img,cv2.HOUGH_GRADIENT,0.9,20,param1=50,param2=30,minRadius=0,maxRadius=45)
    return circles

def draw_lines(img, lines):
    for line in lines:
        a, b = points_from_hough_line(line[0])
        cv2.line(img,(int(a[0]), int(a[1])),(int(b[0]), int(b[1])),(0,0,255),2)

def draw_points(img, points):
    for point in points:
        x, y = point
        if x < 10000 and y < 10000:
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)


def main():
    o_img = cv2.imread(INPUT_IMG)
    edges_img = preprocess_img_for_hough(o_img)
    #v = calc_image_rotation_angle(edges_img)
    # print(v)
    img = o_img # rotate(o_img, v)

    ver_lines, hor_lines = get_lines_from_img(edges_img)

    # if ver_lines:
    #     draw_lines(img, ver_lines)
    # if hor_lines:
    #     draw_lines(img, hor_lines)

    if hor_lines and ver_lines:
        points = intersecting_points_from_lines(ver_lines, hor_lines)
        draw_points(img, points)

    circles = get_circles_from_img(edges_img)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        # cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imwrite(OUTPUT_FINAL, img)
    return 0

if __name__ == "__main__":
    main()