from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
import random
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

def solution(left_img, right_img):
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(left_img, right_img)
    good_matches = match_keypoint(key_points1, key_points2, descriptor1, descriptor2)
    final_H = ransac(good_matches)

    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    points1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points2 = cv2.perspectiveTransform(points, final_H)
    list_of_points = np.concatenate((points1, points2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(final_H)

    output_img = cv2.warpPerspective(left_img, H_translation, (x_max-x_min, y_max-y_min))
    output_img[(-y_min):rows1+(-y_min), (-x_min):cols1+(-x_min)] = right_img
    result_img = output_img
    return result_img

def get_keypoint(left_img, right_img):
    l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()
    
   # surf = cv2.SURF_create()
   # sift = cv2.SIFT_create()

    key_points1 = sift.detect(l_img, None)
    key_points1, descriptor1 = surf.compute(l_img, key_points1)

    key_points2 = sift.detect(r_img, None)
    key_points2, descriptor2 = surf.compute(r_img, key_points2)
    return key_points1, descriptor1, key_points2, descriptor2

def match_keypoint(key_points1, key_points2, descriptor1, descriptor2):
    i = 0
    k = 2
    all_matches = []
    for d1 in descriptor1:
        dist = []
        j = 0
        for d2 in descriptor2:
            dist.append([i, j, np.linalg.norm(d1 - d2)])
            j = j + 1
        dist.sort(key=lambda x: x[2])
        all_matches.append(dist[0:k])
        i = i + 1

    good_matches = []
    for m, n in all_matches:
        if m[2] < 0.75 * n[2]:
            left_pt = key_points1[m[0]].pt
            right_pt = key_points2[m[1]].pt
            good_matches.append(
                [left_pt[0], left_pt[1], right_pt[0], right_pt[1]])
    return good_matches

def homography(points):
    A = []
    for pt in points:
        x, y = pt[0], pt[1]
        X, Y = pt[2], pt[3]
        A.append([x, y, 1, 0, 0, 0, -1 * X * x, -1 * X * y, -1 * X])
        A.append([0, 0, 0, x, y, 1, -1 * Y * x, -1 * Y * y, -1 * Y])

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    H = (vh[-1, :].reshape(3, 3))
    H = H / H[2, 2]
    return H

def ransac(good_pts):
    best_inliers = []
    final_H = []
    t = 5
    for i in range(5000):
        random_pts = random.choices(good_pts, k = 4)
        H = homography(random_pts)
        inliers = []
        for pt in good_pts:
            p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
            p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
            Hp = np.dot(H, p)
            Hp = Hp / Hp[2]
            dist = np.linalg.norm(p_1 - Hp)

            if dist < t:
                inliers.append(pt)

        if len(inliers) > len(best_inliers):
            best_inliers, final_H = inliers, H
    return final_H

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('file')
        if len(files) < 2:
            return "Please upload at least two images."

        # Save uploaded images
        image_paths = []
        for file in files:
            if file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                image_paths.append(filepath)

        # Read images
        images = [cv2.imread(image_path) for image_path in image_paths]

        # Stitch images (pairwise stitching for simplicity)
        result_img = images[0]
        for i in range(1, len(images)):
            result_img = solution(result_img, images[i])

        # Save the result
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
        cv2.imwrite(result_path, result_img)

        return redirect(url_for('show_result'))

    return render_template('index.html')

@app.route('/result')
def show_result():
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
    return render_template('result.html', result_image=result_path)

if __name__ == '__main__':
    app.run(debug=True)

