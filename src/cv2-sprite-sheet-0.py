import cv2

im = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)  # read image
b, g, r, alpha = cv2.split(im.copy())  # split image
g[g == 255] = (
    0  # for the first image where the green channel has 255 on all background pixels
)
imBGR = cv2.merge([b, g, r])  # recombine image in BGR format
imGray = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)  # convert to gray
contours, _ = cv2.findContours(
    imGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)  # contouring
sortedContours = sorted(
    contours, key=cv2.contourArea, reverse=True
)  # sorting, not necessary...
for contourIdx in range(
    0, len(sortedContours) - 1
):  # loop with index for easier saving
    contouredImage = imBGR.copy()  # copy BGR image
    contouredImage = cv2.drawContours(
        contouredImage, sortedContours, contourIdx, (255, 255, 255), -1
    )  # fill contour with white
    extractedImage = cv2.inRange(
        contouredImage, (254, 254, 254), (255, 255, 255)
    )  # extract white from image
    resultImage = cv2.bitwise_and(
        imBGR, imBGR, mask=extractedImage
    )  # AND operator to get only one filled contour at a time
    x, y, w, h = cv2.boundingRect(sortedContours[contourIdx])  # get bounding box
    croppedImage = resultImage[y : y + h, x : x + w]  # crop
    cv2.imwrite("trees_2_contour_" + str(contourIdx) + ".png", croppedImage)  # save
