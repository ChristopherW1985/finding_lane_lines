
# Project: Finding lane lines on the road

## 1. Original pictures

First let's have a look at the original pictures. Besides the six test images provided by Udacity I added two images myself. Those are taken from the "challenge" video which is part of the repository as well. Exploring the inital six images one can highlight the following observations:
* all lane lines are straight 
* they always appear in the same general region of the image
* lane lines are painted in high contrast color to the road surface (either white or yellow) 
* light conditions are similar

Things become more challenging with the two images of the "challenge" video.
* slight curvature of lane lines
* light conditions vary because of shadows or bright sunlight
* other cars are much closer
* a part of the car is visible in the bottom of the picture

We need to develop a algorithm which is general as well as robust to detect lane lines reliabely under varying conditions. 


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```


```python
def display_images(images, figsize = (15, 10)):
    "Display images in two columns. Chooses gray-scale colorbar if image is two-dimensional."
    ncols = 2
    nrows = (len(images) // 2) + (len(images) % 2) 
    
    fig = plt.figure(figsize = figsize)
    for i, image in enumerate(images, start = 1):
        ax = fig.add_subplot(nrows, ncols, i)
        plt.imshow(image) if len(image.shape) == 3 else plt.imshow(image, cmap='gray')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
```


```python
# import images
import glob

images = [mpimg.imread(image) for image in glob.glob("test_images/c*.jpg")]

display_images(images)
```


![png](images/output_3_0.png)


## 2. Grayscale transformation

The Cany edge-detection algorithm expects a grayscale image as input. Typically the following formula is applied to conduct grayscale conversion from RGB colorspace
$$Gray = 0.299\cdot R + 0.587\cdot G + 0.114\cdot B.$$
Lane line markings are painted in high contrast colors (white, yellow, red) to the dark road surface. It would be  	advantageous for the application of the Cany algorithms (allows to set higher thresholds) if we choose a transformation which enhances the contrast between a dark pixel and a yellow/white/red pixel. This can be achieved through 
$$Gray = R + G - B.$$

As illustrated in the following toy problem the contrast (i.e. difference in intensities) betweeen white/yellow/red pixels and gray/black pixels is significantly enhanced using $R+G-B$ compared to $0.299\cdot R + 0.587\cdot G + 0.114\cdot B$.


```python
def grayscale(img):
    "Applies the Grayscale transform according to 0.299*R+0.587*G+0.114*B."
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def grayscale2(img):
    "Applies the Grayscale transform according to R+G-B."
    gray = img[:, :, 0].astype('int16') + img[:, :, 1].astype('int16') - img[:, :, 2].astype('int16') 
    gray[gray > 255] = 255
    gray[gray < 0] = 0
    return gray.astype('uint8')


# create picture with white, yellow, red, gray and black regions
imgo = np.zeros_like(np.copy(images[0]))
imgo[:, 0:200]   = [255, 255, 255]
imgo[:, 200:400] = [255, 255, 0]
imgo[:, 400:600] = [255, 0, 0]
imgo[:, 600:800] = [140, 140, 140]
imgo[:, 800:]    = [0, 0, 0]

# grayscale conversion
img_g1 = grayscale(imgo)
img_g2 = grayscale2(imgo)

# intensity values
print('Intensity values of image after grayscale transformation 0.299R+0.587G+0.114B:')
print("white: %d   yellow: %d   red: %d   gray: %d   black: %d" % 
      (img_g1[1, 150], img_g1[1, 250], img_g1[1, 450], img_g1[1, 650], img_g1[1, 850]))
print('Intensity values of image after grayscale transformation R+G-B:')
print("white: %d   yellow: %d   red: %d   gray: %d   black: %d" % 
      (img_g2[1, 150], img_g2[1, 250], img_g2[1, 450], img_g2[1, 650], img_g2[1, 850]))

display_images([img_g1, img_g2], figsize = (15, 3))
```

    Intensity values of image after grayscale transformation 0.299R+0.587G+0.114B:
    white: 255   yellow: 226   red: 76   gray: 140   black: 0
    Intensity values of image after grayscale transformation R+G-B:
    white: 255   yellow: 255   red: 255   gray: 140   black: 0



![png](images/output_5_1.png)



```python
images_gray = np.copy(images)   

images_gray = [grayscale2(image) for image in images_gray]

    
display_images(images_gray, figsize = (15, 15))
```


![png](images/output_6_0.png)


## 3. Canny edge detection

Next we apply the Canny edge detection algorithm to the images. The algorithm consits of four steps
* apply a Gaussian filter in order to smooth the image 
* compute magnitude of the gradient vector for each pixel
* apply a double thresholding to the gradient
    * if gradient is larger than high threshold pixel is regarded as an edge
    * if gradient is smaller than the low threshold pixel is disgarded
    * if gardient is between high and low threshold pixel is marked as a potential edge
* check if "potential edge" pixels are connected to strong edges
    * if yes: regard them as edges
    * if no: disgard them
The algorithm tries to preserve pixels with high gradient values while filtering out pixels with low gradient values.

Since lane line marking are painted with a high contrast color to the road surface (we even enhanced this contrast with our choice of the grayscale conversion) we have a good chance to detect them even with high thresholds. It is important to choose optimize threhsolds such that lane lines are still detect and noisy edges (i.e. edges which are related to line lines) are suppressed as much as possible. The thresholds have to be adjusted empirically. Canny himself recommends a ratio between 1:2 and 1:3 for the low and high threshold. Thresholds low_thresh = 100 and high_thresh = 150 work best for the problem at hand.


```python
def canny(img, low_thresh, high_thresh):
    "Applies the Canny transform."
    return cv2.Canny(img, low_thresh, high_thresh)

blur_gray = [cv2.GaussianBlur(image, (5, 5), 0) for image in images_gray]
images_canny = [canny(image, 50, 150) for image in blur_gray]


display_images(images_canny, figsize = (15, 15))
```


![png](images/output_9_0.png)


## 4. Edge filtering based on edge orientation

Despite the optimization of high and low thresholds many noisy edges are still present in the figures. These edges cannot be filtered out through a proper choice of thresholds alone. We try to filter out the remaining noisy edges by exploting specific knowledge about lane lines. 

Typical lane lines have an angle with respect to the y-axis in the interval [15$^{\circ}$:165$^{\circ}$]. Any line which doesn't fall in this range is most likely a noisy edge. Based on this oberservation we developed an algorithm to filter out edges with an unusal low (or high) angle with respect to the y-axis.  

First the image is divided in a grid of quadratical subimages with a size of 20x20 pixels. Next the Hough transform is applied to each subimage in order to find all lines. The angle of each line with respect to the y-axis and its magnitude is computed afterwards. If the angle is not in the interval [15$^{\circ}$:165$^{\circ}$] the line is assigned to class 'no lane marking' otherwise to class 'lane marking'. Finally the sum of magnituds of all lines in class 'no lane marking' are compared to the sum of magnituds of all lines in class 'lane marking'. If the sum of class 'no lane marking' is larger than the one of class 'lane marking' the subimage is zeroed assuming that no lane line marking is contained in it. Otherwise the subimage is kept as is.


```python
def subimage(image, size):
    """Return list of all sub-images. Each sub-image is represented by a tuple (sx1, sx2, sy1, sy2) such that
    image[sx1:sx2, sy1:sy2] gives the slice of the sub-image.
    
    Args:
    image -- image to be devided
    size  -- size of sub-image (sizexsize pixels). Size has to be a factor of xdim and ydim of image."""
    xdim, ydim = image.shape
    assert (xdim%size==0) and (ydim%size==0)
    nrow = ydim // size
    ncol = xdim // size
    return [(size*x, size*(x+1), size*y, size*(y+1))  for x in range(ncol) for y in range(nrow)]

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    """Return an array of lines found by the Hough transform [[x1, y1, x2, y2], ...].
    
    Args:
    image        -- binary image (e.g. output of the Canny function of opencv)
    rho          -- distance resolution in pixels 
    theta        -- angle resolution in radians
    threshold    -- threshold + 1 is the minimum number of votes a line needs to be returned
    min_line_len -- minimum length of line to be returned
    max_line_gap -- maximum distance of two line segments to be considered as a single line
    """
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
def draw_lines(image, lines, color=[255, 0, 0], thickness=1):
    """Return image with lines drawn on image.
    
    Args:
    image     -- original image
    lines     -- array of lines [[x1, y1, x2, y2], ...] (e.g. output of cv2.HoughLinesP)
    color     -- color of line (default: red)
    thickness -- thickness of line (default: 1)
    """
    # create color binary image
    imgl = np.zeros_like(image, dtype=np.uint8)
    if len(image.shape)==2:
        imgl = np.dstack((imgl, imgl, imgl))   
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(imgl, (x1, y1), (x2, y2), color, thickness) 
    return imgl
```

First we have to look for good values for the Hough transform. Again these parameters are found empirically. The Hough transformation hough_transform(rho, theta, threshold, min_line_len, max_line_gap) takes six parameters

* image: binary image (e.g. output of the Canny function of opencv)
* rho: distance resolution in pixels 
* theta: angle resolution in radiands
* threshold: threshold + 1 is the minimum number of votes a line needs to be returned
* min_line_len: minimum length of a line needed to be returned
* max_line_gap: maximum distance of two line segments to be considered as a single line

Again one needs to find a compromise between filtering out noisy lines and not filtering out real lines. A Hough transformation with parameters rho=1, theta=np.pi/180, threshold=10, min_line_len=5, max_line_gap=5 turned out to be good choice.


```python
img = images_canny[0][310:320, 10:20]

lines = hough_lines(img, rho=1, theta=np.pi/180, threshold=6, min_line_len=5, max_line_gap=5)
img2 = draw_lines(img, lines, color=[255, 0, 0], thickness=1)

display_images([img, img2], figsize = (15, 7))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-9-8933b6a098fa> in <module>()
          2 
          3 lines = hough_lines(img, rho=1, theta=np.pi/180, threshold=6, min_line_len=5, max_line_gap=5)
    ----> 4 img2 = draw_lines(img, lines, color=[255, 0, 0], thickness=1)
          5 
          6 display_images([img, img2], figsize = (15, 7))


    <ipython-input-8-ee2e8d42951a> in draw_lines(image, lines, color, thickness)
         38     if len(image.shape)==2:
         39         imgl = np.dstack((imgl, imgl, imgl))
    ---> 40     for line in lines:
         41         x1, y1, x2, y2 = line[0]
         42         cv2.line(imgl, (x1, y1), (x2, y2), color, thickness)


    TypeError: 'NoneType' object is not iterable


The next chunk of code implements the filtering algorithm based on edge orientation.


```python
def angle(v1, v2):
    "Return angle between vectors v1 and v2 in degrees."
    return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))*180/np.pi

def sum_max_magn(v1, v2):
    "Return sum of vectors v1 and v2 such that the magnitude of the result vector is maximized."
    a = v1 + v2
    b = v1 - v2
    return a if np.linalg.norm(a) >= np.linalg.norm(b) else b

def line_vectors(image, thrAngle, rho, theta, threshold, min_line_len, max_line_gap):
    """Return two vectors: One representing the sum of all lines with an angle in range [thrAngle:180-thrAngle] with 
    respect to the y-axis and a sencond one for all lines in range [0:thrAngle] || [thrAngle:180-thrAngle]. 
    
    Return None for nl and/or ll if no line in the expected angular range was found.
    
    Args:
    image        -- binary image (e.g. output of the Canny function of opencv)
    thrAngle     -- threshold angle (see return values)
    rho          -- distance resolution in pixels 
    theta        -- angle resolution in radians
    threshold    -- threshold + 1 is the minimum number of votes a line needs to be returned
    min_line_len -- minimum length of line to be returned
    max_line_gap -- maximum distance of two line segments to be considered as a single line    
    Return values:
    nl           -- sum of all lines with an angle in range [0:thrAngle] || [thrAngle:180-thrAngle]
    ll           -- sum of all lines with an angle in range [thrAngle:180-thrAngle]
    """
    assert image.shape != (0, 0)
    lines = hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)  
    nl = np.array([0, 0])
    ll = np.array([0, 0])
    if lines is None: # hough returns None if no lines were found (i.e. don't change original figure)
        return None, None
    ex = np.array([0, 1])
    for line in lines:
        x1, y1, x2, y2 = line[0] 
        el = np.array([y1, x1]) - np.array([y2, x2]) # numpy and opencv interchange rows and columns
        if (angle(el, ex) <= thrAngle) or (angle(el, ex) >= 180-thrAngle):
            nl = sum_max_magn(nl, el) 
        else:
            ll = sum_max_magn(ll, el) 
    if np.array_equal(nl, np.array([0, 0])):
        nl = None
    if np.array_equal(ll, np.array([0, 0])):
        ll = None    
    return nl, ll


def setup_grid(image, size, thrAngle, rho, theta, threshold, min_line_len, max_line_gap):
    """Return grid of sub-images with boolean value indicating if subimage contains potential lane line.

    Args:
    image        -- binary image (e.g. output of the Canny function of opencv)
    size         -- size of sub-images
    thrAngle     -- threshold angle (see return values)
    rho          -- distance resolution in pixels 
    theta        -- angle resolution in radians
    threshold    -- threshold + 1 is the minimum number of votes a line needs to be returned
    min_line_len -- minimum length of line to be returned
    max_line_gap -- maximum distance of two line segments to be considered as a single line    
    Return values:
    grid         -- Dictonary with keys indicating sub-image and boolean values indicating if sub-image contains lane line.
                    {(x1, x2, y1, y2) : True, (x1, x2, y1, y2): False, ...}
    """
    subs = subimage(image, size)
    grid = {}
    for sub in subs:
        x1, x2, y1, y2  = sub
        nl, ll = line_vectors(image[x1:x2, y1:y2], thrAngle, rho, theta, threshold, min_line_len, max_line_gap)
        if ll is None:
            grid[(x1, x2, y1, y2)] = False
        elif nl is None:
            grid[(x1, x2, y1, y2)] = True             
        elif np.linalg.norm(ll) >= np.linalg.norm(nl):
            grid[(x1, x2, y1, y2)] = True            
        elif np.linalg.norm(ll) < np.linalg.norm(nl):
            grid[(x1, x2, y1, y2)] = False 
    return grid    

def heatmap(image, grid):
    """Return image where sub-images with lane=True are highlighted in white (useful to visualize result of setup_grid)."""
    img = np.zeros_like(image)
    for coo, lane in grid.items():
        x1, x2, y1, y2 = coo
        if lane:
            img[x1:x2, y1:y2] = 255 
    return img

def filter_edge_orientation(image, grid):
    """Zeros all sub-images of image (as specified by grid) with lane=False
    
    Args:
    image        -- binary image (e.g. output of the Canny function of opencv)
    grid         -- output of setup_grid
    """
    for coo, lane in grid.items():
        x1, x2, y1, y2 = coo
        if not lane:
            image[x1:x2, y1:y2] = 0
```


```python
images_filter_eo = np.copy(images_canny)

grids = [setup_grid(image, size = 10, thrAngle = 15, rho = 1, theta = np.pi/180, threshold = 7, 
                    min_line_len = 5, max_line_gap = 5) 
        for image in images_canny]
[filter_edge_orientation(image, grid) for image, grid in zip(images_filter_eo, grids)] 
heatmaps = [heatmap(image, grid) for image, grid in zip(images_canny, grids)]

#display_images(heatmaps[0:4], figsize = (15, 3))
display_images(images_filter_eo, figsize = (15, 15))
#display_images(images_canny, figsize = (15, 15))
```


![png](images/output_17_0.png)


The filtering algorithm based on edge orientation works very well. A good portion of noisy edges has been filtered out. However, sometimes a sub-image is zeroed although it contained a piece of the actual lane line marking. In order to fix this issue we apply a mask which looks at the nei


```python
def reconstruct_lane(grid, size):
    """Reconstruct grid: If neighbouring top left, top right or top sub-images contains line lane
    assume sub-image contains line lane as well."""
    agrid = grid.copy() # copy dict
    for coo, lane in grid.items():
        for neigh in neighbours(coo, size):
            if neigh in grid.keys() and grid[neigh]==True:
                agrid[coo] = True
    return agrid        
        
def neighbours(coo, size):
    "Return coordinates of the neighbouring sub-images on the top left, topright and top."
    x1, x2, y1, y2 = coo
    top = (x1 + size, x2 + size, y1, y2)
    topleft = (x1 + size, x2 + size, y1 - size, y2 - size)   
    topright = (x1 + size, x2 + size, y1 + size, y2 + size)
    return (topleft, top, topright)
```


```python
images_filter_eo = np.copy(images_canny)

grids = [reconstruct_lane(setup_grid(image, size = 10, thrAngle = 15, rho = 1, theta = np.pi/180, 
                                     threshold = 7, min_line_len = 5, max_line_gap = 5), size = 10) 
        for image in images_canny]
[filter_edge_orientation(image, grid) for image, grid in zip(images_filter_eo, grids)] 
heatmaps = [heatmap(image, grid) for image, grid in zip(images_canny, grids)]


#display_images(heatmaps[0:4], figsize = (15, 3))
display_images(images_filter_eo, figsize = (15, 15))
#display_images(images_canny[4:6], figsize = (15, 3))
```


![png](images/output_20_0.png)


On the expense of some additional noisy edges the lane lines have been reconstructed.

## 5. Edge filtering based on gradients

Although the previous algorithm filtered out many noisy edges there are still some present. We exploit two additional feature of lane line markings to distinguish them from noisy edges:
* lane line marking consists of two parallel lines (after application of the Cany edge detector)
* the direction of the gradient vector at the location of the lane lines falls in four ...


```python
def phase(image):
    """Return angle of gradient vector at location of each pixel. Expects a gray-scale image.
    Angles are assigned to four quadrantes represented by integers between 1 and 4. 
    1: 270 <= angle <= 360
    2: 180 <= angle <  270
    3: 90  <= angle <  180
    4: 0   <= angle <   90
    """
    gradientX = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    gradientY = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    angle = cv2.phase(gradientX, gradientY, angleInDegrees=True) 
    angle[(0 <= angle[:, :]) & (angle[:, :] < 90)]    = 4 
    angle[(90 <= angle[:, :]) & (angle[:, :] < 180)]  = 3
    angle[(180 <= angle[:, :]) & (angle[:, :] < 270)] = 2
    angle[(270 <= angle[:, :]) & (angle[:, :] <= 360)] = 1
    return angle    
    
def filter_pos_y(iq, fq, size, angle):
    """Check if pixel 'fq' is within 'size' pixels of pixel 'iq'. Otherwise zero pixel 'iq'.
    
    Search is done in increasing y-direction.    
    
    Args:
    iq -- quadrant of starting pixel
    fq -- quadrant we are looking for
    size -- size of windows in which we look for fq
    angle -- numpy array of angle values
    """
    _, ymax = angle.shape
    for pix in np.argwhere(angle == iq):
        flag = False
        for y in np.arange(pix[1], pix[1] + size, 1) :
            if y < ymax and angle[pix[0], y] == fq:
                flag = True
        if not flag:
            angle[pix[0], pix[1]] = 0

def filter_neg_y(iq, fq, size, angle):
    """Check if pixel 'fq' is within 'size' pixels of pixel 'iq'. Otherwise zero pixel 'iq'.
    
    Search is done in decreasing y-direction.
    
    Args:
    iq -- quadrant of starting pixel
    fq -- quadrant we are looking for
    size -- size of windows in which we look for fq
    angle -- numpy array of angle values
    """
    for pix in np.argwhere(angle == iq):
        flag = False
        for y in np.arange(pix[1], pix[1] - size, -1):
            if y >= 0 and angle[pix[0], y] == fq:
                flag = True
        if not flag:
            angle[pix[0], pix[1]] = 0
            
def filter_region(angle, left, right):
    """Divide image in two halves and filter out all pixel which are assigned to angular ranges 
    specified in 'right' but appear on the left of the figure (and viceversa).
    
    Args:
    angle  -- numpy array of angle values
    left   -- list of angular ranges expected to be in the left half of the figure 
    right  -- list of angular ranges expected to be in the right half of the figure 
    """
    xmax, ymax = angle.shape
    center = ymax//2
    for c in left:
        for pix in np.argwhere(angle == c):
            if pix[1] > center:
                angle[pix[0], pix[1]]=0
    for c in right:
        for pix in np.argwhere(angle == c):
            if pix[1] < center:
                angle[pix[0], pix[1]]=0         
    
    

def filter_gradient(imageg, image, width):
    """Filter out noisy edges of image based on assumptions of lane line markig width and direction of the gradient.
    
    Args:
    imageg       -- grayscale image
    image        -- binary image (e.g. output of the Canny function of opencv)
    width        -- maximum width of lane line         
    """
    angle = phase(imageg)
    mask = (image[:, :] == 255)
    angle[~mask] = 0
    filter_pos_y(1, 3, width, angle)  
    filter_pos_y(4, 2, width, angle)  
    filter_neg_y(3, 1, width, angle)  
    filter_neg_y(2, 4, width, angle) 
    #filter_region(angle, right = [1, 3], left = [2, 4])
    #angle[angle != 0] = 255
    return angle.astype('uint8')
    
```


```python
images_filter_g = [filter_gradient(p1, p2, 30)  for p1, p2 in zip(images_gray, images_filter_eo)]         
#images_filter_g = [filter_gradient(p1, p2, 30)  for p1, p2 in zip(images_gray, images_canny)]   
    
display_images(images_filter_g, figsize = (15, 15))
```


![png](images/output_25_0.png)


## 6. Regions selection

The region selection seems to be straight forward because all lane lines appear in the same general region of the pictures. The choice of the region has to take into account that a part of the car is visible at the bottom of some pictures. A polygon with four sides is a reasonable choice. It shouldn't be too restrictive. Objects which fall in this region but are not related to lane lines can still be suppressed through an appropriate filtering algorithms and a tuning of Cany and Hough algorithms.

Each original picture has a size of 540x960. I chose the region between 330 and 490 in x-direction and the full length in y-direction.


```python
def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    bottom_left = (0.10, 0.97)
    top_left = (0.41, 0.64)
    top_right = (0.59, 0.65)
    bottom_right = (0.96, 0.96)
    xdim, ydim  = img.shape
    bl_y, bl_x = bottom_left
    tl_y, tl_x = top_left
    tr_y, tr_x = top_right
    br_y, br_x = bottom_right
    v1 = int(bl_y*ydim), int(bl_x*xdim)
    v2 = int(tl_y*ydim), int(tl_x*xdim)
    v3 = int(tr_y*ydim), int(tr_x*xdim)
    v4 = int(br_y*ydim), int(br_x*xdim)
    vertices = np.array([[v1, v2, v3, v4]])   
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def region_of_interest2(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    v1 = (96, 523)
    v2 = (394, 349)  
    v3 = (566, 349) 
    v4 = (920, 523)
    vertices = np.array([[v1, v2, v3, v4]])   
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image




images_reg = [region_of_interest(image) for image in images_filter_g]

display_images(images_reg, figsize = (15, 15))

```


![png](images/output_27_0.png)



```python
images_reg = [region_of_interest2(image) for image in images_filter_g]

display_images(images_reg, figsize = (15, 15))
```


![png](images/output_28_0.png)



```python
def otsu_filter(img, gimg):
    """Otsu
    """
    imgc = np.copy(img)
    blur = cv2.GaussianBlur(gimg, (5,5), 0)
    ret3, th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = (th3 == 0)
    imgc[mask] = 0                          
    return imgc 

def otsu_filter2(img):
    """Otsu
    """
    blur = cv2.GaussianBlur(img, (5,5), 0)
    ret3, th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)                          
    return th3  
                              
images_otsu = [otsu_filter(p1, p2) for p1, p2 in zip(images_reg, images_gray)]
#images_otsu = [otsu_filter2(p1) for p1 in images_gray]


display_images(images_otsu, figsize = (15,15))
#display_images(images_filter_g[2:4], figsize = (15,15))
```


![png](images/output_29_0.png)


## 7. Finding lane lines


```python
def average_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    avg = {}
    for classes in ((1, 3), (2, 4)):
        c1, c2 = classes
        img = np.copy(image) 
        mask = (img[:, :] == c1) | (img[:, :] == c2)
        img[~mask] = 0
        lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap) 
        if lines is None:
            continue
        
        slope = []
        intercept = []
        length = []
        for line in lines:
            y1, x1, y2, x2 = line[0]  # interchanged x, y coordiantes
            if x1 == x2: continue
            slope.append((y1 - y2)/(x1 - x2))
            intercept.append(y1 - slope[-1]*x1 - slope[-1])
            length.append((x1-x2)**2 + (y1-y2)**2)
            
        # use length of lines as weights
        if slope and intercept and length:
            avgSlope = np.dot(slope, length) / sum(length)  
            avgIntercept = np.dot(intercept, length) / sum(length)      
            if classes == (1, 3):
                avg['right'] = (avgSlope, avgIntercept)
            else:
                avg['left'] = (avgSlope, avgIntercept)                
    return avg
    
def pixel_points(image, avg, upper=1):
    assert 0 < upper <= 1
    if len(image.shape)==2:
        xmax, ymax = image.shape
    else:
        xmax, ymax, _ = image.shape
    xmin = int(upper*xmax)
    points = {'left' : None, 'right' : None}
    for lane, line in avg.items():
        slope, intercept = line
        y1 = int(slope*xmax + intercept)
        y2 = int(slope*xmin + intercept)
        if lane == 'right':
            points['right'] = (y1, xmax, y2, xmin)
            #points['right'] = np.array([[[y1, xmax, y2, xmin]]])
        else:
            points['left'] = (y1, xmax, y2, xmin)
            #points['left'] = np.array([[[y1, xmax, y2, xmin]]])
    return points

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
     
def draw_lines(image, lines, color=[255, 0, 0], thickness=1):
    """Return image with lines drawn on image.
    
    Args:
    image     -- original image
    lines     -- array of lines [[x1, y1, x2, y2], ...] (e.g. output of cv2.HoughLinesP)
    color     -- color of line (default: red)
    thickness -- thickness of line (default: 1)
    """
    # create color binary image
    imgl = np.zeros_like(image, dtype=np.uint8)
    if len(image.shape)==2:
        imgl = np.dstack((imgl, imgl, imgl))   
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(imgl, (x1, y1), (x2, y2), color, thickness) 
    return imgl    


def draw_lane_lines(image, line, color=[255, 0, 0], thickness=20):
    imgl = np.zeros_like(image)
    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(imgl, (x1, y1), (x2, y2),  color, thickness)
    return cv2.addWeighted(image, 1.0, imgl, 0.95, 0.0)
    
    
```


```python
import collections

class FindLaneLines:
    def __init__(self):
        self.left = collections.deque(maxlen=10)
        self.right = collections.deque(maxlen=10)
        self.left_avg = (0, 0, 0, 0)
        self.right_avg = (0, 0, 0, 0)
        self.config = {
           # region
           'bottom_left' : (0.10, 0.97),
           'top_left' : (0.41, 0.64),     
           'top_right' : (0.59, 0.65),  
           'bottom_right' : (0.96, 0.96),
           # canny
           'low' : 50,
           'high' : 150,
           # filter noisy edges
           'size' : 10,
           'angle' : 15,
           'rho1' : 1,
           'theta1' : np.pi/180,
           'thresh1' : 7,
           'min_line_len1' : 5,
           'max_line_gap1' : 5,
           'width' : 30,
           # hough lines
           'rho2' : 1,
           'theta2' : np.pi/180,
           'thresh2' : 20,
           'min_line_len2' : 20,
           'max_line_gap2' : 300        
         }
                
        
    def process_image(self, image):
        # convert image to grayscale and search edges
        img_gray = grayscale2(image)
        #blur_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_canny = canny(img_gray, self.config['low'], self.config['high'])
    
        # filter noisy edges
        grid = setup_grid(img_canny, 
                      size = self.config['size'], 
                      thrAngle = self.config['angle'], 
                      rho = self.config['rho1'], 
                      theta = self.config['theta1'], 
                      threshold = self.config['thresh1'], 
                      min_line_len = self.config['min_line_len1'], 
                      max_line_gap = self.config['max_line_gap1'])
        grid_re = reconstruct_lane(grid, self.config['size'])
        filter_edge_orientation(img_canny, grid_re)   
        images_filter_g = filter_gradient(img_gray, img_canny, self.config['width'])   
    
        # select region
        images_reg = region_of_interest(images_filter_g) 
    
        # apply otsu filter
        images_otsu = otsu_filter(images_reg, img_gray)
    
        # find lane lines
        avg = average_lines(images_otsu, 
                            rho = self.config['rho2'], 
                            theta = self.config['theta2'], 
                            threshold = self.config['thresh2'], 
                            min_line_len = self.config['min_line_len2'],
                            max_line_gap = self.config['max_line_gap2'])
        
        points =  pixel_points(image, avg, 0.62)  
        
            
        def aver_line(lines, line, avg, perc):
            if (line is not None):
                if (len(lines) < 10) or (outlier(avg, line, perc)):
                    lines.append(line) 
                
            if len(lines)>0:
                sumx1 = sumy1 = sumx2 = sumy2 = 0
                for line  in lines:
                    y1, x1, y2, x2 = line
                    sumy1 += y1
                    sumx1 += x1
                    sumy2 += y2
                    sumx2 += x2              
                line = (sumy1/len(lines), sumx1/len(lines), sumy2/len(lines), sumx2/len(lines))
                line = tuple(int(l) for l in line)
            return line
        
        def outlier(avg, line, perc):
            cond1 = line[0]*(1-perc) <= avg[0] <= line[0]*(1+perc) 
            cond2 = line[1]*(1-perc) <= avg[1] <= line[1]*(1+perc) 
            cond3 = line[2]*(1-perc) <= avg[2] <= line[2]*(1+perc) 
            cond4 = line[3]*(1-perc) <= avg[3] <= line[3]*(1+perc) 
            return cond1 and cond2 and cond3 and cond4
                    
        self.left_avg = aver_line(self.left, points['left'], self.left_avg, 0.25)
        self.right_avg = aver_line(self.right, points['right'], self.right_avg, 0.25)
            
        image = draw_lane_lines(image, self.left_avg, color=[255, 0, 0], thickness=10)
        image = draw_lane_lines(image, self.right_avg, color=[255, 0, 0], thickness=10)
        return image

    
    
pic = []
for img in images:
    detector = FindLaneLines()
    pic.append(detector.process_image(img))
    
display_images(pic, figsize = (15, 15))
        
```


![png](images/output_32_0.png)


## Test on Videos


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os


def process_video(video_input, video_output):
    detector = FindLaneLines()

    clip = VideoFileClip(os.path.join('test_videos', video_input))
    processed = clip.fl_image(detector.process_image)
    processed.write_videofile(os.path.join('test_videos_output', video_output), audio=False)
    
  
```


```python
%time process_video('solidWhiteRight.mp4', 'solidWhiteRight.mp4') 

%time process_video('solidYellowLeft.mp4', 'solidYellowLeft.mp4') 

%time process_video('challenge.mp4', 'challenge.mp4') 
```

    [MoviePy] >>>> Building video test_videos_output/solidWhiteRight.mp4
    [MoviePy] Writing video test_videos_output/solidWhiteRight.mp4


    
      0%|          | 0/222 [00:00<?, ?it/s][A
      0%|          | 1/222 [00:00<02:30,  1.47it/s][A
      1%|          | 2/222 [00:01<02:29,  1.47it/s][A
      1%|▏         | 3/222 [00:01<02:25,  1.51it/s][A
      2%|▏         | 4/222 [00:02<02:20,  1.55it/s][A
      2%|▏         | 5/222 [00:03<02:14,  1.61it/s][A
      3%|▎         | 6/222 [00:03<02:08,  1.68it/s][A
      3%|▎         | 7/222 [00:04<02:04,  1.72it/s][A
      4%|▎         | 8/222 [00:04<02:02,  1.75it/s][A
      4%|▍         | 9/222 [00:05<01:58,  1.80it/s][A
      5%|▍         | 10/222 [00:05<01:55,  1.83it/s][A
      5%|▍         | 11/222 [00:06<01:53,  1.85it/s][A
      5%|▌         | 12/222 [00:06<01:53,  1.85it/s][A
      6%|▌         | 13/222 [00:07<01:51,  1.87it/s][A
      6%|▋         | 14/222 [00:07<01:49,  1.90it/s][A
      7%|▋         | 15/222 [00:08<01:47,  1.92it/s][A
      7%|▋         | 16/222 [00:08<01:47,  1.91it/s][A
      8%|▊         | 17/222 [00:09<01:46,  1.92it/s][A
      8%|▊         | 18/222 [00:09<01:44,  1.95it/s][A
      9%|▊         | 19/222 [00:10<01:42,  1.97it/s][A
      9%|▉         | 20/222 [00:10<01:41,  1.99it/s][A
      9%|▉         | 21/222 [00:11<01:40,  1.99it/s][A
     10%|▉         | 22/222 [00:11<01:38,  2.03it/s][A
     10%|█         | 23/222 [00:12<01:38,  2.02it/s][A
     11%|█         | 24/222 [00:12<01:38,  2.02it/s][A
     11%|█▏        | 25/222 [00:13<01:39,  1.98it/s][A
     12%|█▏        | 26/222 [00:13<01:38,  1.99it/s][A
     12%|█▏        | 27/222 [00:14<01:38,  1.99it/s][A
     13%|█▎        | 28/222 [00:14<01:37,  2.00it/s][A
     13%|█▎        | 29/222 [00:15<01:36,  1.99it/s][A
     14%|█▎        | 30/222 [00:15<01:36,  1.99it/s][A
     14%|█▍        | 31/222 [00:16<01:36,  1.98it/s][A
     14%|█▍        | 32/222 [00:16<01:36,  1.96it/s][A
     15%|█▍        | 33/222 [00:17<01:37,  1.94it/s][A
     15%|█▌        | 34/222 [00:17<01:34,  1.99it/s][A
     16%|█▌        | 35/222 [00:18<01:34,  1.98it/s][A
     16%|█▌        | 36/222 [00:19<01:34,  1.97it/s][A
     17%|█▋        | 37/222 [00:19<01:33,  1.97it/s][A
     17%|█▋        | 38/222 [00:20<01:33,  1.97it/s][A
     18%|█▊        | 39/222 [00:20<01:33,  1.96it/s][A
     18%|█▊        | 40/222 [00:21<01:30,  2.00it/s][A
     18%|█▊        | 41/222 [00:21<01:30,  2.01it/s][A
     19%|█▉        | 42/222 [00:21<01:28,  2.04it/s][A
     19%|█▉        | 43/222 [00:22<01:28,  2.01it/s][A
     20%|█▉        | 44/222 [00:22<01:28,  2.00it/s][A
     20%|██        | 45/222 [00:23<01:26,  2.05it/s][A
     21%|██        | 46/222 [00:23<01:25,  2.05it/s][A
     21%|██        | 47/222 [00:24<01:25,  2.05it/s][A
     22%|██▏       | 48/222 [00:24<01:24,  2.05it/s][A
     22%|██▏       | 49/222 [00:25<01:25,  2.02it/s][A
     23%|██▎       | 50/222 [00:25<01:26,  1.99it/s][A
     23%|██▎       | 51/222 [00:26<01:25,  2.01it/s][A
     23%|██▎       | 52/222 [00:26<01:24,  2.01it/s][A
     24%|██▍       | 53/222 [00:27<01:23,  2.03it/s][A
     24%|██▍       | 54/222 [00:27<01:23,  2.01it/s][A
     25%|██▍       | 55/222 [00:28<01:22,  2.03it/s][A
     25%|██▌       | 56/222 [00:28<01:21,  2.04it/s][A
     26%|██▌       | 57/222 [00:29<01:22,  2.01it/s][A
     26%|██▌       | 58/222 [00:29<01:21,  2.02it/s][A
     27%|██▋       | 59/222 [00:30<01:23,  1.95it/s][A
     27%|██▋       | 60/222 [00:30<01:21,  1.99it/s][A
     27%|██▋       | 61/222 [00:31<01:21,  1.98it/s][A
     28%|██▊       | 62/222 [00:31<01:21,  1.96it/s][A
     28%|██▊       | 63/222 [00:32<01:22,  1.93it/s][A
     29%|██▉       | 64/222 [00:32<01:20,  1.97it/s][A
     29%|██▉       | 65/222 [00:33<01:19,  1.98it/s][A
     30%|██▉       | 66/222 [00:33<01:18,  1.99it/s][A
     30%|███       | 67/222 [00:34<01:17,  2.01it/s][A
     31%|███       | 68/222 [00:34<01:16,  2.01it/s][A
     31%|███       | 69/222 [00:35<01:17,  1.97it/s][A
     32%|███▏      | 70/222 [00:35<01:16,  1.98it/s][A
     32%|███▏      | 71/222 [00:36<01:16,  1.97it/s][A
     32%|███▏      | 72/222 [00:36<01:15,  1.99it/s][A
     33%|███▎      | 73/222 [00:37<01:14,  1.99it/s][A
     33%|███▎      | 74/222 [00:37<01:14,  2.00it/s][A
     34%|███▍      | 75/222 [00:38<01:14,  1.96it/s][A
     34%|███▍      | 76/222 [00:39<01:15,  1.94it/s][A
     35%|███▍      | 77/222 [00:39<01:13,  1.97it/s][A
     35%|███▌      | 78/222 [00:40<01:13,  1.97it/s][A
     36%|███▌      | 79/222 [00:40<01:13,  1.96it/s][A
     36%|███▌      | 80/222 [00:41<01:12,  1.95it/s][A
     36%|███▋      | 81/222 [00:41<01:12,  1.95it/s][A
     37%|███▋      | 82/222 [00:42<01:11,  1.95it/s][A
     37%|███▋      | 83/222 [00:42<01:10,  1.97it/s][A
     38%|███▊      | 84/222 [00:43<01:09,  1.98it/s][A
     38%|███▊      | 85/222 [00:43<01:10,  1.95it/s][A
     39%|███▊      | 86/222 [00:44<01:08,  1.98it/s][A
     39%|███▉      | 87/222 [00:44<01:08,  1.98it/s][A
     40%|███▉      | 88/222 [00:45<01:07,  1.97it/s][A
     40%|████      | 89/222 [00:45<01:06,  1.99it/s][A
     41%|████      | 90/222 [00:46<01:06,  1.99it/s][A
     41%|████      | 91/222 [00:46<01:06,  1.97it/s][A
     41%|████▏     | 92/222 [00:47<01:05,  1.99it/s][A
     42%|████▏     | 93/222 [00:47<01:05,  1.97it/s][A
     42%|████▏     | 94/222 [00:48<01:05,  1.96it/s][A
     43%|████▎     | 95/222 [00:48<01:04,  1.97it/s][A
     43%|████▎     | 96/222 [00:49<01:03,  1.98it/s][A
     44%|████▎     | 97/222 [00:49<01:03,  1.97it/s][A
     44%|████▍     | 98/222 [00:50<01:02,  1.98it/s][A
     45%|████▍     | 99/222 [00:50<01:01,  1.99it/s][A
     45%|████▌     | 100/222 [00:51<01:01,  1.99it/s][A
     45%|████▌     | 101/222 [00:51<00:59,  2.02it/s][A
     46%|████▌     | 102/222 [00:52<00:59,  2.02it/s][A
     46%|████▋     | 103/222 [00:52<00:59,  2.02it/s][A
     47%|████▋     | 104/222 [00:53<00:58,  2.01it/s][A
     47%|████▋     | 105/222 [00:53<00:59,  1.98it/s][A
     48%|████▊     | 106/222 [00:54<00:58,  1.99it/s][A
     48%|████▊     | 107/222 [00:54<00:57,  2.00it/s][A
     49%|████▊     | 108/222 [00:55<00:56,  2.02it/s][A
     49%|████▉     | 109/222 [00:55<00:55,  2.04it/s][A
     50%|████▉     | 110/222 [00:56<00:54,  2.05it/s][A
     50%|█████     | 111/222 [00:56<00:53,  2.07it/s][A
     50%|█████     | 112/222 [00:57<00:52,  2.10it/s][A
     51%|█████     | 113/222 [00:57<00:51,  2.12it/s][A
     51%|█████▏    | 114/222 [00:57<00:51,  2.10it/s][A
     52%|█████▏    | 115/222 [00:58<00:51,  2.07it/s][A
     52%|█████▏    | 116/222 [00:58<00:51,  2.07it/s][A
     53%|█████▎    | 117/222 [00:59<00:50,  2.10it/s][A
     53%|█████▎    | 118/222 [00:59<00:49,  2.11it/s][A
     54%|█████▎    | 119/222 [01:00<00:48,  2.11it/s][A
     54%|█████▍    | 120/222 [01:00<00:48,  2.10it/s][A
     55%|█████▍    | 121/222 [01:01<00:48,  2.06it/s][A
     55%|█████▍    | 122/222 [01:01<00:49,  2.01it/s][A
     55%|█████▌    | 123/222 [01:02<00:49,  2.01it/s][A
     56%|█████▌    | 124/222 [01:02<00:49,  1.98it/s][A
     56%|█████▋    | 125/222 [01:03<00:48,  2.00it/s][A
     57%|█████▋    | 126/222 [01:03<00:48,  1.98it/s][A
     57%|█████▋    | 127/222 [01:04<00:47,  1.98it/s][A
     58%|█████▊    | 128/222 [01:04<00:46,  2.01it/s][A
     58%|█████▊    | 129/222 [01:05<00:46,  1.99it/s][A
     59%|█████▊    | 130/222 [01:05<00:46,  2.00it/s][A
     59%|█████▉    | 131/222 [01:06<00:46,  1.95it/s][A
     59%|█████▉    | 132/222 [01:06<00:46,  1.95it/s][A
     60%|█████▉    | 133/222 [01:07<00:46,  1.90it/s][A
     60%|██████    | 134/222 [01:08<00:45,  1.92it/s][A
     61%|██████    | 135/222 [01:08<00:45,  1.91it/s][A
     61%|██████▏   | 136/222 [01:09<00:44,  1.92it/s][A
     62%|██████▏   | 137/222 [01:09<00:45,  1.88it/s][A
     62%|██████▏   | 138/222 [01:10<00:44,  1.88it/s][A
     63%|██████▎   | 139/222 [01:10<00:43,  1.89it/s][A
     63%|██████▎   | 140/222 [01:11<00:43,  1.87it/s][A
     64%|██████▎   | 141/222 [01:11<00:42,  1.89it/s][A
     64%|██████▍   | 142/222 [01:12<00:42,  1.90it/s][A
     64%|██████▍   | 143/222 [01:12<00:41,  1.91it/s][A
     65%|██████▍   | 144/222 [01:13<00:40,  1.91it/s][A
     65%|██████▌   | 145/222 [01:13<00:40,  1.91it/s][A
     66%|██████▌   | 146/222 [01:14<00:39,  1.93it/s][A
     66%|██████▌   | 147/222 [01:14<00:38,  1.93it/s][A
     67%|██████▋   | 148/222 [01:15<00:38,  1.94it/s][A
     67%|██████▋   | 149/222 [01:15<00:37,  1.94it/s][A
     68%|██████▊   | 150/222 [01:16<00:36,  1.96it/s][A
     68%|██████▊   | 151/222 [01:16<00:36,  1.92it/s][A
     68%|██████▊   | 152/222 [01:17<00:36,  1.90it/s][A
     69%|██████▉   | 153/222 [01:18<00:36,  1.89it/s][A
     69%|██████▉   | 154/222 [01:18<00:35,  1.90it/s][A
     70%|██████▉   | 155/222 [01:19<00:35,  1.87it/s][A
     70%|███████   | 156/222 [01:19<00:35,  1.84it/s][A
     71%|███████   | 157/222 [01:20<00:36,  1.80it/s][A
     71%|███████   | 158/222 [01:20<00:35,  1.81it/s][A
     72%|███████▏  | 159/222 [01:21<00:34,  1.81it/s][A
     72%|███████▏  | 160/222 [01:21<00:34,  1.80it/s][A
     73%|███████▎  | 161/222 [01:22<00:33,  1.81it/s][A
     73%|███████▎  | 162/222 [01:22<00:33,  1.82it/s][A
     73%|███████▎  | 163/222 [01:23<00:32,  1.83it/s][A
     74%|███████▍  | 164/222 [01:24<00:31,  1.83it/s][A
     74%|███████▍  | 165/222 [01:24<00:30,  1.85it/s][A
     75%|███████▍  | 166/222 [01:25<00:30,  1.86it/s][A
     75%|███████▌  | 167/222 [01:25<00:29,  1.86it/s][A
     76%|███████▌  | 168/222 [01:26<00:28,  1.86it/s][A
     76%|███████▌  | 169/222 [01:26<00:27,  1.90it/s][A
     77%|███████▋  | 170/222 [01:27<00:26,  1.94it/s][A
     77%|███████▋  | 171/222 [01:27<00:26,  1.96it/s][A
     77%|███████▋  | 172/222 [01:28<00:25,  1.99it/s][A
     78%|███████▊  | 173/222 [01:28<00:24,  2.02it/s][A
     78%|███████▊  | 174/222 [01:29<00:23,  2.03it/s][A
     79%|███████▉  | 175/222 [01:29<00:23,  2.03it/s][A
     79%|███████▉  | 176/222 [01:30<00:22,  2.01it/s][A
     80%|███████▉  | 177/222 [01:30<00:22,  2.02it/s][A
     80%|████████  | 178/222 [01:31<00:22,  2.00it/s][A
     81%|████████  | 179/222 [01:31<00:21,  2.01it/s][A
     81%|████████  | 180/222 [01:32<00:21,  1.99it/s][A
     82%|████████▏ | 181/222 [01:32<00:21,  1.94it/s][A
     82%|████████▏ | 182/222 [01:33<00:20,  1.94it/s][A
     82%|████████▏ | 183/222 [01:33<00:20,  1.93it/s][A
     83%|████████▎ | 184/222 [01:34<00:19,  1.98it/s][A
     83%|████████▎ | 185/222 [01:34<00:18,  2.02it/s][A
     84%|████████▍ | 186/222 [01:35<00:17,  2.03it/s][A
     84%|████████▍ | 187/222 [01:35<00:17,  2.06it/s][A
     85%|████████▍ | 188/222 [01:36<00:16,  2.06it/s][A
     85%|████████▌ | 189/222 [01:36<00:15,  2.07it/s][A
     86%|████████▌ | 190/222 [01:37<00:15,  2.05it/s][A
     86%|████████▌ | 191/222 [01:37<00:15,  2.04it/s][A
     86%|████████▋ | 192/222 [01:38<00:14,  2.01it/s][A
     87%|████████▋ | 193/222 [01:38<00:14,  2.00it/s][A
     87%|████████▋ | 194/222 [01:39<00:14,  1.99it/s][A
     88%|████████▊ | 195/222 [01:39<00:13,  1.97it/s][A
     88%|████████▊ | 196/222 [01:40<00:13,  1.99it/s][A
     89%|████████▊ | 197/222 [01:40<00:12,  2.04it/s][A
     89%|████████▉ | 198/222 [01:41<00:11,  2.05it/s][A
     90%|████████▉ | 199/222 [01:41<00:11,  2.08it/s][A
     90%|█████████ | 200/222 [01:41<00:10,  2.09it/s][A
     91%|█████████ | 201/222 [01:42<00:10,  2.06it/s][A
     91%|█████████ | 202/222 [01:42<00:09,  2.09it/s][A
     91%|█████████▏| 203/222 [01:43<00:09,  2.07it/s][A
     92%|█████████▏| 204/222 [01:43<00:08,  2.10it/s][A
     92%|█████████▏| 205/222 [01:44<00:08,  2.09it/s][A
     93%|█████████▎| 206/222 [01:44<00:07,  2.08it/s][A
     93%|█████████▎| 207/222 [01:45<00:07,  2.11it/s][A
     94%|█████████▎| 208/222 [01:45<00:06,  2.14it/s][A
     94%|█████████▍| 209/222 [01:46<00:06,  2.14it/s][A
     95%|█████████▍| 210/222 [01:46<00:05,  2.10it/s][A
     95%|█████████▌| 211/222 [01:47<00:05,  2.08it/s][A
     95%|█████████▌| 212/222 [01:47<00:04,  2.04it/s][A
     96%|█████████▌| 213/222 [01:48<00:04,  2.04it/s][A
     96%|█████████▋| 214/222 [01:48<00:03,  2.03it/s][A
     97%|█████████▋| 215/222 [01:49<00:03,  2.04it/s][A
     97%|█████████▋| 216/222 [01:49<00:02,  2.09it/s][A
     98%|█████████▊| 217/222 [01:50<00:02,  2.12it/s][A
     98%|█████████▊| 218/222 [01:50<00:01,  2.11it/s][A
     99%|█████████▊| 219/222 [01:51<00:01,  2.14it/s][A
     99%|█████████▉| 220/222 [01:51<00:00,  2.18it/s][A
    100%|█████████▉| 221/222 [01:52<00:00,  2.13it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidWhiteRight.mp4 
    
    CPU times: user 2min 22s, sys: 446 ms, total: 2min 22s
    Wall time: 1min 53s
    [MoviePy] >>>> Building video test_videos_output/solidYellowLeft.mp4
    [MoviePy] Writing video test_videos_output/solidYellowLeft.mp4


    
      0%|          | 0/682 [00:00<?, ?it/s][A
      0%|          | 1/682 [00:00<06:58,  1.63it/s][A
      0%|          | 2/682 [00:01<06:59,  1.62it/s][A
      0%|          | 3/682 [00:01<06:52,  1.64it/s][A
      1%|          | 4/682 [00:02<06:43,  1.68it/s][A
      1%|          | 5/682 [00:03<07:02,  1.60it/s][A
      1%|          | 6/682 [00:03<06:46,  1.66it/s][A
      1%|          | 7/682 [00:04<06:38,  1.69it/s][A
      1%|          | 8/682 [00:04<06:30,  1.72it/s][A
      1%|▏         | 9/682 [00:05<06:29,  1.73it/s][A
      1%|▏         | 10/682 [00:05<06:25,  1.74it/s][A
      2%|▏         | 11/682 [00:06<06:29,  1.72it/s][A
      2%|▏         | 12/682 [00:07<06:26,  1.73it/s][A
      2%|▏         | 13/682 [00:07<06:30,  1.71it/s][A
      2%|▏         | 14/682 [00:08<06:27,  1.73it/s][A
      2%|▏         | 15/682 [00:08<06:25,  1.73it/s][A
      2%|▏         | 16/682 [00:09<06:19,  1.75it/s][A
      2%|▏         | 17/682 [00:09<06:21,  1.74it/s][A
      3%|▎         | 18/682 [00:10<06:21,  1.74it/s][A
      3%|▎         | 19/682 [00:11<06:17,  1.76it/s][A
      3%|▎         | 20/682 [00:11<06:18,  1.75it/s][A
      3%|▎         | 21/682 [00:12<06:19,  1.74it/s][A
      3%|▎         | 22/682 [00:12<06:20,  1.73it/s][A
      3%|▎         | 23/682 [00:13<06:15,  1.75it/s][A
      4%|▎         | 24/682 [00:13<06:16,  1.75it/s][A
      4%|▎         | 25/682 [00:14<06:19,  1.73it/s][A
      4%|▍         | 26/682 [00:15<06:20,  1.72it/s][A
      4%|▍         | 27/682 [00:15<06:24,  1.70it/s][A
      4%|▍         | 28/682 [00:16<06:27,  1.69it/s][A
      4%|▍         | 29/682 [00:16<06:32,  1.67it/s][A
      4%|▍         | 30/682 [00:17<06:22,  1.70it/s][A
      5%|▍         | 31/682 [00:18<06:21,  1.70it/s][A
      5%|▍         | 32/682 [00:18<06:15,  1.73it/s][A
      5%|▍         | 33/682 [00:19<06:11,  1.75it/s][A
      5%|▍         | 34/682 [00:19<06:08,  1.76it/s][A
      5%|▌         | 35/682 [00:20<06:05,  1.77it/s][A
      5%|▌         | 36/682 [00:20<06:02,  1.78it/s][A
      5%|▌         | 37/682 [00:21<05:58,  1.80it/s][A
      6%|▌         | 38/682 [00:21<05:57,  1.80it/s][A
      6%|▌         | 39/682 [00:22<05:55,  1.81it/s][A
      6%|▌         | 40/682 [00:23<05:56,  1.80it/s][A
      6%|▌         | 41/682 [00:23<06:01,  1.78it/s][A
      6%|▌         | 42/682 [00:24<06:17,  1.69it/s][A
      6%|▋         | 43/682 [00:24<06:19,  1.68it/s][A
      6%|▋         | 44/682 [00:25<06:13,  1.71it/s][A
      7%|▋         | 45/682 [00:26<06:04,  1.75it/s][A
      7%|▋         | 46/682 [00:26<06:06,  1.74it/s][A
      7%|▋         | 47/682 [00:27<06:07,  1.73it/s][A
      7%|▋         | 48/682 [00:27<06:07,  1.73it/s][A
      7%|▋         | 49/682 [00:28<06:03,  1.74it/s][A
      7%|▋         | 50/682 [00:28<06:02,  1.74it/s][A
      7%|▋         | 51/682 [00:29<06:03,  1.74it/s][A
      8%|▊         | 52/682 [00:30<06:04,  1.73it/s][A
      8%|▊         | 53/682 [00:30<06:02,  1.73it/s][A
      8%|▊         | 54/682 [00:31<06:03,  1.73it/s][A
      8%|▊         | 55/682 [00:31<06:09,  1.70it/s][A
      8%|▊         | 56/682 [00:32<06:12,  1.68it/s][A
      8%|▊         | 57/682 [00:33<06:12,  1.68it/s][A
      9%|▊         | 58/682 [00:33<06:10,  1.69it/s][A
      9%|▊         | 59/682 [00:34<06:03,  1.71it/s][A
      9%|▉         | 60/682 [00:34<05:55,  1.75it/s][A
      9%|▉         | 61/682 [00:35<06:03,  1.71it/s][A
      9%|▉         | 62/682 [00:35<06:01,  1.71it/s][A
      9%|▉         | 63/682 [00:36<06:03,  1.70it/s][A
      9%|▉         | 64/682 [00:37<06:03,  1.70it/s][A
     10%|▉         | 65/682 [00:37<06:00,  1.71it/s][A
     10%|▉         | 66/682 [00:38<05:56,  1.73it/s][A
     10%|▉         | 67/682 [00:38<05:51,  1.75it/s][A
     10%|▉         | 68/682 [00:39<05:53,  1.74it/s][A
     10%|█         | 69/682 [00:39<05:53,  1.74it/s][A
     10%|█         | 70/682 [00:40<05:46,  1.77it/s][A
     10%|█         | 71/682 [00:41<05:46,  1.76it/s][A
     11%|█         | 72/682 [00:41<05:47,  1.76it/s][A
     11%|█         | 73/682 [00:42<05:48,  1.75it/s][A
     11%|█         | 74/682 [00:42<05:46,  1.75it/s][A
     11%|█         | 75/682 [00:43<05:44,  1.76it/s][A
     11%|█         | 76/682 [00:43<05:46,  1.75it/s][A
     11%|█▏        | 77/682 [00:44<05:41,  1.77it/s][A
     11%|█▏        | 78/682 [00:45<05:45,  1.75it/s][A
     12%|█▏        | 79/682 [00:45<05:39,  1.78it/s][A
     12%|█▏        | 80/682 [00:46<05:39,  1.77it/s][A
     12%|█▏        | 81/682 [00:46<05:35,  1.79it/s][A
     12%|█▏        | 82/682 [00:47<05:33,  1.80it/s][A
     12%|█▏        | 83/682 [00:47<05:29,  1.82it/s][A
     12%|█▏        | 84/682 [00:48<05:33,  1.80it/s][A
     12%|█▏        | 85/682 [00:48<05:28,  1.82it/s][A
     13%|█▎        | 86/682 [00:49<05:27,  1.82it/s][A
     13%|█▎        | 87/682 [00:50<05:28,  1.81it/s][A
     13%|█▎        | 88/682 [00:50<05:25,  1.83it/s][A
     13%|█▎        | 89/682 [00:51<05:33,  1.78it/s][A
     13%|█▎        | 90/682 [00:51<05:33,  1.78it/s][A
     13%|█▎        | 91/682 [00:52<05:38,  1.75it/s][A
     13%|█▎        | 92/682 [00:52<05:43,  1.72it/s][A
     14%|█▎        | 93/682 [00:53<05:44,  1.71it/s][A
     14%|█▍        | 94/682 [00:54<05:36,  1.75it/s][A
     14%|█▍        | 95/682 [00:54<05:32,  1.77it/s][A
     14%|█▍        | 96/682 [00:55<05:31,  1.77it/s][A
     14%|█▍        | 97/682 [00:55<05:26,  1.79it/s][A
     14%|█▍        | 98/682 [00:56<05:25,  1.79it/s][A
     15%|█▍        | 99/682 [00:56<05:22,  1.81it/s][A
     15%|█▍        | 100/682 [00:57<05:26,  1.78it/s][A
     15%|█▍        | 101/682 [00:57<05:24,  1.79it/s][A
     15%|█▍        | 102/682 [00:58<05:30,  1.75it/s][A
     15%|█▌        | 103/682 [00:59<05:32,  1.74it/s][A
     15%|█▌        | 104/682 [00:59<05:31,  1.74it/s][A
     15%|█▌        | 105/682 [01:00<05:30,  1.74it/s][A
     16%|█▌        | 106/682 [01:00<05:33,  1.73it/s][A
     16%|█▌        | 107/682 [01:01<05:30,  1.74it/s][A
     16%|█▌        | 108/682 [01:02<05:29,  1.74it/s][A
     16%|█▌        | 109/682 [01:02<05:29,  1.74it/s][A
     16%|█▌        | 110/682 [01:03<05:29,  1.74it/s][A
     16%|█▋        | 111/682 [01:03<05:33,  1.71it/s][A
     16%|█▋        | 112/682 [01:04<05:36,  1.69it/s][A
     17%|█▋        | 113/682 [01:04<05:37,  1.68it/s][A
     17%|█▋        | 114/682 [01:05<05:38,  1.68it/s][A
     17%|█▋        | 115/682 [01:06<05:41,  1.66it/s][A
     17%|█▋        | 116/682 [01:06<05:45,  1.64it/s][A
     17%|█▋        | 117/682 [01:07<05:39,  1.66it/s][A
     17%|█▋        | 118/682 [01:08<05:39,  1.66it/s][A
     17%|█▋        | 119/682 [01:08<05:33,  1.69it/s][A
     18%|█▊        | 120/682 [01:09<05:28,  1.71it/s][A
     18%|█▊        | 121/682 [01:09<05:28,  1.71it/s][A
     18%|█▊        | 122/682 [01:10<05:28,  1.70it/s][A
     18%|█▊        | 123/682 [01:10<05:21,  1.74it/s][A
     18%|█▊        | 124/682 [01:11<05:18,  1.75it/s][A
     18%|█▊        | 125/682 [01:12<05:19,  1.75it/s][A
     18%|█▊        | 126/682 [01:12<05:17,  1.75it/s][A
     19%|█▊        | 127/682 [01:13<05:15,  1.76it/s][A
     19%|█▉        | 128/682 [01:13<05:15,  1.76it/s][A
     19%|█▉        | 129/682 [01:14<05:20,  1.73it/s][A
     19%|█▉        | 130/682 [01:14<05:18,  1.73it/s][A
     19%|█▉        | 131/682 [01:15<05:15,  1.75it/s][A
     19%|█▉        | 132/682 [01:15<05:07,  1.79it/s][A
     20%|█▉        | 133/682 [01:16<05:14,  1.75it/s][A
     20%|█▉        | 134/682 [01:17<05:16,  1.73it/s][A
     20%|█▉        | 135/682 [01:17<05:07,  1.78it/s][A
     20%|█▉        | 136/682 [01:18<05:03,  1.80it/s][A
     20%|██        | 137/682 [01:18<04:59,  1.82it/s][A
     20%|██        | 138/682 [01:19<05:01,  1.80it/s][A
     20%|██        | 139/682 [01:19<05:00,  1.81it/s][A
     21%|██        | 140/682 [01:20<04:57,  1.82it/s][A
     21%|██        | 141/682 [01:20<04:51,  1.86it/s][A
     21%|██        | 142/682 [01:21<04:52,  1.84it/s][A
     21%|██        | 143/682 [01:22<04:52,  1.84it/s][A
     21%|██        | 144/682 [01:22<04:46,  1.88it/s][A
     21%|██▏       | 145/682 [01:23<04:52,  1.84it/s][A
     21%|██▏       | 146/682 [01:23<04:51,  1.84it/s][A
     22%|██▏       | 147/682 [01:24<04:59,  1.79it/s][A
     22%|██▏       | 148/682 [01:24<05:02,  1.76it/s][A
     22%|██▏       | 149/682 [01:25<05:05,  1.75it/s][A
     22%|██▏       | 150/682 [01:25<05:03,  1.75it/s][A
     22%|██▏       | 151/682 [01:26<05:07,  1.73it/s][A
     22%|██▏       | 152/682 [01:27<05:09,  1.71it/s][A
     22%|██▏       | 153/682 [01:27<05:16,  1.67it/s][A
     23%|██▎       | 154/682 [01:28<05:13,  1.68it/s][A
     23%|██▎       | 155/682 [01:29<05:14,  1.68it/s][A
     23%|██▎       | 156/682 [01:29<05:17,  1.66it/s][A
     23%|██▎       | 157/682 [01:30<05:08,  1.70it/s][A
     23%|██▎       | 158/682 [01:30<05:06,  1.71it/s][A
     23%|██▎       | 159/682 [01:31<05:02,  1.73it/s][A
     23%|██▎       | 160/682 [01:31<04:58,  1.75it/s][A
     24%|██▎       | 161/682 [01:32<05:00,  1.73it/s][A
     24%|██▍       | 162/682 [01:33<05:04,  1.71it/s][A
     24%|██▍       | 163/682 [01:33<05:01,  1.72it/s][A
     24%|██▍       | 164/682 [01:34<05:02,  1.71it/s][A
     24%|██▍       | 165/682 [01:34<04:59,  1.72it/s][A
     24%|██▍       | 166/682 [01:35<04:58,  1.73it/s][A
     24%|██▍       | 167/682 [01:35<05:00,  1.71it/s][A
     25%|██▍       | 168/682 [01:36<04:59,  1.71it/s][A
     25%|██▍       | 169/682 [01:37<04:53,  1.75it/s][A
     25%|██▍       | 170/682 [01:37<04:48,  1.77it/s][A
     25%|██▌       | 171/682 [01:38<04:45,  1.79it/s][A
     25%|██▌       | 172/682 [01:38<04:45,  1.79it/s][A
     25%|██▌       | 173/682 [01:39<04:46,  1.78it/s][A
     26%|██▌       | 174/682 [01:39<04:43,  1.79it/s][A
     26%|██▌       | 175/682 [01:40<04:37,  1.82it/s][A
     26%|██▌       | 176/682 [01:40<04:34,  1.84it/s][A
     26%|██▌       | 177/682 [01:41<04:31,  1.86it/s][A
     26%|██▌       | 178/682 [01:41<04:29,  1.87it/s][A
     26%|██▌       | 179/682 [01:42<04:29,  1.87it/s][A
     26%|██▋       | 180/682 [01:43<04:29,  1.86it/s][A
     27%|██▋       | 181/682 [01:43<04:37,  1.80it/s][A
     27%|██▋       | 182/682 [01:44<04:38,  1.80it/s][A
     27%|██▋       | 183/682 [01:44<04:33,  1.82it/s][A
     27%|██▋       | 184/682 [01:45<04:29,  1.85it/s][A
     27%|██▋       | 185/682 [01:45<04:27,  1.86it/s][A
     27%|██▋       | 186/682 [01:46<04:22,  1.89it/s][A
     27%|██▋       | 187/682 [01:46<04:19,  1.91it/s][A
     28%|██▊       | 188/682 [01:47<04:17,  1.92it/s][A
     28%|██▊       | 189/682 [01:47<04:15,  1.93it/s][A
     28%|██▊       | 190/682 [01:48<04:14,  1.93it/s][A
     28%|██▊       | 191/682 [01:48<04:19,  1.89it/s][A
     28%|██▊       | 192/682 [01:49<04:22,  1.87it/s][A
     28%|██▊       | 193/682 [01:50<04:25,  1.84it/s][A
     28%|██▊       | 194/682 [01:50<04:22,  1.86it/s][A
     29%|██▊       | 195/682 [01:51<04:15,  1.90it/s][A
     29%|██▊       | 196/682 [01:51<04:15,  1.90it/s][A
     29%|██▉       | 197/682 [01:52<04:12,  1.92it/s][A
     29%|██▉       | 198/682 [01:52<04:09,  1.94it/s][A
     29%|██▉       | 199/682 [01:53<04:08,  1.94it/s][A
     29%|██▉       | 200/682 [01:53<04:05,  1.97it/s][A
     29%|██▉       | 201/682 [01:54<04:06,  1.95it/s][A
     30%|██▉       | 202/682 [01:54<04:05,  1.95it/s][A
     30%|██▉       | 203/682 [01:55<04:12,  1.90it/s][A
     30%|██▉       | 204/682 [01:55<04:16,  1.86it/s][A
     30%|███       | 205/682 [01:56<04:19,  1.84it/s][A
     30%|███       | 206/682 [01:56<04:22,  1.81it/s][A
     30%|███       | 207/682 [01:57<04:19,  1.83it/s][A
     30%|███       | 208/682 [01:57<04:16,  1.84it/s][A
     31%|███       | 209/682 [01:58<04:19,  1.82it/s][A
     31%|███       | 210/682 [01:59<04:23,  1.79it/s][A
     31%|███       | 211/682 [01:59<04:28,  1.75it/s][A
     31%|███       | 212/682 [02:00<04:30,  1.74it/s][A
     31%|███       | 213/682 [02:00<04:26,  1.76it/s][A
     31%|███▏      | 214/682 [02:01<04:27,  1.75it/s][A
     32%|███▏      | 215/682 [02:01<04:28,  1.74it/s][A
     32%|███▏      | 216/682 [02:02<04:27,  1.74it/s][A
     32%|███▏      | 217/682 [02:03<04:22,  1.77it/s][A
     32%|███▏      | 218/682 [02:03<04:18,  1.80it/s][A
     32%|███▏      | 219/682 [02:04<04:18,  1.79it/s][A
     32%|███▏      | 220/682 [02:04<04:17,  1.79it/s][A
     32%|███▏      | 221/682 [02:05<04:19,  1.78it/s][A
     33%|███▎      | 222/682 [02:05<04:10,  1.83it/s][A
     33%|███▎      | 223/682 [02:06<04:07,  1.86it/s][A
     33%|███▎      | 224/682 [02:06<04:09,  1.84it/s][A
     33%|███▎      | 225/682 [02:07<04:08,  1.84it/s][A
     33%|███▎      | 226/682 [02:07<04:06,  1.85it/s][A
     33%|███▎      | 227/682 [02:08<04:04,  1.86it/s][A
     33%|███▎      | 228/682 [02:09<04:08,  1.83it/s][A
     34%|███▎      | 229/682 [02:09<04:10,  1.81it/s][A
     34%|███▎      | 230/682 [02:10<04:07,  1.82it/s][A
     34%|███▍      | 231/682 [02:10<04:06,  1.83it/s][A
     34%|███▍      | 232/682 [02:11<04:06,  1.83it/s][A
     34%|███▍      | 233/682 [02:11<04:08,  1.80it/s][A
     34%|███▍      | 234/682 [02:12<04:05,  1.82it/s][A
     34%|███▍      | 235/682 [02:12<04:04,  1.83it/s][A
     35%|███▍      | 236/682 [02:13<04:02,  1.84it/s][A
     35%|███▍      | 237/682 [02:13<03:58,  1.87it/s][A
     35%|███▍      | 238/682 [02:14<03:57,  1.87it/s][A
     35%|███▌      | 239/682 [02:15<03:55,  1.88it/s][A
     35%|███▌      | 240/682 [02:15<03:54,  1.89it/s][A
     35%|███▌      | 241/682 [02:16<04:03,  1.81it/s][A
     35%|███▌      | 242/682 [02:16<04:02,  1.81it/s][A
     36%|███▌      | 243/682 [02:17<04:03,  1.80it/s][A
     36%|███▌      | 244/682 [02:17<04:00,  1.82it/s][A
     36%|███▌      | 245/682 [02:18<03:58,  1.83it/s][A
     36%|███▌      | 246/682 [02:18<03:54,  1.86it/s][A
     36%|███▌      | 247/682 [02:19<03:54,  1.86it/s][A
     36%|███▋      | 248/682 [02:19<03:53,  1.86it/s][A
     37%|███▋      | 249/682 [02:20<03:56,  1.83it/s][A
     37%|███▋      | 250/682 [02:21<03:53,  1.85it/s][A
     37%|███▋      | 251/682 [02:21<03:58,  1.81it/s][A
     37%|███▋      | 252/682 [02:22<03:53,  1.84it/s][A
     37%|███▋      | 253/682 [02:22<03:50,  1.86it/s][A
     37%|███▋      | 254/682 [02:23<03:54,  1.83it/s][A
     37%|███▋      | 255/682 [02:23<03:53,  1.83it/s][A
     38%|███▊      | 256/682 [02:24<03:53,  1.83it/s][A
     38%|███▊      | 257/682 [02:24<03:46,  1.87it/s][A
     38%|███▊      | 258/682 [02:25<03:41,  1.91it/s][A
     38%|███▊      | 259/682 [02:25<03:40,  1.92it/s][A
     38%|███▊      | 260/682 [02:26<03:42,  1.89it/s][A
     38%|███▊      | 261/682 [02:26<03:46,  1.86it/s][A
     38%|███▊      | 262/682 [02:27<03:46,  1.85it/s][A
     39%|███▊      | 263/682 [02:28<03:48,  1.83it/s][A
     39%|███▊      | 264/682 [02:28<03:47,  1.84it/s][A
     39%|███▉      | 265/682 [02:29<03:46,  1.85it/s][A
     39%|███▉      | 266/682 [02:29<03:45,  1.84it/s][A
     39%|███▉      | 267/682 [02:30<03:41,  1.87it/s][A
     39%|███▉      | 268/682 [02:30<03:41,  1.87it/s][A
     39%|███▉      | 269/682 [02:31<03:40,  1.87it/s][A
     40%|███▉      | 270/682 [02:31<03:40,  1.87it/s][A
     40%|███▉      | 271/682 [02:32<03:41,  1.86it/s][A
     40%|███▉      | 272/682 [02:32<03:39,  1.87it/s][A
     40%|████      | 273/682 [02:33<03:40,  1.86it/s][A
     40%|████      | 274/682 [02:33<03:36,  1.88it/s][A
     40%|████      | 275/682 [02:34<03:36,  1.88it/s][A
     40%|████      | 276/682 [02:35<03:37,  1.86it/s][A
     41%|████      | 277/682 [02:35<03:39,  1.85it/s][A
     41%|████      | 278/682 [02:36<03:35,  1.88it/s][A
     41%|████      | 279/682 [02:36<03:35,  1.87it/s][A
     41%|████      | 280/682 [02:37<03:35,  1.86it/s][A
     41%|████      | 281/682 [02:37<03:31,  1.90it/s][A
     41%|████▏     | 282/682 [02:38<03:33,  1.87it/s][A
     41%|████▏     | 283/682 [02:38<03:31,  1.88it/s][A
     42%|████▏     | 284/682 [02:39<03:28,  1.91it/s][A
     42%|████▏     | 285/682 [02:39<03:29,  1.90it/s][A
     42%|████▏     | 286/682 [02:40<03:24,  1.94it/s][A
     42%|████▏     | 287/682 [02:40<03:22,  1.95it/s][A
     42%|████▏     | 288/682 [02:41<03:18,  1.98it/s][A
     42%|████▏     | 289/682 [02:41<03:20,  1.96it/s][A
     43%|████▎     | 290/682 [02:42<03:21,  1.94it/s][A
     43%|████▎     | 291/682 [02:42<03:18,  1.97it/s][A
     43%|████▎     | 292/682 [02:43<03:16,  1.99it/s][A
     43%|████▎     | 293/682 [02:43<03:12,  2.02it/s][A
     43%|████▎     | 294/682 [02:44<03:10,  2.03it/s][A
     43%|████▎     | 295/682 [02:44<03:11,  2.02it/s][A
     43%|████▎     | 296/682 [02:45<03:07,  2.06it/s][A
     44%|████▎     | 297/682 [02:45<03:04,  2.09it/s][A
     44%|████▎     | 298/682 [02:46<03:03,  2.10it/s][A
     44%|████▍     | 299/682 [02:46<03:04,  2.08it/s][A
     44%|████▍     | 300/682 [02:47<03:03,  2.08it/s][A
     44%|████▍     | 301/682 [02:47<03:06,  2.04it/s][A
     44%|████▍     | 302/682 [02:48<03:12,  1.97it/s][A
     44%|████▍     | 303/682 [02:48<03:11,  1.97it/s][A
     45%|████▍     | 304/682 [02:49<03:12,  1.97it/s][A
     45%|████▍     | 305/682 [02:49<03:12,  1.96it/s][A
     45%|████▍     | 306/682 [02:50<03:11,  1.97it/s][A
     45%|████▌     | 307/682 [02:50<03:14,  1.93it/s][A
     45%|████▌     | 308/682 [02:51<03:16,  1.91it/s][A
     45%|████▌     | 309/682 [02:51<03:16,  1.89it/s][A
     45%|████▌     | 310/682 [02:52<03:15,  1.90it/s][A
     46%|████▌     | 311/682 [02:52<03:15,  1.89it/s][A
     46%|████▌     | 312/682 [02:53<03:16,  1.89it/s][A
     46%|████▌     | 313/682 [02:53<03:14,  1.90it/s][A
     46%|████▌     | 314/682 [02:54<03:15,  1.88it/s][A
     46%|████▌     | 315/682 [02:55<03:15,  1.88it/s][A
     46%|████▋     | 316/682 [02:55<03:15,  1.87it/s][A
     46%|████▋     | 317/682 [02:56<03:14,  1.87it/s][A
     47%|████▋     | 318/682 [02:56<03:14,  1.87it/s][A
     47%|████▋     | 319/682 [02:57<03:14,  1.87it/s][A
     47%|████▋     | 320/682 [02:57<03:15,  1.85it/s][A
     47%|████▋     | 321/682 [02:58<03:13,  1.87it/s][A
     47%|████▋     | 322/682 [02:58<03:13,  1.86it/s][A
     47%|████▋     | 323/682 [02:59<03:11,  1.88it/s][A
     48%|████▊     | 324/682 [02:59<03:09,  1.89it/s][A
     48%|████▊     | 325/682 [03:00<03:03,  1.94it/s][A
     48%|████▊     | 326/682 [03:00<02:57,  2.00it/s][A
     48%|████▊     | 327/682 [03:01<02:56,  2.01it/s][A
     48%|████▊     | 328/682 [03:01<02:56,  2.00it/s][A
     48%|████▊     | 329/682 [03:02<02:54,  2.02it/s][A
     48%|████▊     | 330/682 [03:02<02:52,  2.04it/s][A
     49%|████▊     | 331/682 [03:03<02:59,  1.96it/s][A
     49%|████▊     | 332/682 [03:03<02:56,  1.99it/s][A
     49%|████▉     | 333/682 [03:04<02:53,  2.02it/s][A
     49%|████▉     | 334/682 [03:04<02:52,  2.01it/s][A
     49%|████▉     | 335/682 [03:05<02:55,  1.98it/s][A
     49%|████▉     | 336/682 [03:05<02:52,  2.00it/s][A
     49%|████▉     | 337/682 [03:06<02:55,  1.96it/s][A
     50%|████▉     | 338/682 [03:06<02:57,  1.93it/s][A
     50%|████▉     | 339/682 [03:07<02:54,  1.96it/s][A
     50%|████▉     | 340/682 [03:07<02:52,  1.98it/s][A
     50%|█████     | 341/682 [03:08<02:50,  2.00it/s][A
     50%|█████     | 342/682 [03:08<02:50,  1.99it/s][A
     50%|█████     | 343/682 [03:09<02:49,  2.00it/s][A
     50%|█████     | 344/682 [03:09<02:48,  2.00it/s][A
     51%|█████     | 345/682 [03:10<02:45,  2.03it/s][A
     51%|█████     | 346/682 [03:10<02:46,  2.02it/s][A
     51%|█████     | 347/682 [03:11<02:47,  2.00it/s][A
     51%|█████     | 348/682 [03:11<02:47,  1.99it/s][A
     51%|█████     | 349/682 [03:12<02:47,  1.98it/s][A
     51%|█████▏    | 350/682 [03:12<02:45,  2.01it/s][A
     51%|█████▏    | 351/682 [03:13<02:41,  2.04it/s][A
     52%|█████▏    | 352/682 [03:13<02:38,  2.09it/s][A
     52%|█████▏    | 353/682 [03:14<02:36,  2.10it/s][A
     52%|█████▏    | 354/682 [03:14<02:34,  2.12it/s][A
     52%|█████▏    | 355/682 [03:15<02:33,  2.13it/s][A
     52%|█████▏    | 356/682 [03:15<02:32,  2.13it/s][A
     52%|█████▏    | 357/682 [03:16<02:33,  2.12it/s][A
     52%|█████▏    | 358/682 [03:16<02:31,  2.14it/s][A
     53%|█████▎    | 359/682 [03:16<02:28,  2.17it/s][A
     53%|█████▎    | 360/682 [03:17<02:29,  2.15it/s][A
     53%|█████▎    | 361/682 [03:17<02:34,  2.08it/s][A
     53%|█████▎    | 362/682 [03:18<02:34,  2.07it/s][A
     53%|█████▎    | 363/682 [03:18<02:36,  2.04it/s][A
     53%|█████▎    | 364/682 [03:19<02:37,  2.02it/s][A
     54%|█████▎    | 365/682 [03:19<02:36,  2.03it/s][A
     54%|█████▎    | 366/682 [03:20<02:34,  2.04it/s][A
     54%|█████▍    | 367/682 [03:20<02:33,  2.05it/s][A
     54%|█████▍    | 368/682 [03:21<02:35,  2.02it/s][A
     54%|█████▍    | 369/682 [03:21<02:33,  2.04it/s][A
     54%|█████▍    | 370/682 [03:22<02:32,  2.05it/s][A
     54%|█████▍    | 371/682 [03:22<02:34,  2.02it/s][A
     55%|█████▍    | 372/682 [03:23<02:35,  1.99it/s][A
     55%|█████▍    | 373/682 [03:23<02:34,  2.00it/s][A
     55%|█████▍    | 374/682 [03:24<02:34,  1.99it/s][A
     55%|█████▍    | 375/682 [03:24<02:36,  1.97it/s][A
     55%|█████▌    | 376/682 [03:25<02:37,  1.95it/s][A
     55%|█████▌    | 377/682 [03:25<02:35,  1.97it/s][A
     55%|█████▌    | 378/682 [03:26<02:33,  1.98it/s][A
     56%|█████▌    | 379/682 [03:26<02:32,  1.99it/s][A
     56%|█████▌    | 380/682 [03:27<02:33,  1.96it/s][A
     56%|█████▌    | 381/682 [03:28<02:34,  1.95it/s][A
     56%|█████▌    | 382/682 [03:28<02:30,  1.99it/s][A
     56%|█████▌    | 383/682 [03:29<02:32,  1.96it/s][A
     56%|█████▋    | 384/682 [03:29<02:33,  1.94it/s][A
     56%|█████▋    | 385/682 [03:30<02:33,  1.94it/s][A
     57%|█████▋    | 386/682 [03:30<02:32,  1.95it/s][A
     57%|█████▋    | 387/682 [03:31<02:32,  1.94it/s][A
     57%|█████▋    | 388/682 [03:31<02:31,  1.95it/s][A
     57%|█████▋    | 389/682 [03:32<02:30,  1.95it/s][A
     57%|█████▋    | 390/682 [03:32<02:29,  1.95it/s][A
     57%|█████▋    | 391/682 [03:33<02:32,  1.90it/s][A
     57%|█████▋    | 392/682 [03:33<02:33,  1.89it/s][A
     58%|█████▊    | 393/682 [03:34<02:38,  1.83it/s][A
     58%|█████▊    | 394/682 [03:34<02:34,  1.87it/s][A
     58%|█████▊    | 395/682 [03:35<02:34,  1.86it/s][A
     58%|█████▊    | 396/682 [03:35<02:32,  1.87it/s][A
     58%|█████▊    | 397/682 [03:36<02:30,  1.90it/s][A
     58%|█████▊    | 398/682 [03:36<02:29,  1.90it/s][A
     59%|█████▊    | 399/682 [03:37<02:28,  1.91it/s][A
     59%|█████▊    | 400/682 [03:37<02:28,  1.90it/s][A
     59%|█████▉    | 401/682 [03:38<02:22,  1.97it/s][A
     59%|█████▉    | 402/682 [03:38<02:25,  1.93it/s][A
     59%|█████▉    | 403/682 [03:39<02:22,  1.95it/s][A
     59%|█████▉    | 404/682 [03:39<02:18,  2.01it/s][A
     59%|█████▉    | 405/682 [03:40<02:17,  2.02it/s][A
     60%|█████▉    | 406/682 [03:40<02:15,  2.03it/s][A
     60%|█████▉    | 407/682 [03:41<02:15,  2.03it/s][A
     60%|█████▉    | 408/682 [03:41<02:14,  2.03it/s][A
     60%|█████▉    | 409/682 [03:42<02:14,  2.04it/s][A
     60%|██████    | 410/682 [03:42<02:12,  2.05it/s][A
     60%|██████    | 411/682 [03:43<02:13,  2.03it/s][A
     60%|██████    | 412/682 [03:43<02:10,  2.06it/s][A
     61%|██████    | 413/682 [03:44<02:12,  2.02it/s][A
     61%|██████    | 414/682 [03:44<02:12,  2.02it/s][A
     61%|██████    | 415/682 [03:45<02:13,  2.00it/s][A
     61%|██████    | 416/682 [03:45<02:14,  1.97it/s][A
     61%|██████    | 417/682 [03:46<02:14,  1.97it/s][A
     61%|██████▏   | 418/682 [03:46<02:12,  1.99it/s][A
     61%|██████▏   | 419/682 [03:47<02:15,  1.95it/s][A
     62%|██████▏   | 420/682 [03:47<02:15,  1.94it/s][A
     62%|██████▏   | 421/682 [03:48<02:17,  1.90it/s][A
     62%|██████▏   | 422/682 [03:49<02:16,  1.91it/s][A
     62%|██████▏   | 423/682 [03:49<02:18,  1.87it/s][A
     62%|██████▏   | 424/682 [03:50<02:19,  1.85it/s][A
     62%|██████▏   | 425/682 [03:50<02:15,  1.89it/s][A
     62%|██████▏   | 426/682 [03:51<02:16,  1.87it/s][A
     63%|██████▎   | 427/682 [03:51<02:13,  1.91it/s][A
     63%|██████▎   | 428/682 [03:52<02:14,  1.89it/s][A
     63%|██████▎   | 429/682 [03:52<02:11,  1.92it/s][A
     63%|██████▎   | 430/682 [03:53<02:08,  1.96it/s][A
     63%|██████▎   | 431/682 [03:53<02:08,  1.96it/s][A
     63%|██████▎   | 432/682 [03:54<02:08,  1.94it/s][A
     63%|██████▎   | 433/682 [03:54<02:07,  1.96it/s][A
     64%|██████▎   | 434/682 [03:55<02:06,  1.96it/s][A
     64%|██████▍   | 435/682 [03:55<02:05,  1.97it/s][A
     64%|██████▍   | 436/682 [03:56<02:05,  1.95it/s][A
     64%|██████▍   | 437/682 [03:56<02:06,  1.94it/s][A
     64%|██████▍   | 438/682 [03:57<02:07,  1.92it/s][A
     64%|██████▍   | 439/682 [03:57<02:05,  1.94it/s][A
     65%|██████▍   | 440/682 [03:58<02:06,  1.91it/s][A
     65%|██████▍   | 441/682 [03:58<02:07,  1.90it/s][A
     65%|██████▍   | 442/682 [03:59<02:06,  1.90it/s][A
     65%|██████▍   | 443/682 [03:59<02:06,  1.90it/s][A
     65%|██████▌   | 444/682 [04:00<02:06,  1.89it/s][A
     65%|██████▌   | 445/682 [04:01<02:08,  1.85it/s][A
     65%|██████▌   | 446/682 [04:01<02:05,  1.88it/s][A
     66%|██████▌   | 447/682 [04:02<02:04,  1.89it/s][A
     66%|██████▌   | 448/682 [04:02<02:04,  1.89it/s][A
     66%|██████▌   | 449/682 [04:03<02:02,  1.90it/s][A
     66%|██████▌   | 450/682 [04:03<02:02,  1.89it/s][A
     66%|██████▌   | 451/682 [04:04<02:05,  1.84it/s][A
     66%|██████▋   | 452/682 [04:04<02:06,  1.82it/s][A
     66%|██████▋   | 453/682 [04:05<02:05,  1.82it/s][A
     67%|██████▋   | 454/682 [04:05<02:03,  1.85it/s][A
     67%|██████▋   | 455/682 [04:06<02:03,  1.84it/s][A
     67%|██████▋   | 456/682 [04:07<02:05,  1.80it/s][A
     67%|██████▋   | 457/682 [04:07<02:05,  1.79it/s][A
     67%|██████▋   | 458/682 [04:08<02:04,  1.80it/s][A
     67%|██████▋   | 459/682 [04:08<02:02,  1.81it/s][A
     67%|██████▋   | 460/682 [04:09<02:01,  1.83it/s][A
     68%|██████▊   | 461/682 [04:09<02:00,  1.83it/s][A
     68%|██████▊   | 462/682 [04:10<01:59,  1.84it/s][A
     68%|██████▊   | 463/682 [04:10<01:58,  1.85it/s][A
     68%|██████▊   | 464/682 [04:11<01:57,  1.85it/s][A
     68%|██████▊   | 465/682 [04:11<01:55,  1.87it/s][A
     68%|██████▊   | 466/682 [04:12<01:55,  1.87it/s][A
     68%|██████▊   | 467/682 [04:12<01:56,  1.84it/s][A
     69%|██████▊   | 468/682 [04:13<01:57,  1.82it/s][A
     69%|██████▉   | 469/682 [04:14<01:57,  1.81it/s][A
     69%|██████▉   | 470/682 [04:14<01:56,  1.82it/s][A
     69%|██████▉   | 471/682 [04:15<01:57,  1.79it/s][A
     69%|██████▉   | 472/682 [04:15<01:57,  1.79it/s][A
     69%|██████▉   | 473/682 [04:16<01:58,  1.77it/s][A
     70%|██████▉   | 474/682 [04:16<01:57,  1.76it/s][A
     70%|██████▉   | 475/682 [04:17<01:58,  1.75it/s][A
     70%|██████▉   | 476/682 [04:18<01:57,  1.75it/s][A
     70%|██████▉   | 477/682 [04:18<01:54,  1.79it/s][A
     70%|███████   | 478/682 [04:19<01:52,  1.82it/s][A
     70%|███████   | 479/682 [04:19<01:50,  1.84it/s][A
     70%|███████   | 480/682 [04:20<01:50,  1.82it/s][A
     71%|███████   | 481/682 [04:20<01:50,  1.81it/s][A
     71%|███████   | 482/682 [04:21<01:49,  1.83it/s][A
     71%|███████   | 483/682 [04:21<01:48,  1.84it/s][A
     71%|███████   | 484/682 [04:22<01:46,  1.86it/s][A
     71%|███████   | 485/682 [04:22<01:47,  1.83it/s][A
     71%|███████▏  | 486/682 [04:23<01:46,  1.84it/s][A
     71%|███████▏  | 487/682 [04:24<01:49,  1.79it/s][A
     72%|███████▏  | 488/682 [04:24<01:47,  1.81it/s][A
     72%|███████▏  | 489/682 [04:25<01:45,  1.82it/s][A
     72%|███████▏  | 490/682 [04:25<01:44,  1.83it/s][A
     72%|███████▏  | 491/682 [04:26<01:44,  1.82it/s][A
     72%|███████▏  | 492/682 [04:26<01:43,  1.83it/s][A
     72%|███████▏  | 493/682 [04:27<01:42,  1.84it/s][A
     72%|███████▏  | 494/682 [04:27<01:42,  1.84it/s][A
     73%|███████▎  | 495/682 [04:28<01:42,  1.82it/s][A
     73%|███████▎  | 496/682 [04:28<01:40,  1.84it/s][A
     73%|███████▎  | 497/682 [04:29<01:41,  1.82it/s][A
     73%|███████▎  | 498/682 [04:30<01:41,  1.81it/s][A
     73%|███████▎  | 499/682 [04:30<01:41,  1.80it/s][A
     73%|███████▎  | 500/682 [04:31<01:44,  1.74it/s][A
     73%|███████▎  | 501/682 [04:31<01:45,  1.72it/s][A
     74%|███████▎  | 502/682 [04:32<01:44,  1.72it/s][A
     74%|███████▍  | 503/682 [04:33<01:45,  1.70it/s][A
     74%|███████▍  | 504/682 [04:33<01:47,  1.66it/s][A
     74%|███████▍  | 505/682 [04:34<01:49,  1.62it/s][A
     74%|███████▍  | 506/682 [04:35<01:50,  1.60it/s][A
     74%|███████▍  | 507/682 [04:35<01:52,  1.56it/s][A
     74%|███████▍  | 508/682 [04:36<01:54,  1.52it/s][A
     75%|███████▍  | 509/682 [04:37<01:57,  1.48it/s][A
     75%|███████▍  | 510/682 [04:37<02:00,  1.43it/s][A
     75%|███████▍  | 511/682 [04:38<02:05,  1.37it/s][A
     75%|███████▌  | 512/682 [04:39<02:08,  1.32it/s][A
     75%|███████▌  | 513/682 [04:40<02:09,  1.31it/s][A
     75%|███████▌  | 514/682 [04:41<02:09,  1.30it/s][A
     76%|███████▌  | 515/682 [04:41<02:08,  1.30it/s][A
     76%|███████▌  | 516/682 [04:42<02:05,  1.32it/s][A
     76%|███████▌  | 517/682 [04:43<02:03,  1.34it/s][A
     76%|███████▌  | 518/682 [04:43<02:00,  1.36it/s][A
     76%|███████▌  | 519/682 [04:44<01:57,  1.38it/s][A
     76%|███████▌  | 520/682 [04:45<01:58,  1.37it/s][A
     76%|███████▋  | 521/682 [04:46<01:55,  1.39it/s][A
     77%|███████▋  | 522/682 [04:46<01:52,  1.42it/s][A
     77%|███████▋  | 523/682 [04:47<01:52,  1.41it/s][A
     77%|███████▋  | 524/682 [04:48<01:54,  1.38it/s][A
     77%|███████▋  | 525/682 [04:49<01:54,  1.37it/s][A
     77%|███████▋  | 526/682 [04:49<01:58,  1.32it/s][A
     77%|███████▋  | 527/682 [04:50<01:56,  1.33it/s][A
     77%|███████▋  | 528/682 [04:51<01:53,  1.36it/s][A
     78%|███████▊  | 529/682 [04:51<01:51,  1.38it/s][A
     78%|███████▊  | 530/682 [04:52<01:50,  1.37it/s][A
     78%|███████▊  | 531/682 [04:53<01:50,  1.37it/s][A
     78%|███████▊  | 532/682 [04:54<01:50,  1.35it/s][A
     78%|███████▊  | 533/682 [04:54<01:51,  1.34it/s][A
     78%|███████▊  | 534/682 [04:55<01:53,  1.31it/s][A
     78%|███████▊  | 535/682 [04:56<01:55,  1.27it/s][A
     79%|███████▊  | 536/682 [04:57<01:56,  1.25it/s][A
     79%|███████▊  | 537/682 [04:58<01:51,  1.30it/s][A
     79%|███████▉  | 538/682 [04:58<01:47,  1.34it/s][A
     79%|███████▉  | 539/682 [04:59<01:42,  1.39it/s][A
     79%|███████▉  | 540/682 [05:00<01:37,  1.46it/s][A
     79%|███████▉  | 541/682 [05:00<01:31,  1.54it/s][A
     79%|███████▉  | 542/682 [05:01<01:24,  1.65it/s][A
     80%|███████▉  | 543/682 [05:01<01:20,  1.73it/s][A
     80%|███████▉  | 544/682 [05:02<01:17,  1.79it/s][A
     80%|███████▉  | 545/682 [05:02<01:13,  1.86it/s][A
     80%|████████  | 546/682 [05:03<01:12,  1.87it/s][A
     80%|████████  | 547/682 [05:03<01:10,  1.91it/s][A
     80%|████████  | 548/682 [05:04<01:09,  1.92it/s][A
     80%|████████  | 549/682 [05:04<01:08,  1.94it/s][A
     81%|████████  | 550/682 [05:05<01:07,  1.96it/s][A
     81%|████████  | 551/682 [05:05<01:05,  1.99it/s][A
     81%|████████  | 552/682 [05:06<01:05,  1.97it/s][A
     81%|████████  | 553/682 [05:06<01:06,  1.95it/s][A
     81%|████████  | 554/682 [05:07<01:05,  1.95it/s][A
     81%|████████▏ | 555/682 [05:07<01:04,  1.98it/s][A
     82%|████████▏ | 556/682 [05:08<01:03,  1.99it/s][A
     82%|████████▏ | 557/682 [05:08<01:03,  1.98it/s][A
     82%|████████▏ | 558/682 [05:09<01:03,  1.97it/s][A
     82%|████████▏ | 559/682 [05:09<01:02,  1.97it/s][A
     82%|████████▏ | 560/682 [05:10<01:01,  1.97it/s][A
     82%|████████▏ | 561/682 [05:10<01:01,  1.96it/s][A
     82%|████████▏ | 562/682 [05:11<01:00,  1.99it/s][A
     83%|████████▎ | 563/682 [05:11<01:01,  1.95it/s][A
     83%|████████▎ | 564/682 [05:12<00:59,  1.97it/s][A
     83%|████████▎ | 565/682 [05:12<00:59,  1.95it/s][A
     83%|████████▎ | 566/682 [05:13<00:58,  1.98it/s][A
     83%|████████▎ | 567/682 [05:13<00:58,  1.97it/s][A
     83%|████████▎ | 568/682 [05:14<00:58,  1.96it/s][A
     83%|████████▎ | 569/682 [05:14<00:58,  1.94it/s][A
     84%|████████▎ | 570/682 [05:15<00:57,  1.95it/s][A
     84%|████████▎ | 571/682 [05:15<00:57,  1.92it/s][A
     84%|████████▍ | 572/682 [05:16<00:58,  1.89it/s][A
     84%|████████▍ | 573/682 [05:17<00:58,  1.86it/s][A
     84%|████████▍ | 574/682 [05:17<00:58,  1.84it/s][A
     84%|████████▍ | 575/682 [05:18<00:58,  1.81it/s][A
     84%|████████▍ | 576/682 [05:18<00:58,  1.80it/s][A
     85%|████████▍ | 577/682 [05:19<00:58,  1.81it/s][A
     85%|████████▍ | 578/682 [05:19<00:55,  1.86it/s][A
     85%|████████▍ | 579/682 [05:20<00:54,  1.88it/s][A
     85%|████████▌ | 580/682 [05:20<00:53,  1.91it/s][A
     85%|████████▌ | 581/682 [05:21<00:53,  1.90it/s][A
     85%|████████▌ | 582/682 [05:21<00:51,  1.94it/s][A
     85%|████████▌ | 583/682 [05:22<00:51,  1.91it/s][A
     86%|████████▌ | 584/682 [05:22<00:50,  1.94it/s][A
     86%|████████▌ | 585/682 [05:23<00:50,  1.92it/s][A
     86%|████████▌ | 586/682 [05:23<00:50,  1.91it/s][A
     86%|████████▌ | 587/682 [05:24<00:49,  1.90it/s][A
     86%|████████▌ | 588/682 [05:24<00:49,  1.90it/s][A
     86%|████████▋ | 589/682 [05:25<00:49,  1.88it/s][A
     87%|████████▋ | 590/682 [05:26<00:48,  1.88it/s][A
     87%|████████▋ | 591/682 [05:26<00:48,  1.88it/s][A
     87%|████████▋ | 592/682 [05:27<00:47,  1.90it/s][A
     87%|████████▋ | 593/682 [05:27<00:47,  1.87it/s][A
     87%|████████▋ | 594/682 [05:28<00:47,  1.86it/s][A
     87%|████████▋ | 595/682 [05:28<00:47,  1.85it/s][A
     87%|████████▋ | 596/682 [05:29<00:45,  1.88it/s][A
     88%|████████▊ | 597/682 [05:29<00:44,  1.91it/s][A
     88%|████████▊ | 598/682 [05:30<00:43,  1.92it/s][A
     88%|████████▊ | 599/682 [05:30<00:43,  1.91it/s][A
     88%|████████▊ | 600/682 [05:31<00:41,  1.95it/s][A
     88%|████████▊ | 601/682 [05:31<00:41,  1.94it/s][A
     88%|████████▊ | 602/682 [05:32<00:41,  1.93it/s][A
     88%|████████▊ | 603/682 [05:32<00:41,  1.91it/s][A
     89%|████████▊ | 604/682 [05:33<00:40,  1.92it/s][A
     89%|████████▊ | 605/682 [05:33<00:40,  1.88it/s][A
     89%|████████▉ | 606/682 [05:34<00:40,  1.89it/s][A
     89%|████████▉ | 607/682 [05:34<00:39,  1.92it/s][A
     89%|████████▉ | 608/682 [05:35<00:37,  1.95it/s][A
     89%|████████▉ | 609/682 [05:35<00:37,  1.96it/s][A
     89%|████████▉ | 610/682 [05:36<00:36,  1.96it/s][A
     90%|████████▉ | 611/682 [05:37<00:36,  1.93it/s][A
     90%|████████▉ | 612/682 [05:37<00:36,  1.92it/s][A
     90%|████████▉ | 613/682 [05:38<00:35,  1.94it/s][A
     90%|█████████ | 614/682 [05:38<00:34,  1.96it/s][A
     90%|█████████ | 615/682 [05:39<00:34,  1.94it/s][A
     90%|█████████ | 616/682 [05:39<00:34,  1.94it/s][A
     90%|█████████ | 617/682 [05:40<00:34,  1.91it/s][A
     91%|█████████ | 618/682 [05:40<00:33,  1.92it/s][A
     91%|█████████ | 619/682 [05:41<00:32,  1.92it/s][A
     91%|█████████ | 620/682 [05:41<00:32,  1.90it/s][A
     91%|█████████ | 621/682 [05:42<00:32,  1.89it/s][A
     91%|█████████ | 622/682 [05:42<00:31,  1.88it/s][A
     91%|█████████▏| 623/682 [05:43<00:31,  1.89it/s][A
     91%|█████████▏| 624/682 [05:43<00:30,  1.88it/s][A
     92%|█████████▏| 625/682 [05:44<00:30,  1.87it/s][A
     92%|█████████▏| 626/682 [05:44<00:29,  1.89it/s][A
     92%|█████████▏| 627/682 [05:45<00:29,  1.89it/s][A
     92%|█████████▏| 628/682 [05:45<00:28,  1.92it/s][A
     92%|█████████▏| 629/682 [05:46<00:27,  1.94it/s][A
     92%|█████████▏| 630/682 [05:46<00:26,  1.94it/s][A
     93%|█████████▎| 631/682 [05:47<00:26,  1.90it/s][A
     93%|█████████▎| 632/682 [05:48<00:26,  1.91it/s][A
     93%|█████████▎| 633/682 [05:48<00:25,  1.91it/s][A
     93%|█████████▎| 634/682 [05:49<00:25,  1.92it/s][A
     93%|█████████▎| 635/682 [05:49<00:24,  1.88it/s][A
     93%|█████████▎| 636/682 [05:50<00:24,  1.90it/s][A
     93%|█████████▎| 637/682 [05:50<00:23,  1.90it/s][A
     94%|█████████▎| 638/682 [05:51<00:22,  1.92it/s][A
     94%|█████████▎| 639/682 [05:51<00:22,  1.92it/s][A
     94%|█████████▍| 640/682 [05:52<00:21,  1.95it/s][A
     94%|█████████▍| 641/682 [05:52<00:20,  2.00it/s][A
     94%|█████████▍| 642/682 [05:53<00:19,  2.00it/s][A
     94%|█████████▍| 643/682 [05:53<00:19,  2.00it/s][A
     94%|█████████▍| 644/682 [05:54<00:19,  1.99it/s][A
     95%|█████████▍| 645/682 [05:54<00:18,  2.00it/s][A
     95%|█████████▍| 646/682 [05:55<00:17,  2.03it/s][A
     95%|█████████▍| 647/682 [05:55<00:17,  1.97it/s][A
     95%|█████████▌| 648/682 [05:56<00:17,  1.94it/s][A
     95%|█████████▌| 649/682 [05:56<00:16,  1.96it/s][A
     95%|█████████▌| 650/682 [05:57<00:16,  2.00it/s][A
     95%|█████████▌| 651/682 [05:57<00:15,  1.95it/s][A
     96%|█████████▌| 652/682 [05:58<00:15,  1.98it/s][A
     96%|█████████▌| 653/682 [05:58<00:14,  1.94it/s][A
     96%|█████████▌| 654/682 [05:59<00:14,  1.92it/s][A
     96%|█████████▌| 655/682 [05:59<00:14,  1.91it/s][A
     96%|█████████▌| 656/682 [06:00<00:13,  1.93it/s][A
     96%|█████████▋| 657/682 [06:00<00:13,  1.90it/s][A
     96%|█████████▋| 658/682 [06:01<00:12,  1.90it/s][A
     97%|█████████▋| 659/682 [06:01<00:12,  1.88it/s][A
     97%|█████████▋| 660/682 [06:02<00:11,  1.90it/s][A
     97%|█████████▋| 661/682 [06:03<00:11,  1.85it/s][A
     97%|█████████▋| 662/682 [06:03<00:10,  1.86it/s][A
     97%|█████████▋| 663/682 [06:04<00:10,  1.85it/s][A
     97%|█████████▋| 664/682 [06:04<00:09,  1.87it/s][A
     98%|█████████▊| 665/682 [06:05<00:09,  1.88it/s][A
     98%|█████████▊| 666/682 [06:05<00:08,  1.89it/s][A
     98%|█████████▊| 667/682 [06:06<00:07,  1.91it/s][A
     98%|█████████▊| 668/682 [06:06<00:07,  1.89it/s][A
     98%|█████████▊| 669/682 [06:07<00:06,  1.89it/s][A
     98%|█████████▊| 670/682 [06:07<00:06,  1.89it/s][A
     98%|█████████▊| 671/682 [06:08<00:05,  1.90it/s][A
     99%|█████████▊| 672/682 [06:08<00:05,  1.90it/s][A
     99%|█████████▊| 673/682 [06:09<00:04,  1.86it/s][A
     99%|█████████▉| 674/682 [06:09<00:04,  1.85it/s][A
     99%|█████████▉| 675/682 [06:10<00:03,  1.85it/s][A
     99%|█████████▉| 676/682 [06:10<00:03,  1.88it/s][A
     99%|█████████▉| 677/682 [06:11<00:02,  1.91it/s][A
     99%|█████████▉| 678/682 [06:12<00:02,  1.90it/s][A
    100%|█████████▉| 679/682 [06:12<00:01,  1.91it/s][A
    100%|█████████▉| 680/682 [06:13<00:01,  1.85it/s][A
    100%|█████████▉| 681/682 [06:13<00:00,  1.86it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidYellowLeft.mp4 
    
    CPU times: user 7min 44s, sys: 1.42 s, total: 7min 45s
    Wall time: 6min 15s
    [MoviePy] >>>> Building video test_videos_output/challenge.mp4
    [MoviePy] Writing video test_videos_output/challenge.mp4


    
      0%|          | 0/251 [00:00<?, ?it/s][A
      0%|          | 1/251 [00:01<05:01,  1.21s/it][A
      1%|          | 2/251 [00:02<05:23,  1.30s/it][A
      1%|          | 3/251 [00:03<05:20,  1.29s/it][A
      2%|▏         | 4/251 [00:05<05:13,  1.27s/it][A
      2%|▏         | 5/251 [00:06<05:08,  1.25s/it][A
      2%|▏         | 6/251 [00:07<05:04,  1.24s/it][A
      3%|▎         | 7/251 [00:08<05:05,  1.25s/it][A
      3%|▎         | 8/251 [00:10<05:01,  1.24s/it][A
      4%|▎         | 9/251 [00:11<04:58,  1.24s/it][A
      4%|▍         | 10/251 [00:12<04:56,  1.23s/it][A
      4%|▍         | 11/251 [00:13<04:56,  1.23s/it][A
      5%|▍         | 12/251 [00:15<04:55,  1.24s/it][A
      5%|▌         | 13/251 [00:16<04:54,  1.24s/it][A
      6%|▌         | 14/251 [00:17<04:54,  1.24s/it][A
      6%|▌         | 15/251 [00:18<04:52,  1.24s/it][A
      6%|▋         | 16/251 [00:20<04:50,  1.24s/it][A
      7%|▋         | 17/251 [00:21<04:50,  1.24s/it][A
      7%|▋         | 18/251 [00:22<04:49,  1.24s/it][A
      8%|▊         | 19/251 [00:23<04:47,  1.24s/it][A
      8%|▊         | 20/251 [00:24<04:46,  1.24s/it][A
      8%|▊         | 21/251 [00:26<04:45,  1.24s/it][A
      9%|▉         | 22/251 [00:27<04:45,  1.25s/it][A
      9%|▉         | 23/251 [00:28<04:50,  1.27s/it][A
     10%|▉         | 24/251 [00:30<04:50,  1.28s/it][A
     10%|▉         | 25/251 [00:31<04:50,  1.28s/it][A
     10%|█         | 26/251 [00:32<04:54,  1.31s/it][A
     11%|█         | 27/251 [00:34<04:56,  1.32s/it][A
     11%|█         | 28/251 [00:35<04:57,  1.33s/it][A
     12%|█▏        | 29/251 [00:36<04:58,  1.35s/it][A
     12%|█▏        | 30/251 [00:38<04:59,  1.36s/it][A
     12%|█▏        | 31/251 [00:39<05:03,  1.38s/it][A
     13%|█▎        | 32/251 [00:41<05:03,  1.39s/it][A
     13%|█▎        | 33/251 [00:42<05:02,  1.39s/it][A
     14%|█▎        | 34/251 [00:43<05:04,  1.40s/it][A
     14%|█▍        | 35/251 [00:45<05:05,  1.42s/it][A
     14%|█▍        | 36/251 [00:46<05:06,  1.43s/it][A
     15%|█▍        | 37/251 [00:48<05:07,  1.44s/it][A
     15%|█▌        | 38/251 [00:49<05:07,  1.44s/it][A
     16%|█▌        | 39/251 [00:51<05:09,  1.46s/it][A
     16%|█▌        | 40/251 [00:52<05:11,  1.48s/it][A
     16%|█▋        | 41/251 [00:54<05:11,  1.49s/it][A
     17%|█▋        | 42/251 [00:55<05:12,  1.50s/it][A
     17%|█▋        | 43/251 [00:57<05:16,  1.52s/it][A
     18%|█▊        | 44/251 [00:58<05:16,  1.53s/it][A
     18%|█▊        | 45/251 [01:00<05:18,  1.55s/it][A
     18%|█▊        | 46/251 [01:02<05:18,  1.55s/it][A
     19%|█▊        | 47/251 [01:03<05:18,  1.56s/it][A
     19%|█▉        | 48/251 [01:05<05:19,  1.58s/it][A
     20%|█▉        | 49/251 [01:06<05:22,  1.60s/it][A
     20%|█▉        | 50/251 [01:08<05:22,  1.60s/it][A
     20%|██        | 51/251 [01:10<05:24,  1.62s/it][A
     21%|██        | 52/251 [01:11<05:25,  1.63s/it][A
     21%|██        | 53/251 [01:13<05:28,  1.66s/it][A
     22%|██▏       | 54/251 [01:15<05:31,  1.68s/it][A
     22%|██▏       | 55/251 [01:17<05:33,  1.70s/it][A
     22%|██▏       | 56/251 [01:18<05:33,  1.71s/it][A
     23%|██▎       | 57/251 [01:20<05:31,  1.71s/it][A
     23%|██▎       | 58/251 [01:22<05:31,  1.72s/it][A
     24%|██▎       | 59/251 [01:23<05:30,  1.72s/it][A
     24%|██▍       | 60/251 [01:25<05:30,  1.73s/it][A
     24%|██▍       | 61/251 [01:27<05:32,  1.75s/it][A
     25%|██▍       | 62/251 [01:29<05:33,  1.76s/it][A
     25%|██▌       | 63/251 [01:31<05:33,  1.78s/it][A
     25%|██▌       | 64/251 [01:33<05:42,  1.83s/it][A
     26%|██▌       | 65/251 [01:35<05:47,  1.87s/it][A
     26%|██▋       | 66/251 [01:36<05:51,  1.90s/it][A
     27%|██▋       | 67/251 [01:39<05:55,  1.93s/it][A
     27%|██▋       | 68/251 [01:41<05:59,  1.97s/it][A
     27%|██▋       | 69/251 [01:43<06:10,  2.04s/it][A
     28%|██▊       | 70/251 [01:45<06:27,  2.14s/it][A
     28%|██▊       | 71/251 [01:48<06:38,  2.21s/it][A
     29%|██▊       | 72/251 [01:50<06:48,  2.28s/it][A
     29%|██▉       | 73/251 [01:53<07:00,  2.36s/it][A
     29%|██▉       | 74/251 [01:55<07:15,  2.46s/it][A
     30%|██▉       | 75/251 [01:58<07:32,  2.57s/it][A
     30%|███       | 76/251 [02:01<07:48,  2.67s/it][A
     31%|███       | 77/251 [02:04<08:08,  2.80s/it][A
     31%|███       | 78/251 [02:07<08:32,  2.96s/it][A
     31%|███▏      | 79/251 [02:11<08:54,  3.11s/it][A
     32%|███▏      | 80/251 [02:14<08:58,  3.15s/it][A
     32%|███▏      | 81/251 [02:17<08:56,  3.16s/it][A
     33%|███▎      | 82/251 [02:20<08:53,  3.16s/it][A
     33%|███▎      | 83/251 [02:23<08:44,  3.12s/it][A
     33%|███▎      | 84/251 [02:27<08:44,  3.14s/it][A
     34%|███▍      | 85/251 [02:30<08:34,  3.10s/it][A
     34%|███▍      | 86/251 [02:33<08:27,  3.08s/it][A
     35%|███▍      | 87/251 [02:36<08:22,  3.06s/it][A
     35%|███▌      | 88/251 [02:38<08:04,  2.97s/it][A
     35%|███▌      | 89/251 [02:41<07:48,  2.89s/it][A
     36%|███▌      | 90/251 [02:44<07:27,  2.78s/it][A
     36%|███▋      | 91/251 [02:46<07:13,  2.71s/it][A
     37%|███▋      | 92/251 [02:49<06:58,  2.63s/it][A
     37%|███▋      | 93/251 [02:51<06:39,  2.53s/it][A
     37%|███▋      | 94/251 [02:53<06:18,  2.41s/it][A
     38%|███▊      | 95/251 [02:55<05:56,  2.29s/it][A
     38%|███▊      | 96/251 [02:57<05:40,  2.20s/it][A
     39%|███▊      | 97/251 [02:59<05:21,  2.09s/it][A
     39%|███▉      | 98/251 [03:01<05:10,  2.03s/it][A
     39%|███▉      | 99/251 [03:03<05:00,  1.97s/it][A
     40%|███▉      | 100/251 [03:05<04:56,  1.97s/it][A
     40%|████      | 101/251 [03:06<04:52,  1.95s/it][A
     41%|████      | 102/251 [03:08<04:51,  1.96s/it][A
     41%|████      | 103/251 [03:10<04:51,  1.97s/it][A
     41%|████▏     | 104/251 [03:12<04:48,  1.96s/it][A
     42%|████▏     | 105/251 [03:14<04:46,  1.97s/it][A
     42%|████▏     | 106/251 [03:16<04:49,  2.00s/it][A
     43%|████▎     | 107/251 [03:19<04:52,  2.03s/it][A
     43%|████▎     | 108/251 [03:21<04:57,  2.08s/it][A
     43%|████▎     | 109/251 [03:23<05:03,  2.13s/it][A
     44%|████▍     | 110/251 [03:25<05:06,  2.17s/it][A
     44%|████▍     | 111/251 [03:28<05:08,  2.20s/it][A
     45%|████▍     | 112/251 [03:30<05:08,  2.22s/it][A
     45%|████▌     | 113/251 [03:32<05:08,  2.24s/it][A
     45%|████▌     | 114/251 [03:34<05:07,  2.24s/it][A
     46%|████▌     | 115/251 [03:37<05:07,  2.26s/it][A
     46%|████▌     | 116/251 [03:39<05:05,  2.26s/it][A
     47%|████▋     | 117/251 [03:41<05:05,  2.28s/it][A
     47%|████▋     | 118/251 [03:44<05:06,  2.30s/it][A
     47%|████▋     | 119/251 [03:46<05:16,  2.40s/it][A
     48%|████▊     | 120/251 [03:49<05:28,  2.50s/it][A
     48%|████▊     | 121/251 [03:52<05:35,  2.58s/it][A
     49%|████▊     | 122/251 [03:54<05:38,  2.62s/it][A
     49%|████▉     | 123/251 [03:57<05:40,  2.66s/it][A
     49%|████▉     | 124/251 [04:00<05:43,  2.71s/it][A
     50%|████▉     | 125/251 [04:03<05:48,  2.77s/it][A
     50%|█████     | 126/251 [04:06<06:01,  2.89s/it][A
     51%|█████     | 127/251 [04:10<06:20,  3.07s/it][A
     51%|█████     | 128/251 [04:13<06:31,  3.18s/it][A
     51%|█████▏    | 129/251 [04:16<06:34,  3.23s/it][A
     52%|█████▏    | 130/251 [04:20<06:38,  3.29s/it][A
     52%|█████▏    | 131/251 [04:23<06:43,  3.36s/it][A
     53%|█████▎    | 132/251 [04:27<06:47,  3.43s/it][A
     53%|█████▎    | 133/251 [04:31<06:49,  3.47s/it][A
     53%|█████▎    | 134/251 [04:34<07:01,  3.60s/it][A
     54%|█████▍    | 135/251 [04:38<07:09,  3.71s/it][A
     54%|█████▍    | 136/251 [04:42<07:05,  3.70s/it][A
     55%|█████▍    | 137/251 [04:46<06:56,  3.65s/it][A
     55%|█████▍    | 138/251 [04:49<06:44,  3.58s/it][A
     55%|█████▌    | 139/251 [04:52<06:36,  3.54s/it][A
     56%|█████▌    | 140/251 [04:56<06:35,  3.56s/it][A
     56%|█████▌    | 141/251 [04:59<06:26,  3.51s/it][A
     57%|█████▋    | 142/251 [05:03<06:17,  3.47s/it][A
     57%|█████▋    | 143/251 [05:06<06:10,  3.43s/it][A
     57%|█████▋    | 144/251 [05:09<06:02,  3.39s/it][A
     58%|█████▊    | 145/251 [05:13<05:52,  3.33s/it][A
     58%|█████▊    | 146/251 [05:16<05:42,  3.26s/it][A
     59%|█████▊    | 147/251 [05:19<05:31,  3.18s/it][A
     59%|█████▉    | 148/251 [05:22<05:23,  3.14s/it][A
     59%|█████▉    | 149/251 [05:25<05:15,  3.09s/it][A
     60%|█████▉    | 150/251 [05:27<05:00,  2.98s/it][A
     60%|██████    | 151/251 [05:30<04:44,  2.84s/it][A
     61%|██████    | 152/251 [05:32<04:26,  2.69s/it][A
     61%|██████    | 153/251 [05:35<04:09,  2.55s/it][A
     61%|██████▏   | 154/251 [05:37<03:58,  2.46s/it][A
     62%|██████▏   | 155/251 [05:39<03:50,  2.40s/it][A
     62%|██████▏   | 156/251 [05:41<03:38,  2.30s/it][A
     63%|██████▎   | 157/251 [05:43<03:29,  2.23s/it][A
     63%|██████▎   | 158/251 [05:45<03:19,  2.15s/it][A
     63%|██████▎   | 159/251 [05:47<03:10,  2.07s/it][A
     64%|██████▎   | 160/251 [05:49<03:03,  2.02s/it][A
     64%|██████▍   | 161/251 [05:51<02:57,  1.98s/it][A
     65%|██████▍   | 162/251 [05:53<02:57,  2.00s/it][A
     65%|██████▍   | 163/251 [05:55<02:54,  1.99s/it][A
     65%|██████▌   | 164/251 [05:57<02:52,  1.98s/it][A
     66%|██████▌   | 165/251 [05:59<02:48,  1.96s/it][A
     66%|██████▌   | 166/251 [06:01<02:46,  1.96s/it][A
     67%|██████▋   | 167/251 [06:03<02:43,  1.95s/it][A
     67%|██████▋   | 168/251 [06:05<02:40,  1.94s/it][A
     67%|██████▋   | 169/251 [06:06<02:37,  1.92s/it][A
     68%|██████▊   | 170/251 [06:08<02:35,  1.92s/it][A
     68%|██████▊   | 171/251 [06:10<02:32,  1.90s/it][A
     69%|██████▊   | 172/251 [06:12<02:28,  1.88s/it][A
     69%|██████▉   | 173/251 [06:14<02:26,  1.87s/it][A
     69%|██████▉   | 174/251 [06:16<02:22,  1.85s/it][A
     70%|██████▉   | 175/251 [06:17<02:19,  1.83s/it][A
     70%|███████   | 176/251 [06:19<02:15,  1.81s/it][A
     71%|███████   | 177/251 [06:21<02:12,  1.79s/it][A
     71%|███████   | 178/251 [06:23<02:09,  1.77s/it][A
     71%|███████▏  | 179/251 [06:24<02:06,  1.75s/it][A
     72%|███████▏  | 180/251 [06:26<02:02,  1.73s/it][A
     72%|███████▏  | 181/251 [06:28<02:01,  1.73s/it][A
     73%|███████▎  | 182/251 [06:29<01:58,  1.71s/it][A
     73%|███████▎  | 183/251 [06:31<01:54,  1.68s/it][A
     73%|███████▎  | 184/251 [06:33<01:51,  1.67s/it][A
     74%|███████▎  | 185/251 [06:34<01:49,  1.67s/it][A
     74%|███████▍  | 186/251 [06:36<01:47,  1.66s/it][A
     75%|███████▍  | 187/251 [06:38<01:45,  1.66s/it][A
     75%|███████▍  | 188/251 [06:39<01:44,  1.66s/it][A
     75%|███████▌  | 189/251 [06:41<01:43,  1.67s/it][A
     76%|███████▌  | 190/251 [06:43<01:41,  1.66s/it][A
     76%|███████▌  | 191/251 [06:44<01:39,  1.66s/it][A
     76%|███████▋  | 192/251 [06:46<01:36,  1.63s/it][A
     77%|███████▋  | 193/251 [06:47<01:33,  1.62s/it][A
     77%|███████▋  | 194/251 [06:49<01:31,  1.61s/it][A
     78%|███████▊  | 195/251 [06:51<01:30,  1.61s/it][A
     78%|███████▊  | 196/251 [06:52<01:28,  1.60s/it][A
     78%|███████▊  | 197/251 [06:54<01:26,  1.60s/it][A
     79%|███████▉  | 198/251 [06:55<01:25,  1.62s/it][A
     79%|███████▉  | 199/251 [06:57<01:24,  1.62s/it][A
     80%|███████▉  | 200/251 [06:59<01:23,  1.64s/it][A
     80%|████████  | 201/251 [07:00<01:22,  1.65s/it][A
     80%|████████  | 202/251 [07:02<01:21,  1.66s/it][A
     81%|████████  | 203/251 [07:04<01:19,  1.67s/it][A
     81%|████████▏ | 204/251 [07:05<01:18,  1.66s/it][A
     82%|████████▏ | 205/251 [07:07<01:16,  1.66s/it][A
     82%|████████▏ | 206/251 [07:09<01:14,  1.66s/it][A
     82%|████████▏ | 207/251 [07:10<01:12,  1.65s/it][A
     83%|████████▎ | 208/251 [07:12<01:10,  1.65s/it][A
     83%|████████▎ | 209/251 [07:14<01:09,  1.66s/it][A
     84%|████████▎ | 210/251 [07:15<01:08,  1.66s/it][A
     84%|████████▍ | 211/251 [07:17<01:07,  1.69s/it][A
     84%|████████▍ | 212/251 [07:19<01:06,  1.70s/it][A
     85%|████████▍ | 213/251 [07:21<01:03,  1.68s/it][A
     85%|████████▌ | 214/251 [07:22<01:01,  1.67s/it][A
     86%|████████▌ | 215/251 [07:24<00:59,  1.66s/it][A
     86%|████████▌ | 216/251 [07:25<00:57,  1.64s/it][A
     86%|████████▋ | 217/251 [07:27<00:55,  1.64s/it][A
     87%|████████▋ | 218/251 [07:29<00:53,  1.63s/it][A
     87%|████████▋ | 219/251 [07:30<00:51,  1.62s/it][A
     88%|████████▊ | 220/251 [07:32<00:49,  1.61s/it][A
     88%|████████▊ | 221/251 [07:34<00:48,  1.62s/it][A
     88%|████████▊ | 222/251 [07:35<00:46,  1.60s/it][A
     89%|████████▉ | 223/251 [07:37<00:44,  1.60s/it][A
     89%|████████▉ | 224/251 [07:38<00:43,  1.60s/it][A
     90%|████████▉ | 225/251 [07:40<00:41,  1.59s/it][A
     90%|█████████ | 226/251 [07:41<00:39,  1.59s/it][A
     90%|█████████ | 227/251 [07:43<00:37,  1.58s/it][A
     91%|█████████ | 228/251 [07:45<00:36,  1.58s/it][A
     91%|█████████ | 229/251 [07:46<00:34,  1.58s/it][A
     92%|█████████▏| 230/251 [07:48<00:33,  1.59s/it][A
     92%|█████████▏| 231/251 [07:49<00:31,  1.60s/it][A
     92%|█████████▏| 232/251 [07:51<00:30,  1.59s/it][A
     93%|█████████▎| 233/251 [07:52<00:28,  1.57s/it][A
     93%|█████████▎| 234/251 [07:54<00:26,  1.56s/it][A
     94%|█████████▎| 235/251 [07:56<00:24,  1.55s/it][A
     94%|█████████▍| 236/251 [07:57<00:23,  1.55s/it][A
     94%|█████████▍| 237/251 [07:59<00:21,  1.55s/it][A
     95%|█████████▍| 238/251 [08:00<00:19,  1.53s/it][A
     95%|█████████▌| 239/251 [08:02<00:18,  1.53s/it][A
     96%|█████████▌| 240/251 [08:03<00:16,  1.54s/it][A
     96%|█████████▌| 241/251 [08:05<00:15,  1.58s/it][A
     96%|█████████▋| 242/251 [08:06<00:14,  1.58s/it][A
     97%|█████████▋| 243/251 [08:08<00:12,  1.60s/it][A
     97%|█████████▋| 244/251 [08:10<00:11,  1.60s/it][A
     98%|█████████▊| 245/251 [08:11<00:09,  1.60s/it][A
     98%|█████████▊| 246/251 [08:13<00:08,  1.62s/it][A
     98%|█████████▊| 247/251 [08:15<00:06,  1.62s/it][A
     99%|█████████▉| 248/251 [08:16<00:04,  1.62s/it][A
     99%|█████████▉| 249/251 [08:18<00:03,  1.62s/it][A
    100%|█████████▉| 250/251 [08:19<00:01,  1.62s/it][A
    100%|██████████| 251/251 [08:21<00:00,  1.62s/it][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/challenge.mp4 
    
    CPU times: user 9min 1s, sys: 2.39 s, total: 9min 4s
    Wall time: 8min 24s

