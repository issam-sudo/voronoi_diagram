import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
from optparse import OptionParser
import os

os.system('python3 crop.py')

#args
parser = OptionParser()
parser.add_option("-i", "--image", type="string", dest="imagename",
                  metavar="FILE", help="Name of Image")
parser.add_option("-c", "--count", type="int", dest="count",
                  default=0, help="amount of voronoi-points")
parser.add_option("-l", "--color", type="string", dest="color",
                  help="type of color")
parser.add_option("-g", "--grid", type="string", dest="grid",
                  help="type of grid")
parser.add_option("-e", "--epaisseur", type="int", dest="epaisseur",
                default=1 ,help="epaisseur cellule")
(options, args) = parser.parse_args()


img_path = options.imagename


type_color = options.color
type_grid = options.grid
epaisseur = options.epaisseur
if (type_color is None):
    print("No color type given")
    quit()
if (img_path is None):
    print("No image file given")
    quit()
img = cv2.imread(img_path)
num_cells = int(options.count)
if (num_cells is None):
    print("No amount of cells given")
    quit()
avg_colors =[]

avg_colors =[(40.5825 ,40.117 ,100.6975 ,0),(45.0725 ,45.4625, 101.74 , 0) ,(50 ,80, 100 , 0),(101.0725 ,120.4625, 120.74 , 0),(60,60,60,0),(99,99,99,0)]
avg_colors =np.array(avg_colors)

def scale_points(points ):
    """
    scale the points to the size of the image
    """
    scaled_points = []
    for x, y  in points:
        
        x = x * img.shape[1]
        y = y * img.shape[0]
        scaled_points.append([x, y])

    return scaled_points

def generate_voronoi_diagram(img):
    if (type_grid == "random"):
        points = np.random.rand(num_cells, 2)
        points = scale_points(points )
        keyPoints =np.array(points)
    elif (type_grid =="const"):
        brief = cv2.xfeatures2d.SIFT_create(num_cells)
        kp = brief.detect(img,None)
        keyPoints = cv2.KeyPoint_convert(kp)    
 
    points =[]
    for keyPoint in keyPoints: 
        points.append((keyPoint[0], keyPoint[1]))

    size = img.shape
    print(size)
    subdiv2DShape = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(subdiv2DShape);
   
    for p in points :
        subdiv.insert(p)

    return subdiv

def drawVoronoi(img, subdiv): 
    
    voronoi = np.zeros(img.shape, dtype = img.dtype)
    (facets, centers) = subdiv.getVoronoiFacetList([])
    
     
    for facetsIndex in range(0,len(facets),1):
    
        # Generate array of polygon corners
        facetArray = []
        
        for facet in facets[facetsIndex] :
        
            facetArray.append(facet)
 
        # Get average color of polygon from original image
        mask = np.zeros(img.shape[:2], np.uint8)
        
        cv2.fillPoly(mask, np.int32([facetArray]), (255,255,255));
       
        color = cv2.mean(img, mask)

       
        distance_matrix = np.linalg.norm(color - avg_colors, axis=1)
        idx = np.argmin(distance_matrix)
        
        r =  avg_colors[idx][0]
        g =  avg_colors[idx][1]
        b =  avg_colors[idx][2]
        a =  avg_colors[idx][3]
        # Fill polygon with average color
        intFacet = np.array(facetArray, np.int)
        if (type_color =="All"):
            cv2.fillConvexPoly(voronoi, intFacet ,color);
        elif type_color =="pallete":
            cv2.fillConvexPoly(voronoi, intFacet ,(r,g,b,a));
        else:
            cv2.fillConvexPoly(voronoi, intFacet ,(255255,255,255,255));
     
        # Draw lines around polygon
        polyFacets = np.array([intFacet])
        cv2.polylines(voronoi, polyFacets, True, (0, 0, 0), epaisseur, cv2.LINE_AA, 0) 
        
    return voronoi

path, imagename = os.path.split(img_path)
imagename = imagename.split(".")[0]    
voronoi_diagram = drawVoronoi(img, generate_voronoi_diagram(img))
cv2.imwrite(imagename + "-voronoi.jpg",voronoi_diagram)


