"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import Point

def ref_grid(xmin=-180, ymin=-90, xmax=180, ymax=90, width=0.1, height=0.1):

    """
    Create a reference grid of rectangular polygons within a specified bounding box.

    Parameters:
    - xmin (float): The minimum X-coordinate (longitude) of the bounding box.
    - ymin (float): The minimum Y-coordinate (latitude) of the bounding box.
    - xmax (float): The maximum X-coordinate (longitude) of the bounding box.
    - ymax (float): The maximum Y-coordinate (latitude) of the bounding box.
    - width (float): The width of each grid cell in degrees.
    - height (float): The height of each grid cell in degrees.

    Returns:
    - grid (geopandas.GeoDataFrame): A GeoDataFrame containing rectangular polygons that form the reference grid.
    
    This function generates a reference grid of rectangular polygons covering the specified geographic area
    defined by the bounding box. The width and height parameters determine the size of each grid cell. The
    resulting grid is a GeoDataFrame with additional attributes, including the area of each cell and a unique
    identifier for each cell.

    Note:
    - The coordinate system for the input (xmin, ymin, xmax, ymax) is assumed to be in WGS84 (EPSG:4326).
    - The resulting grid will have its area calculated and is transformed to EPSG:6933 for area measurement.

    Example usage:
    grid = ref_grid(-180, -90, 180, 90, 1, 1)
    """

    rows = int(np.ceil((ymax-ymin) / height))
    cols = int(np.ceil((xmax-xmin) / width))
    num_of_cells = rows * cols
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax- height
    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom =YbottomOrigin
        for j in range(rows):
            polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)]))
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width
    grid = gpd.GeoDataFrame({'geometry':polygons}, crs = 4326)
    def getXY(pt):
        return (pt.x, pt.y)
    centroidseries = grid['geometry'].centroid
    grid['lon'],grid['lat'] = [list(t) for t in zip(*map(getXY, centroidseries))]
    List = list(range(1, (num_of_cells + 1)))
    string = 'poly'
    polygon_num = ["{}{}".format(string,i) for i in List]
    grid_area= grid['geometry'].to_crs(6933)
    grid['area'] = grid_area.area
    grid['polygon_num'] = polygon_num
    return grid