import nibabel as nib
import cv2 as cv
import numpy as np

##########################################
#   - Lung Segmentation on a CT scan
#
#   This project is meant to read the chest CT scan
#   which is in .nii or .nii.gz format and then
#   find the area of lungs on it. It does that using various 
#   colors, the idea is to colors 3 different parts with different colors
#   and then just count all of the pixel of a certain color
#   The problem was that air in the lungs and air outside of the body
#   were both black so main challenge was how do differentiate
#   between those two areas.
##########################################



# Colors can NOT be repeat
# for different types
class colors:
    air = (0, 0, 255) # it wont be visible at the end result
    tissue = (80, 80, 200)
    non_body = (60, 60, 60)
    lungs = (240, 160, 160)

# THRESHOLD
# controls from which value is pixel considered tissue and from which air
# if pixel is under the threshold it will be considered air and above as tissue
# if its value is above 90 body will be full of random spikes, not smooth
# if its under 30 a lot of lung area will be considered tissue, also full of spikes
threshold = 40 

# JUMP_SIZE
# controls what amount of the pixel can be skipped for it to still be considered same color
# so if its set to 20 that means that if i see a lot of red and then up to 20 pixel
# of some other color i will considered it red too, its used so some small objects can be filled
# for example, blood vessels and patients bed... its also in the CT and in order to be removed
# jump size needs TO BE OVER 10 and also UNDER 25!
jump_size = 15

def lungDetection(file_path):
    """
    This function takes path of a chest CT file (in .nii.gz or .nii)
    And then from it, extracts lungs and calculates there area.

    The function returns 3 values:
        - Original image
        - Colored image on which can be seen lungs
        - Area of the lungs in mm2
    """

    # Loads the CT scan file using nibabel librarie
    # Because library has some problems with return type
    # of function load i have to set it manually (DataobjImage class)
    file: nib.DataobjImage = nib.load(file_path) 
    
    # CT scan file are composed of headers and the actual images
    # so to extract image data get_fdata func is used
    raw_image = file.get_fdata() 

    # In order for image to be turned into opencv Mat object it
    # first needs to be normalized. This is done by subtracting 
    # every pixel in the image by the minimum value that is present on the image.
    # This way we get that minimal value of the pixel is 0.
    # Then its divided by difference of maximum pixel value and minimum
    # That makes it so that all of the pixels are between 0 and 1 of value, 
    # then they are multiplied by 255 to represent intensity of a pixel using a single byte
    # At the end values are converted to unsigned 8-bit intigers
    normalized = (raw_image - np.min(raw_image)) / (np.max(raw_image) - np.min(raw_image))
    normalized *= 255
    normalized = normalized.astype(np.uint8)

    # Now that normalized image data is converted to opencv's Mat object
    image = cv.cvtColor(normalized, cv.COLOR_GRAY2BGR)
    original = image.copy() # Copy the image now so we can return unmodified version too

    # Images dimensions are loaded
    height, width, _ = image.shape

    # This variable will be used to store coordinates of the last
    # black pixel that has been seen in the current row
    last = (0, 0) # (x, y)

    # Using this for loop, program is going row by row, pixel by pixel
    for y in range(height):
        for x in range(width):
            # If color of the pixel is less then the threshold color
            if all(image[y, x] < threshold):
                # Set pixel to air color
                image[y, x] = colors.air

                # If its the same row as last air pixel and
                # If difference between last black pixel and current black pixel is
                # less then jump_size then it should color be colored too
                if (y == last[1]) and (x - last[0] < jump_size):
                    
                    # For each pixel that was in the gap set air color
                    for i in range(1, x-last[0]):
                        image[y, last[0]+i] = colors.air
                
                # Then it sets the last to current pixel coordinates
                last = (x, y)
            else: 
                # If image was above the threshold then make it tissue color
                image[y, x] = colors.tissue
    
    # Now both air in the lungs and air outside of the body are colors the same
    # and other organs (like hart and ribs) are tissue color

    # Display the current image (if you want to current results)
    # cv.imshow('Air & Tissue', image)
    
    # This goes over each pixel again to distinguishes the air from outside
    # of the body from the air that is inside of the lungs
    for y in range(height):

        # This stores all of the edges (points on which color changes)
        # If this is a image row (- is air and = is tissue):
        # -------======---------====--------
        # This would be edges: (1, 2, 3, 4)
        # -------1=====2--------3===4-------
        # Its places where color changes
        edges = []

        # This goes over each pixel in the row
        for x in range(width):
            # If current pixels color is not equal the the previous
            # pixels color then it adds the current coordinates to edges
            if(x != 0 and not all(image[y, x] == image[y, x-1])):
                edges.append((x, y))
        
        # There was a small error that few pixels near the end of the row
        # would make a bug so if pixel of the last edge are tissue color
        # then just remove that edge, its a FAKE EDGE
        if len(edges) > 0 and all(image[y, edges[-1][0]] == colors.tissue):
            edges.pop()
        

        fill = True

        # Iterate over each pixel of the row again
        # It will be filling everything before the first edge and everything after the last edge
        for x in range(width):
            if len(edges) > 0:
                # If first edge is passed turn the fill Off
                if x == edges[0][0]: fill = False
                
                # If the last edge is passed turn the fill on again
                if x == edges[-1][0]: fill = True
            
            # If fill is true make the pixel non body color
            if fill: image[y, x] = colors.non_body
    
    area = 0
    # It goes over each pixel AGAIN and if its air color 
    # then that means that that's air inside the lungs
    # since the one that was outside was colored differently
    # in the last for loop
    for y in range(height):
        for x in range(width):
            # Checks if pixel is air color
            if all(image[y, x] == colors.air):
                # Add that pixel to lungs area
                area += 1
                # Color it into lungs color
                image[y, x] = colors.lungs

    # Now area in in pixels and needs to be converted to mm2
    voxel_dimensions = file.header.get_zooms() # This loads the zoom levels from the header
    voxel_area = voxel_dimensions[0] * voxel_dimensions[1] # The its area its calculated
    area = area * voxel_area # Then real are ration is multiplied by pixel area

    return original, image, area



original, lungs, area = lungDetection("./src/slice016.nii.gz")
cv.imshow('Original CT', original)
cv.imshow('Segmented CT', lungs)
print("Lung area:", area)
cv.waitKey(0)