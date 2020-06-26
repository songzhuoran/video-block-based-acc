import arcpy, envipy

envipy.Initialize(arcpy)
from arcpy import env
env.workspace="/home/songzhuoran/video/video-block-based-acc/residual_img/"
rasters = arcpy.ListRasters("*", "png")
for raster in rasters:
    inraster = raster
    outraster= "/home/songzhuoran/video/video-block-based-acc/residual_img2/"+raster.strip(".png")+".tif"
    arcpy.ConvertRaster_envi(inraster, outraster, "TIFF","")
print("All have done")
