# src/postprocess/geo.py
"""CRS helpers that do NOT use pyproj.

pyproj is blocked by an Application Control policy on this machine, so reprojection is done
with rasterio.warp instead (rasterio ships its own PROJ library).
"""
import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform as _warp_transform
from shapely.ops import transform as _shapely_transform

WGS84 = CRS.from_epsg(4326)


def utm_crs_for(lon, lat):
    """UTM zone (WGS84) that contains a lon/lat point, e.g. Abuja -> EPSG:32632."""
    zone = min(max(int((lon + 180) // 6) + 1, 1), 60)
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def metric_crs_for(crs, bounds):
    """A CRS in metres for the area: the raster's own CRS if projected, else its UTM zone."""
    if not crs.is_geographic:
        return crs  # already projected (UTM etc.), so units are metres
    lon = (bounds.left + bounds.right) / 2
    lat = (bounds.bottom + bounds.top) / 2
    return utm_crs_for(lon, lat)


def reproject_geom(geom, src_crs, dst_crs):
    """Reproject one shapely geometry between two rasterio CRS objects."""
    if src_crs == dst_crs:
        return geom

    def _f(x, y, *rest):
        xs, ys = _warp_transform(src_crs, dst_crs, list(x), list(y))
        return np.array(xs), np.array(ys)

    return _shapely_transform(_f, geom)