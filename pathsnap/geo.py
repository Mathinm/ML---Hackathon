"""Geometry helpers.

Lon/lat are angular, so every trip is projected into a local Azimuthal Equidistant
(AEQD) plane centred on the trip's GPS centroid. Distances near the centre are true
metres and distortion stays well below GPS noise for city/regional-scale trips.
"""
import numpy as np
from pyproj import Transformer


def make_projector(lon, lat):
    lon0, lat0 = float(np.mean(lon)), float(np.mean(lat))
    fwd = Transformer.from_crs(
        "EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", always_xy=True)
    return fwd


def project(fwd, lonlat):
    x, y = fwd.transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y])


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised distance from points to segments. Returns (dist, t in [0,1], proj_x, proj_y)."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = np.where(L2 > 0, ((px - ax) * vx + (py - ay) * vy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy), t, qx, qy


def bearing(dx, dy):
    """Compass-style angle in degrees [0,360) of a planar vector (x=east, y=north)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
