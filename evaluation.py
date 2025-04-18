from shapely.geometry.point import Point

def intersection_over_union(circ1_dict, circ2_dict):
    """
    Calculated the Intersection over the Union to give a measure of
    overlap between two circles specified via configs in dictionary. 
    You can use a threshold on this score to turn it into a measure of 
    "successful detection".
    """
    for key in ('row', 'col', 'radius'):
        try:
            assert key in circ1_dict and key in circ2_dict
        except AssertionError:
            raise ValueError(f"Must submit two dictionaries with the keys ('row','col','radius'). {key} not found in one circle dict.")

    shape1 = Point(circ1_dict['row'], circ1_dict['col']).buffer(circ1_dict['radius'])
    shape2 = Point(circ2_dict['row'], circ2_dict['col']).buffer(circ2_dict['radius'])

    return shape1.intersection(shape2).area / shape1.union(shape2).area

