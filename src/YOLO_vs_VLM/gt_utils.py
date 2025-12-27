import numpy as np

def get_ground_truth_bbox(event, object_id):
    """
    Estrae la Bounding Box (GT) dai metadati del simulatore.
    Ritorna: [xmin, ymin, xmax, ymax] o None se non visibile.
    """
    if object_id not in event.instance_masks:
        return None
    
    mask = event.instance_masks[object_id]
    rows, cols = np.where(mask)
    
    if len(rows) == 0 or len(cols) == 0:
        return None

    # Calcolo coordinate (ymin, xmin, ymax, xmax) -> convertiamo in x, y, x, y
    ymin, ymax = np.min(rows), np.max(rows)
    xmin, xmax = np.min(cols), np.max(cols)
    
    # Restituisce formato [xmin, ymin, xmax, ymax]
    return [float(xmin), float(ymin), float(xmax), float(ymax)]