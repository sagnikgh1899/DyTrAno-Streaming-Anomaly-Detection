"""
List of all the constants used throughout the program
"""
OUTLIER_DETECTION_ENABLED = False  # Outlier_Detection
CLUSTERING_ENABLED = True  # Clustering
DISPLAY_DATA_POINT_STATS = True


NUMBER_OF_NEIGHBORS = 50  # This is K
CLUSTERING_ALGORITHM = 'ball_tree'  # 'kd_tree'
SIGMA = 1e-9  # Small multiple for regularization, adjust as needed
DATASET_NAME = "Corners"
DELTA = 0.7 # Threshold for density change
BETA = 0.7 # Decay constant for EwMA, can be adjusted
DISPLAY_DENSITY = False
NEIGHBORHOOD_CONTRIBUTION_DIFFERENCE = 0.3  # 0.6 #0.3 #0.5 #0.2 #0.05 #0.1 # Epsilon
DELTA_FOR_FILTRATION = 0.2
