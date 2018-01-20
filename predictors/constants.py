import os

class FLAGS:
    lr = 5e-3
    n_classes = 2  # Outputs a binary mask
    dropout = None
    batch_size = 1
    num_epochs = 50
    # Early stopping is true
    image_x = image_y = 570
    image_c = 4
    
    models_dir = "../models"