# ICL_cirrus_project

## Image generation:

### cluster_metadata.fits 
includes all of the information for the halo_ids we use (mass, redshift, position, total flux etc)

### main_v2_rotation.py 

is used to generate the cluster images (it also calls functions from: galaxy_class.py, icl.py, utils.py) 
for each halo_id we include at the end of the script: 
- it loads the flagship file for the specific cluster 
- finds the BCG (galaxy closest to the center out of 3 brightest)
- creates the coordinate system
- makes the 5 versions of randomized angle galaxy distributions (those are saved as new coordinate arrays for the galaxy list)
- makes the stamp for each galaxy and populates the 5 canvases accordingly
- saves the galaxy canvases (5) and position files (5)
- exactly the same for galactic halos
- calculates PA and ellipticity of satellite galaxies (5 versions according to new galaxy positions distribution), applies it to the ICL, makes profile
- saves ICL canvases (5), as well as the e and PA values

### final_image_construction.py

is then used to compose the individual galaxies, halos and ICL and also add background and noise (also cirrus in the future)
- it applies necessary image corrections (to undo flux normalization we had originally done to plot with galsim and also adjust flux to pixelscale)
- adds component images
- image augmentation (new images, flip H, flip V, rotate by 90deg)
- gets random background from galaxy_random_canvases_500.fits
- makes noise and cuts the image centers
- masks sources and saves different masked images (diff input types) and the masks themselves (in case we need them for future prediction-target comparison)
- clips everything below detection limit
- saves everything

the above were used to create the three sets of images: training_set.fits, validation_set.fits, mini_training_set.fits\
(mini is part of the full training set, while the validation set is completely independent (no shared clusters or variations of them with any of the training sets))



## Training:

### data -- mini_training_set.fits is the one that was used for the initial models training so far
### code -- unet_test.ipynb was used for the training and result evaluation. It includes:
- Basic UNet Architecture (test, NOT USED)
- CMB Reconstructed Model (test, NOT USED)
- ICL Cirrus Model (USED HERE), includes:
  -model
  -loading input data as different input types (select input mode depending on what we want to train on)
  - 'trial run' trains for 600 epochs and saves the model every 200, keeps track of training and validation loss
  - the above are plotted then, as well as some examples from the validation set
  - opening models for evaluation after saving them, plotted some results
  - comparison 
  - resuming training (i haven't yet used it, i was always training in one go, but it should be ok)

*** also note that the plotting, comparisons etc are still a bit messy, and some of them may not be 100% correct (ex. comparing a model results with validation of wrong input type)\
(i was carefull about the ones i used in the presentations but the notebook is still a bit messy so there might be old tests still there:)

The results of the training (5 models for 5 different input modes) can be directly opened using the notebook without training again.\
They are named according to input mode and #of epochs (ex. model_extended_keep_bcg_600.pth)

    
                      
                

  
    
