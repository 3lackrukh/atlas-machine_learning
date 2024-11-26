This folder contains files created for practicing transfer learning
using Keras imported from TensorFlow 2.15

Objectives: 
- Using one of the applications listed in Keras Applications
- Trained model saved in the current working directory as cifar10.h5
- Saved model should be compiled
- Saved model should have a validation accuracy of 87% or higher
- Script should not run when the file is imported

Saved model cifar10.h5 achieves validation accuracy of 94.48%

Folders V1, V2, V3, and FINAL_MODEL contain:
- a version numbered -transfer.py
    -- the version specific architecture and training scheme
- a saved model
    -- * saved models in these files may not be optimized
    -- * check full_model save implementation
    -- * confirm top_model is loaded from best weights before
         compilation with key_model into full_model
- 3 png results images from latest version configuration
    -- class_accuracy.png
    -- confusion_matrix.png
    -- training_history.png
- output.log
    -- outputs for various configurations
    -- * use to confirm configuration for images


transfer_learning folder also contains

- compact_wsl.sh
    -- a shell script for compressing wsl virtual disk if needed

- 3 jupytr notebooks
    -- images_by_category demonstrates cifar10 database is shuffled
    -- upscaling_comparison demonstrates upscaling techniques' performance
    -- preprocessing_scheme for testing preprocessing strategies