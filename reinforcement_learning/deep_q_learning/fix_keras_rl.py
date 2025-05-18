#!/usr/bin/env python3
"""
Module to fix the compatibility issue between keras-rl2 and gymnasium
"""
import os
import rl

def apply_patch():
    """Apply the patch to fix keras-rl2 compatibility with keras"""
    rl_path = os.path.dirname(rl.__file__)
    callbacks_path = os.path.join(rl_path, 'callbacks.py')
    
    with open(callbacks_path, 'r') as file:
        content = file.read()
        
    fixed_content = content.replace(
        'from tensorflow.keras import __version__ as KERAS_VERSION',
        'from keras import __version__ as KERAS_VERSION'
    )
    
    with open(callbacks_path, 'w') as file:
        file.write(fixed_content)
        
    print("✓ keras-rl2 compatibility patch applied")

if __name__ == "__main__":
    apply_patch()