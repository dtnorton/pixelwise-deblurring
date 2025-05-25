# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:27:54 2024

@author: denni
"""

inputs = r"""
python deblur_main.py --matfile C:\Users\denni\OneDrive\AFRL\WESTAR-T38\pixelwise_deblurring\seq_02_sample.mat --tau 0.011 --n 7 --N 17 --lmd 0.5 --indices 1141 118 1151--output-prefix 

python deblur_main.py --matfile C:\Users\denni\OneDrive\AFRL\WESTAR-T38\pixelwise_deblurring\seq_02_sample.mat --tau 0.011 --n 7 --N 17 --lmd 0.5 --indices 1141 1146 1151 --output-prefix 

python deblur_main.py --matfile C:\Users\denni\OneDrive\AFRL\WESTAR-T38\pixelwise_deblurring\seq_02_sample.mat --tau 0.011 --n 7 --N 17 --lmd 0.5 --indices 1141 1146 1151 --output-prefix 

python deblur_main.py --matfile C:\Users\denni\OneDrive\AFRL\WESTAR-T38\pixelwise_deblurring\seq_02_sample.mat --tau 0.011 --n 7 --N 17 --lmd 0.5 --indices 1141 1146 1151 --output-prefix 
"""
python C: \Users\denni\OneDrive\Documents\GitHub\pixelwise-deblurring\src\deblur_main.py - -matfile C: \Users\denni\OneDrive\AFRL\WESTAR-T38\pixelwise_deblurring\seq_02_sample.mat - -tau 0.011 - -n 7 - -N 500 - -lmd 0.5 - -indices 0 1000 1500 2000 2500 - -roi 187 254 61 119 - -output-prefix bicycle_experiment

C:\Users\dtnor>python C:\Users\dtnor\OneDrive\Documents\GitHub\pixelwise-deblurring\src\deblur_main.py --matfile C:\Users\dtnor\OneDrive\AFRL\WESTAR-T38\pixelwise_deblurring\seq_02_sample.mat --tau 0.011 --n 7 --N 101 --T 0.005 --lmd 0.0001 --kappa 0.1 --indices 1130 1150 --output-prefix bicycle_experimentD