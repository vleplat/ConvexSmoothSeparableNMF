# CSSNMF - Convex Smooth Separable NMF
We present a new convex model for the Smooth Separable NMF

This MATLAB software reproduces the results from the following paper:

```
@article{Pan_Gillis_leplat2025,
      title={A provably-correct convex model for smooth separable NMF}, 
      author={Junjun Pan and Valentin Leplat and Michael Ng and Nicolas Gillis},
      year={2024},
      journal={arXiv preprint arXiv:XYZZ.00007} 
}
```
See <INSERT ADDRESS> 

## Acknowledgements

The baseline algorithms used in the manuscript are courtesy of their respective authors.


## Content
 
 - /Libraries : contains helpful libaries; in particular
   - Libraries/FGNSR-master/ contains the code of Fast Gradient method for Nonnegative Sparse Regression proposed in Ref[1] - https://github.com/rluce/FGNSR

   - Libraries/smoothed-separable-nmf-main/ contains the code of smooth separable NMF proposed in Ref[2] -  https://gitlab.com/nnadisic/smoothed-separable-nmf
 
 - /Datasets : contains test data sets.

 - /Utils : contains helpful files and MatLab routines to run the demos.
   
 - /Methods: contains the MatLab implementations of the Algorithms 1 and 2 developped in the paper.
 - /Results: all the results can be downloaded from https://www.dropbox.com/scl/fo/w3jc5x54oivzrs8m41j2s/ADH7gnCR3RMrYxKuyDowtP4?rlkey=gz3zxjgtwt55j4zjhh4f0xmf8&st=2wbztn45&dl=0 

 - test files detailed in next Section

ATTENTION: to use our method, first download the FGNSR MATLAB framework, compile the mex files for your computer. Finally, copy paste the file "fgnsr_alg1_copy_to_move_later.m" currently in "/Methods" to ./Libraries/FGNSR-master/matlab, and rename it in "fgnsr_alg1.m"
   
## Test files
 
 Test files are available. To proceed, open and start one of the following files:
 
- test_script_Synthetic_Diri.m : run demo for Synthetic test with M = W*[H0 H1] + N, where H0 has disjoint row supports and H1 follows a Dirichlet Distribition, see Section 6.1 of the paper. 
- test_script_Synthetic_Outliers.m : run demo for the comparison of aggregation techniques, that is mean vs median, see Section 6.2 of the paper. 
- test_script_Synthetic_Middle.m : run additional test, dubbed as Middle points and adversarial noise, check the content of the file to get the details.
- test_script_HSI_XX.m : run demo for Hyperspectral Image Unmixing for three data sets XX={JasperRidge, Urban, Samson}, section 6.2 of the paper.

## References

[1]: Gillis, N., & Luce, R. (2018). A Fast Gradient Method for Nonnegative Sparse Regression With Self-Dictionary. IEEE Transactions on Image Processing, 27(1), 24–37. \\

[2]: Nadisic, N.,  Gillis, N., Kervazo, C. Smoothed Separable Nonnegative Matrix Factorization. Linear Algebra and Its Applications 676, pp. 174-204, 2023. 
