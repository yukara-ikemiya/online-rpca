Online Robust PCA
=================

Batch and Online Robust PCA (Robust Principal Component Analysis) implementation and examples (Python).

Robust PCA based on Principal Component Pursuit (**RPCA-PCP**) is the most popular RPCA algorithm which decomposes the observed matrix M into a low-rank matrix L and a sparse matrix S by solving Principal Component Pursuit:

> \min ||L||_* + \lambda ||S||_1

> s.t. L + S = M

where ||.||_* is a nuclear norm, ||.||_1 is L1-norm. 

Folder rpca contains various batch and online Robust PCA algorithms.

  * pcp.py: Robust PCA based on Principal Component Pursuit (RPCA-PCP). 
    Reference: Candes, Emmanuel J., et al. "Robust principal component analysis." Journal of the ACM (JACM) 58.3 (2011): 11.

  * spca.py: Stable Principal Component Pursuit (Zhou et al., 2009). This implementation uses the Accelerated Proximal Gradient method with a fixed mu_iter.
    Reference: Zhou, Zihan, et al. "Stable principal component pursuit." Information Theory Proceedings (ISIT), 2010 IEEE International Symposium on. IEEE, 2010. 
        
        


