import numpy as np

def IDGI(Gradients, Predictions):
    """
    IDGI algorithm:
    
    The IDGI is compatible with any IG based method, e.g., Integrated gradients (IG), Guided Integrated gradients (GIG), Blur Integrated gradients (BlurIG), ....
    For more detail, please check our paper: 
    Args:
        Gradients (list of np.array or np.array): All the gradients that are computed from the Integraded gradients path.
                                                  For instance, when compute IG, the gradients are needed for each x_j on the path. e.g. df_c(x_j)/dx_j.
                                                  Gradients is the list (or np.array) which contains all the computed gradients for the IG-base method, 
                                                  and each element in Gradients is the type of np.array.
        Predictions (list of float or np.array): List of float numbers.
                                                 Predictions contains all the predicted value for each points on the path of IG-based methods.
                                                 For instance, the value of f_c(x_j) for each x_j on the path.
                                                 Predictions is the list (or np.array) which contains all the computed target values for IG-based method, 
                                                 and each element in Predictions is a float.
    
    Return:
        IDGI result: Same size as the gradient, e.g., Gradients[0]
    """
    assert len(Gradients) == len(Predictions)
    
    idgi_result = np.zeros_like(Gradients[0])
    for i in range(len(Gradients) -1):
        # We ignore the last gradient, e.g., the gradient of on the original image, since IDGI requires the prediction difference component, e.g., d.
        d = Predictions[i+1] - Predictions[i]
        element_product = Gradients[i]**2
        idgi_result += element_product*d/np.sum(element_product)
    return idgi_result