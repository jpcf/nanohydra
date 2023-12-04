import numpy as np

def vs_creator(X, Y, class_to_elim, method="onevsall"):
    if(method.lower() == "onevsall"):
        for y in Y:
            # Move class 
            np.putmask(y, y > class_to_elim-1, class_to_elim+1)
            np.putmask(y, y < class_to_elim,   0)
            np.putmask(y, y > 0,               1)
    elif(method.lower() == "allvsall_butone"):
        for y in Y:
            #X[ds] = np.delete(X[ds], y==class_to_elim, axis=0)
            y = np.delete(y, y==class_to_elim)
    else:
        assert False, f"Unknown Method Chosen: '{method}'"