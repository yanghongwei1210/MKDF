def calculateMMDDIS(domain1, domain2):
    import numpy as np
    from mmdxx import mmd
    import random
    random.seed(a=3)

    if len(domain1) > len(domain2):
        I = random.sample(range(0, len(domain1)), len(domain2))
        samplez = list(domain1[z] for z in I)
        domain1 = samplez
    else:
        I = random.sample(range(0, len(domain2)), len(domain1))
        corpusz = list(domain2[z] for z in I)
        domain2 = corpusz



    x1 = np.asarray(domain1)
    x2 = np.asarray(domain2)
    print("20",x1.shape)
    print(x2.shape)
    if x1.size <= x2.size:
        x2 = x2[:x1.shape[0], :]
    else:
        x1 = x1[:x2.shape[0], :]

    [sigma, value] = mmd(x1, x2, sigma=None, verbose=False)

    return abs(value)
