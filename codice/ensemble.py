import numpy as np

def weighted_majority (xs, ys, hypothesis, beta):
    #initialize the weights,
    weights = np.ones ((len (xs) + 1, len (hypothesis)))

    cum_loss = np.zeros (len (xs))

    predictions = np.ones (len(hypothesis))

    for i, x, y in zip(xrange (len(xs)), xs, ys):

        for j, h in enumerate(hypothesis):
            predictions[j] = h (x)

            #updating weight
            if predictions[j] == y:
                weights[i+1, j] = weights[i, j]
            else:
                weights[i+1, j] = weights[i, j] * beta

        #voting process
        pos_ind, neg_ind = (predictions  == 1), (predictions  == -1)

        pos_sum, neg_sum = np.sum (weights[i, pos_ind]), np.sum (weights[i, neg_ind])

        output = (1 if pos_sum > neg_sum else -1)

        if y == output: #correct prediction
            cum_loss[i] = (0 if i == 0 else cum_loss[i-1])
        else:#incorrect prediction
            cum_loss[i] = (1 if i == 0 else cum_loss[i-1] + 1)
            
    return weights, cum_loss