import numpy as np

def average_precision(gt, scores, threshold):

    def cumulative_sum(a):
        L = [a[0]]
        prefix_sum=a[0]
        for idx in range(1, len(a)):
            prefix_sum += a[idx]
            L.append(prefix_sum)
        return np.array(L)
    
    predictions = (scores>threshold).astype(int)
    correct_retrievals = np.logical_and(predictions, gt).astype(int)
    # print(f"gt: {gt}\npr: {predictions}\ncr: {correct_retrievals}\n")

    if correct_retrievals.sum() == 0:
        avg_prec = 0
    else:
        tot_correct_retreivals_upto_K = cumulative_sum(correct_retrievals)
        prec_at_k = (tot_correct_retreivals_upto_K*correct_retrievals)/(1+np.arange(len(gt)))
        avg_prec = prec_at_k.sum()/tot_correct_retreivals_upto_K[-1]
        assert tot_correct_retreivals_upto_K[-1] == correct_retrievals.sum()
        # print(f"pr: {prec_at_k}\napr: {avg_prec}\n, {correct_retrievals.sum(), tot_correct_retreivals_upto_K[-1]}")

    return avg_prec


def MTWV(gt, scores, thresholds, beta=0.1):

    def TWV():
        predictions = (scores>=threshold).astype(int)

        gt_pred = np.vstack((gt, predictions)).T
        u,c = np.unique(gt_pred, axis=0, return_counts=True)

        if np.equal(u, [1,0]).all(axis=1).any():
            miss_counts = c[np.equal(u, [1,0]).all(axis=1)][0]
            p_miss = miss_counts/gt_pred[:,0].sum()
        else:
            miss_counts = 0
            p_miss = 0.0

        if np.equal(u, [0,1]).all(axis=1).any():
            false_alarm_counts = c[np.equal(u, [0,1]).all(axis=1)][0]
            p_fa = false_alarm_counts/(len(gt_pred)-gt_pred[:,0].sum())
        else:
            false_alarm_counts = 0
            p_fa = 0
        
        TWV = 1-(p_miss + beta*p_fa)
        # print(threshold, miss_counts, false_alarm_counts, p_miss, p_fa, TWV)
        return TWV

    twv = []
    for threshold in thresholds:
        twv.append(TWV())

    mtwv = max(twv)
    return twv, mtwv


def MRR(gt, scores, threshold):
    pred = (scores > threshold).astype(int)
    indices = np.where((pred == 1) & (gt == 1))[0] + 1
    if indices.size > 0:
        mrr = 1/indices[0]
    else:
        mrr = 0
    return mrr