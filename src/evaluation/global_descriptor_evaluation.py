import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import faiss

def evaluate_global_descriptor(model, eval_set, batch_size=4, device="cuda", num_workers=8):
    """Evaluate a global descriptor model on a dataset
    Args:
        model (nn.Module): the global descriptor model to evaluate
        dataloader (DataLoader): the dataloader to use for evaluation
    Returns:
        float: the mean matching score over the dataset
    """

    test_data_loader = DataLoader(dataset=eval_set, shuffle=False, batch_size=batch_size,
                pin_memory=True, num_workers=num_workers)

    model.eval()
    pool_size = model.get_global_desc_dim()

    with torch.no_grad():
        pbar = tqdm(enumerate(test_data_loader, 0),
                    unit=' images',
                    unit_scale=batch_size,
                    total=len(test_data_loader),
                    smoothing=0,
                    disable=False)
        dbFeat = np.empty((len(eval_set), pool_size))
        for (i, (input, indices)) in pbar:
            img = input.to(device)
            B,C, H, W = img.shape

            out = model(img)
            out = model.post_processing(out, H, W)
            vlad = out['vlad']

            dbFeat[indices.numpy(),:] = vlad.data.squeeze().detach().cpu().numpy()

            pbar.set_description('Eval Global Descriptors')

    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')

    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    n_values = [1, 5, 10, 20]

    _, predictions = faiss_index.search(qFeat, max(n_values))

    gt = eval_set.getPositives()

    # correct_at_n = np.zeros(len(n_values))
    # for qIx, pred in enumerate(predictions):
    #     for i, n in enumerate(n_values):
    #         # if in top N then also in top NN, where NN > N
    #         if np.any(np.in1d(pred[:n], gt[qIx])):
    #             correct_at_n[i:] += 1
    #             break
    n_max = max(n_values)

    match_ratio_at_n = np.zeros(len(n_values))
    count_n = np.zeros(len(n_values))
    correct_hist = np.zeros(n_max)

    for qIx, pred in enumerate(predictions):
            # if in top N then also in top NN, where NN > N
            correct_matches = np.in1d(pred[:n_max], gt[qIx])
            total_matches = len(gt[qIx])
            match_idxs = np.where(correct_matches)
            if np.any(correct_matches):
                first_hit = match_idxs[0].min()
                correct_hist[first_hit:] += 1
            for i, n in enumerate(n_values):
                if total_matches > 0:
                    match_ratio_at_n[i] += sum(correct_matches[:n]) / min(total_matches, n)
                    count_n[i] += 1

    match_ratio_at_n = match_ratio_at_n / count_n
    recall_hist = correct_hist / eval_set.dbStruct.numQ
    #recall_at_n = correct_at_n / eval_set.dbStruct.numQ


    recalls = {}  # make dict for output
    auc = {}
    match_ratio = {}
    for i, n in enumerate(n_values):
        recalls[n] = recall_hist[n-1]
        auc[n] = np.sum(recall_hist[:n])/n
        match_ratio[n] = match_ratio_at_n[i]
        print("====> Recall@{}: {:.4f} AUC: {:.4f} MR: {:.4f}".format(n, recalls[n], auc[n], match_ratio[n]))
    return {'Recall': recalls, 'AUC': auc, 'MatchRatio': match_ratio}

