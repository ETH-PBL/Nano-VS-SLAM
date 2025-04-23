from src.evaluation.global_descriptor_evaluation import evaluate_global_descriptor
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py
from math import log10, ceil
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import warnings
warnings.filterwarnings("ignore")
import faiss

def get_clusters(model, cluster_set, initcache, nPerImage=100, threads=8, cacheBatchSize=32, device='cuda',
                 num_clusters=64,nDescriptors=50000, cacheDir='./cache'):
    
    
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=0, batch_size=cacheBatchSize, shuffle=False, 
                pin_memory=False,
                sampler=sampler)

    if not exists(join(cacheDir, 'centroids')):
        makedirs(join(cacheDir, 'centroids'))

    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            model = model.to(device)
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, model.encoder_dim], 
                        dtype=np.float32)

            for iteration, sample in enumerate(data_loader, 1):
            #for iteration, (input,_) in enumerate(data_loader, 1):
                input = sample['image'].to(device)
                #input = input.to(device)
                image_descriptors = model.only_encoder(input).view(input.size(0), model.encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/cacheBatchSize)), flush=True)
                del input, image_descriptors
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(model.encoder_dim, num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')
        
        
def init_netvlad(model, cluster_set, num_clusters=64, device='cuda', cacheDir='./cache'):
    initcache = join(cacheDir, 'centroids', 'Scene_parse_' + str(num_clusters) + '_desc_cen.hdf5')
    if exists(initcache):
        remove(initcache)
        
    get_clusters(model, cluster_set, initcache, device=device, num_clusters=num_clusters, cacheDir=cacheDir)
    
    with h5py.File(initcache, mode='r') as h5: 
        clsts = h5.get("centroids")[...]
        traindescs = h5.get("descriptors")[...]
        model.init_netvlad(clsts, traindescs)
        model = model.to(device)
        del clsts, traindescs
        
    remove(initcache)