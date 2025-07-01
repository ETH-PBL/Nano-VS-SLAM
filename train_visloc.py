import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import numpy as np
import h5py
from math import ceil
import wandb
from tqdm import tqdm
import faiss

from os.path import join, exists
from os import makedirs, remove
import warnings, os, argparse


import src.data.pittsburgh as dataset
from src.evaluation.global_descriptor_evaluation import evaluate_global_descriptor
from utils.utils import load_checkpoint, save_checkpoint, set_seed, load_json

from src.kp2dtiny.models.kp2d_former import KeypointFormer
from src.kp2dtiny.models.kp2dtiny import KP2DTinyV2, get_config, KP2DTinyV3

warnings.filterwarnings("ignore")


def args_parse():
    parser = argparse.ArgumentParser(description="Train NetVLAD with triplet loss")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--out_model_path",
        type=str,
        default="checkpoint.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--cacheRefreshRate",
        type=int,
        default=1000,
        help="How often to refresh cache, in number of queries. 0 for off",
    )
    parser.add_argument(
        "--cacheBatchSize",
        type=int,
        default=32,
        help="Batch size for caching and testing",
    )
    parser.add_argument(
        "--nEpochs", type=int, default=10, help="number of epochs to train for"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="number of threads for data loader to use",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="margin for triplet loss (default: 0.1)",
    )
    parser.add_argument(
        "--cacheDir",
        type=str,
        default="./cache",
        help="path to cache folder for storing precomputed descriptor pool",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=28,
        help="Number of classes in the training set",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="datasets.json",
        help="Path to dataset config file",
    )
    parser.add_argument("--seed", type=int, default=42069, help="Random seed")
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone"
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=64,
        help="Number of NetVlad clusters. Default=64",
    )
    parser.add_argument("--im_h", type=int, default=120, help="Image height")
    parser.add_argument("--im_w", type=int, default=160, help="Image width")
    parser.add_argument(
        "--large_netvlad", action="store_true", help="larger netvlad head"
    )
    parser.add_argument(
        "--model_type", type=str, default="KeypointNet", help="Model type"
    )
    parser.add_argument(
        "--tiny_v2",
        action="store_true",
        help="Use tiny v2 which has a different segmentation head",
    )
    parser.add_argument("--config", type=str, default="S", help="Model config")
    parser.add_argument(
        "--to_mcu", action="store_true", help="Convert model to MCU format"
    )
    parser.add_argument("--depth", action="store_true", help="Depth of the network")
    return parser.parse_args()


def get_clusters(model, args, cluster_set, initcache, nPerImage=100):
    nDescriptors = 50000
    nIm = ceil(nDescriptors / nPerImage)

    sampler = SubsetRandomSampler(
        np.random.choice(len(cluster_set), nIm, replace=False)
    )
    data_loader = DataLoader(
        dataset=cluster_set,
        num_workers=args.threads,
        batch_size=args.cacheBatchSize,
        shuffle=False,
        pin_memory=False,
        sampler=sampler,
    )

    if not exists(join(args.cacheDir, "centroids")):
        makedirs(join(args.cacheDir, "centroids"))

    with h5py.File(initcache, mode="w") as h5:
        with torch.no_grad():
            model.eval()
            model = model.to(args.device)
            print("====> Extracting Descriptors")
            dbFeat = h5.create_dataset(
                "descriptors", [nDescriptors, model.encoder_dim], dtype=np.float32
            )

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(args.device)
                image_descriptors = (
                    model.only_encoder(input)
                    .view(input.size(0), model.encoder_dim, -1)
                    .permute(0, 2, 1)
                )

                batchix = (iteration - 1) * args.cacheBatchSize * nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(
                        image_descriptors.size(1), nPerImage, replace=False
                    )
                    startix = batchix + ix * nPerImage
                    dbFeat[startix : startix + nPerImage, :] = (
                        image_descriptors[ix, sample, :].detach().cpu().numpy()
                    )

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print(
                        "==> Batch ({}/{})".format(
                            iteration, ceil(nIm / args.cacheBatchSize)
                        ),
                        flush=True,
                    )
                del input, image_descriptors

        print("====> Clustering..")
        niter = 100
        kmeans = faiss.Kmeans(
            model.encoder_dim, model.num_clusters, niter=niter, verbose=False
        )
        kmeans.train(dbFeat[...])

        print("====> Storing centroids", kmeans.centroids.shape)
        h5.create_dataset("centroids", data=kmeans.centroids)
        print("====> Done!")


def train(epoch, args):
    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging
    log_freq_loss = 500

    subsetN = ceil(len(train_set) / args.cacheRefreshRate)
    # TODO randomise the arange before splitting?
    subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)

    nBatches = (len(train_set) + args.batch_size - 1) // args.batch_size
    train_set.cache = join(args.cacheDir, train_set.whichSet + "_feat_cache.hdf5")

    for subIter in range(subsetN):
        print("====> Building Cache")
        model.eval()
        with h5py.File(train_set.cache, mode="w") as h5:
            pool_size = model.get_netvlad_dim()
            h5feat = h5.create_dataset(
                "features", [len(whole_train_set), pool_size], dtype=np.float32
            )
            pbar = tqdm(
                enumerate(whole_training_data_loader, 1),
                total=len(whole_training_data_loader),
                unit_scale=args.cacheBatchSize,
            )
            with torch.no_grad():
                for iteration, (input, indices) in pbar:
                    img = input.to(args.device)
                    B, C, H, W = img.shape
                    out = model(img)
                    out = model.post_processing(out, H, W)
                    vlad_encoding = out["vlad"]
                    h5feat[indices.detach().numpy(), :] = (
                        vlad_encoding.detach().cpu().numpy()
                    )
                    pbar.set_description(
                        "====> Extracting Features {}/{}".format(
                            iteration, len(whole_training_data_loader)
                        )
                    )
                    del input, vlad_encoding, out

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(
            dataset=sub_train_set,
            num_workers=8,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
        )

        model.train()
        if args.freeze_backbone:
            model.freeze_backbone()

        pbar = tqdm(
            enumerate(training_data_loader, 1),
            total=len(training_data_loader),
            unit_scale=args.batch_size,
        )
        for iteration, (query, positives, negatives, negCounts, indices) in pbar:
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])

            out = model(input.to(args.device))
            out = model.post_processing(out, H, W)
            vlad_encoding = out["vlad"]

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(
                        vladQ[i : i + 1], vladP[i : i + 1], vladN[negIx : negIx + 1]
                    )

            loss /= nNeg.float().to(
                args.device
            )  # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives
            del out
            batch_loss = loss.item()
            epoch_loss += batch_loss
            pbar.set_description(
                (
                    "==> Epoch[{}][{}]({}/{}): Loss: {:.4f}".format(
                        epoch, subIter, iteration, nBatches, batch_loss
                    )
                )
            )

            if iteration % log_freq_loss == 0:
                wandb.log({"loss/vlad_loss": batch_loss})
        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()  # delete HDF5 cache


if __name__ == "__main__":
    args = args_parse()
    set_seed(args.seed)
    size = (args.im_h, args.im_w)
    dataset_config = load_json(args.dataset_config)

    # if args.model_type == 'vgg16':
    #     model = VGG16Net(pretrained=True,encoder_dim = 512, num_clusters=64, vladv2=False)
    #     alternative_norm = False
    # elif args.model_type == 'KeypointNet':
    #     model = KeypointNet(device=args.device, nClasses=args.n_classes, num_clusters=args.num_clusters, large_netvlad=args.large_netvlad)
    #     alternative_norm = True
    # elif args.model_type == 'KP2Dtiny':
    #     model = KeypointNetRaw(**KP2D_TINY, nClasses=args.n_classes, large_netvlad=args.large_netvlad, v2_seg=args.tiny_v2, device=args.device)
    #     alternative_norm = True
    if args.model_type == "KeypointFormer":
        model = KeypointFormer(num_classes=args.n_classes, device=args.device)
        alternative_norm = True
    elif args.model_type == "KP2DtinyV2":
        alternative_norm = True
        conf = get_config(args.config, to_mcu=args.to_mcu)
        model = KP2DTinyV2(
            **conf, nClasses=args.n_classes, mem_efficient=True, depth=args.depth
        )
    elif args.model_type == "KP2DtinyV3":
        conf = get_config(args.config, v3=True, to_mcu=args.to_mcu)
        model = KP2DTinyV3(
            **conf, nClasses=args.n_classes, mem_efficient=True, depth=args.depth
        )
        alternative_norm = True
    else:
        raise ValueError("Invalid model type")
    if args.freeze_backbone:
        model.freeze_backbone()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    config = {
        "input_args": vars(args),
        "dataset_config": dataset_config,
        "size": size,
        "script": os.path.basename(__file__),
        "model_info": model.gather_info(),
    }
    wandb.init(config=config, project="Masterthesis-test", entity="thomacdebabo")

    whole_test_set = dataset.get_whole_val_set(
        dataset_config["pittsburgh_data_path"], size, alternative_norm=alternative_norm
    )
    whole_train_set = dataset.get_whole_training_set(
        dataset_config["pittsburgh_data_path"], size, alternative_norm=alternative_norm
    )
    whole_training_data_loader = DataLoader(
        dataset=whole_train_set,
        batch_size=args.cacheBatchSize,
        shuffle=False,
        pin_memory=True,
        num_workers=args.threads,
    )

    train_set = dataset.get_training_query_set(
        dataset_config["pittsburgh_data_path"],
        size,
        margin=args.margin,
        alternative_norm=alternative_norm,
    )

    history = None
    if args.model_path:
        state_dict, optimizer_state, history = load_checkpoint(
            args.model_path, optimizer_key="optimizer_visloc"
        )
        model.load_state_dict(state_dict, strict=False)

        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    model = model.cuda()

    initcache = join(
        args.cacheDir,
        "centroids",
        whole_train_set.dataset + "_" + str(args.num_clusters) + "_desc_cen.hdf5",
    )
    if exists(initcache):
        remove(initcache)

    cluster_set = dataset.get_whole_training_set(
        dataset_config["pittsburgh_data_path"], size, onlyDB=True
    )
    get_clusters(model, args, cluster_set, initcache)

    with h5py.File(initcache, mode="r") as h5:
        clsts = h5.get("centroids")[...]
        traindescs = h5.get("descriptors")[...]
        if model.global_descriptor_method == "netvlad":
            model.init_netvlad(clsts, traindescs)
        model = model.to(args.device)
        del clsts, traindescs

    remove(initcache)
    if args.freeze_backbone:
        model.freeze_backbone()
    wandb.watch(model, log_freq=100, log="all")
    criterion = torch.nn.TripletMarginLoss(
        margin=args.margin**0.5, p=2, reduction="sum"
    ).to(args.device)
    evaluate_global_descriptor(
        model, whole_test_set, device=args.device, num_workers=args.threads
    )

    best_score = 0
    for i in range(args.nEpochs):
        train(i, args)

        results = evaluate_global_descriptor(
            model, whole_test_set, device=args.device, num_workers=args.threads
        )

        wandb.log({"val/": {"visloc": results}})
        print(results)
        checkpoint = {
            "epoch": i + 1,
            "state_dict": model.state_dict(),
            "optimizer_visloc": optimizer.state_dict(),
            "config": config,
            "results": {"VisLoc": results},
            "history": history,
        }
        save_checkpoint(checkpoint, args.out_model_path)

        # if results[5] > best_score:
        #     best_score = results[5]
        #     save_checkpoint(checkpoint, "best_model_vlad.ckpt")
