from matplotlib import pyplot as plt
import numpy as np
from utils.utils import load_json
import pandas as pd
def create_latex_table(data, metrics):
    str = ''
    for i,m in enumerate(metrics):
        line = '& {} & {:.3f}& {:.3f}& {:.3f} \\\\ \n'.format(m, data[0][i],data[1][i],data[2][i]).replace('%', '\%')
        str += line
    return str

def KP2Dtiny_Plot(ax, data, names):


    labels = ['Repeatability [%]', 'Localization error [pixel%]', 'd1 [%]', 'd3 [%]', 'd5 [%]', 'MScore [%]']
    width = 0.1
    colors = ['#BD1F21', '#DD2C2F', '#00397a', '#057af0', '#008000', '#1f991f']
    hatches = ['///', 'xxx', 'ooo', '+++', '.ooo', '...']
    edgecolor = 'white'
    offsets = [-2.7, -1.7, -0.5, 0.5, 1.7, 2.7]
    x = np.arange(len(labels))
    for i in range(len(data)):
        ax.bar(x + width * offsets[i], data[i], width, label=names[i], color=colors[i], hatch=hatches[i],
               edgecolor=edgecolor)
    # rects1 = ax.bar(x - width * 2.7, data[0], width, label=names[0], color='#BD1F21', hatch=hatches[0],
    #                 edgecolor=edgecolor)
    # rects2 = ax.bar(x - width * 1.7,  data[1], width, label=names[1], color='#DD2C2F', hatch=hatches[1],
    #                 edgecolor=edgecolor)
    # rects3 = ax.bar(x - width * 0.5,  data[2], width, label=names[2], color='#00397a', hatch=hatches[2],
    #                 edgecolor=edgecolor)
    # rects4 = ax.bar(x + width * 0.5,  data[3], width, label=names[3], color='#057af0', hatch=hatches[3],
    #                 edgecolor=edgecolor)
    # rects5 = ax.bar(x + width * 1.7,  data[4], width, label=names[4], color='#008000', hatch=hatches[4],
    #                 edgecolor=edgecolor)
    # rects6 = ax.bar(x + width * 2.7,  data[5], width, label=names[5], color='#1f991f', hatch=hatches[5],
    #                 edgecolor=edgecolor)

    # Set the x-axis tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Low Resolution (88x88)")

def parse_keypoint_data(data):
    metrics = ['repeatability', 'localization', 'c1', 'c3', 'c5', 'mscore']
    results = []
    for m in metrics:
        results.append(data[m])
    auc = [data['auc']["1"],data['auc']["3"],data['auc']["5"]]
    return results, auc


def AUC_Plot(precision, recall):
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def get_keypoints_results(data, name, quantized = False):
    keys = ['120x160', '240x320', '480x640']
    sub_keys = ["keypoints_top1000", "keypoints_top300"]
    metrics = ['repeatability', 'localization', 'c1', 'c3', 'c5', 'mscore']
    results = []
    for key in keys:
        if key not in data:
            continue
        for sub_key in sub_keys:
            if sub_key not in data[key]:
                continue
            k= sub_key.replace('keypoints_top', '')
            out = {'name': name, 'res': key, 'k': k, 'quantized': quantized}
            for m in metrics:
                out[m] = data[key][sub_key][m]
            for auc_key in ['1', '3', '5']:
                out['auc_'+ auc_key] = data[key][sub_key]['auc'][auc_key]
            results.append(out.copy())
    return results

def get_segmentation_results(data, name, quantized = False):
    keys = ['120x160', '240x320', '480x640']
    sub_keys = ["segmentation"]
    metrics = ['IoU', 'IoU_macro', 'accuracy', 'f1']
    results = []
    for key in keys:
        if key not in data:
            continue
        for sub_key in sub_keys:
            if sub_key not in data[key]:
                continue
            out = {'name': name, 'res': key, "quantized": quantized}
            for m in metrics:
                out[m] = data[key][sub_key][m]

            results.append(out.copy())
    return results

def get_visloc_results(data, name, quantized = False):
    keys = ['120x160', '240x320', '480x640']

    sub_keys = ["visloc"]
    metrics = ['1', '5', '10', '20']
    ms = ['AUC', 'MatchRatio', 'Recall']
    results = []
    for key in keys:
        if key not in data:
            continue
        for sub_key in sub_keys:
            if sub_key not in data[key]:
                continue
            out = {'name': name, 'res': key, 'quantized': quantized}
            for n in ms:
                for m in metrics:
                    out[ n + "_"+m] = data[key][sub_key][n][m]

            results.append(out.copy())
    return results

def get_visual_odometry_results(data, name, quantized = False):
    keys = ['128x256', '128x512', '256x1024']
    results = []
    for k in keys:
        key = "visual_odometry_" + k
        if key not in data:
            continue
        out = {'name': name, 'res': key, "quantized": quantized}
        for m in data[key].keys():
            out[m] = data[key][m]
        results.append(out.copy())
    return results

from glob import glob
from pathlib import Path
#V3_N = load_json("V3_N_Results.json")
#tiny_A_CS = load_json("results_tiny_A_CS.json")
#tiny_A = load_json("results_tiny_A.json")
#result_files = glob("./results_iou/*/results_*.json")
result_files = glob("./results_quantized/*/results_*.json")
#result_files.extend(glob("./trained_checkpoints_results/*/results_*.json"))
model_results = [(Path(f).name[8:-12], load_json(f)) for f in result_files]



keypoints_results = []
segmentation_results = []
visloc_results = []
visual_odometry_results = []
model_info =[]

for model in model_results:
    name, d = model
    data = d['results']
    model_info.append({'name': name, 'global_desc_dim': d['info']['global_desc_dim'],
                       'params': d['info']['params'],
                       'epoch': d['info']['epoch'],
                       'dataset': d['info']['dataset'],
                       "quantized": d["info"].get("quantized", False)})
    quantized = d["info"].get("quantized", False)
    keypoints_results.extend(get_keypoints_results(data, name,  quantized))


    try:
        segmentation_results.extend(get_segmentation_results(data, name,  quantized))
    except:
        pass
    visloc_results.extend(get_visloc_results(data, name,  quantized))

    try:
        visual_odometry_results.extend(get_visual_odometry_results(data, name,  quantized))
    except:
        pass


keypoints_df = pd.DataFrame(keypoints_results)
segmentation_df = pd.DataFrame(segmentation_results)
visloc_df = pd.DataFrame(visloc_results)
visual_odometry_df = pd.DataFrame(visual_odometry_results)
model_info_df = pd.DataFrame(model_info)

seg = segmentation_df[segmentation_df['res']=='120x160']
vo = visual_odometry_df[visual_odometry_df['res']=='visual_odometry_256x1024']

visloc_df = visloc_df[visloc_df['res']=='120x160']

kp = keypoints_df[keypoints_df['res']=='120x160']
kp = kp[kp['k']=='300']

kp = kp.merge(vo, on="name")
kp = kp.merge(seg, on="name")
kp = kp.merge(model_info_df, on="name")
kp = kp.merge(visloc_df, on="name", suffixes=('_vo', '_visloc'))
print(keypoints_df)
print(segmentation_df)
print(visloc_df)
print(visual_odometry_df)


# making latex tables for keypoints


rename_dict = {'repeatability': 'rep',
               'localization': 'loc err',
                'mscore': 'ms', 'auc_1': 'auc1', 'auc_3': 'auc3', 'auc_5': 'auc5'}



for res in ['120x160', '240x320', '480x640']:
    for k in ['300', '1000']:
        caption= "Keypoint results for {} resolution and {} keypoints".format(res, k)

        filtered_df = keypoints_df[(keypoints_df['res']==res) & (keypoints_df['k']==k)]

        tab = filtered_df.rename(columns = rename_dict).drop(['res', 'k'], axis=1).to_latex(index=False,float_format="%.2f", caption=caption).replace('_', '\_')
        print(tab)


for res in ['120x160', '240x320', '480x640']:

    caption= "Segmentation results for {} resolution ".format(res)

    filtered_df = segmentation_df[segmentation_df['res']==res]

    tab = filtered_df.drop(['res'], axis=1).to_latex(index=False,float_format="%.2f", caption=caption).replace('_', '\_')
    print(tab)

# making latex tables for visual odometry


rename_dict = {}



for res in ['visual_odometry_256x1024', 'visual_odometry_128x512', 'visual_odometry_128x256']:

    caption= "Visual Odometry results for {} resolution".format(res[-7:])

    filtered_df = visual_odometry_df[visual_odometry_df['res']==res]

    tab = filtered_df.rename(columns = rename_dict).drop(['res', 'min','std'], axis=1).to_latex(index=False,float_format="%.2f", caption=caption).replace('_', '\_')
    print(tab)