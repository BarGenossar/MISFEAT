import os 
import pandas as pd
import matplotlib.pyplot as plt


data_name = 'startup'
data_path = f'./RealWorldData/{data_name}'
data = pd.read_pickle(f'{data_path}/dataset.pkl')
# data_path = f'./data'
# data = pd.read_csv(f'{data_path}/{data_name}.csv')

if not os.path.exists(f"{data_path}/EDA"):
    os.makedirs(f"{data_path}/EDA")

## if plot subgroup
gids = [gid for gid in range(data['subgroup'].nunique())]
features = list(data.columns)
features.remove('y')
features.remove('subgroup')

fig, ax = plt.subplots(len(gids), len(features))
for col_header, feat in zip(ax[0], features):
    col_header.set_title(feat)
for gid, row_header in enumerate(ax[:, 0]):
    row_header.set_ylabel(f"g{gid}", rotation=0, size='large')

for gid in gids:
    print(f"Subgroup {gid}")
    data_0 = data[data['subgroup'] == gid]
    print(f"number of records: {len(data_0)}")

    for idx, col in enumerate(features):
        print(len(set(data_0[col])))
        ax[gid, idx].hist(data_0[col])
        ax[gid, idx].set_xticks([])
        ax[gid, idx].set_yticks([])
# plt.tight_layout()
plt.savefig(f'{data_path}/EDA/dist.png')
plt.close()


###
# gids = [gid for gid in range(data['subgroup'].nunique())]
# features = list(data.columns)
# features.remove('y')
# features.remove('subgroup')

# fig, ax = plt.subplots(len(gids), len(features))
# for col_header, feat in zip(ax[0], features):
#     col_header.set_title(feat)
# for gid, row_header in enumerate(ax[:, 0]):
#     row_header.set_ylabel(f"g{gid}", rotation=0, size='large')

# for gid in gids:
#     print(f"Subgroup {gid}")
#     data_0 = data[data['subgroup'] == gid]
#     print(f"number of records: {len(data_0)}")

#     for idx, col in enumerate(features):
#         print(len(set(data_0[col])))
#         ax[gid, idx].hist(data_0[col])
#         ax[gid, idx].set_xticks([])
#         ax[gid, idx].set_yticks([])
# plt.tight_layout()
# plt.savefig(f'{data_path}/EDA/dist.png')
plt.close()