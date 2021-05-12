import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

names = ["ID", "Flow.ID", "Source.IP", "Source.Port", "Destination.IP",
    "Destination.Port", "Protocol", "Timestamp", "Flow.Duration",
    "Total.Fwd.Packets", "Total.Backward.Packets", "Total.Length.of.Fwd.Packets",
    "Total.Length.of.Bwd.Packets", "Fwd.Packet.Length.Max", "Fwd.Packet.Length.Min",
    "Fwd.Packet.Length.Mean", "Fwd.Packet.Length.Std", "Bwd.Packet.Length.Max",
    "Bwd.Packet.Length.Min", "Bwd.Packet.Length.Mean", "Bwd.Packet.Length.Std", "Flow.Bytes.s",
    "Flow.Packets.s", "Flow.IAT.Mean", "Flow.IAT.Std", "Flow.IAT.Max", "Flow.IAT.Min", "Fwd.IAT.Total",
    "Fwd.IAT.Mean", "Fwd.IAT.Std", "Fwd.IAT.Max", "Fwd.IAT.Min", "Bwd.IAT.Total", "Bwd.IAT.Mean", "Bwd.IAT.Std",
    "Bwd.IAT.Max", "Bwd.IAT.Min", "Fwd.PSH.Flags", "Bwd.PSH.Flags", "Fwd.URG.Flags", "Bwd.URG.Flags",
    "Fwd.Header.Length", "Bwd.Header.Length", "Fwd.Packets.s", "Bwd.Packets.s", "Min.Packet.Length",
    "Max.Packet.Length", "Packet.Length.Mean", "Packet.Length.Std", "Packet.Length.Variance",
    "FIN.Flag.Count", "SYN.Flag.Count", "RST.Flag.Count", "PSH.Flag.Count", "ACK.Flag.Count",
    "URG.Flag.Count", "CWE.Flag.Count", "ECE.Flag.Count", "Down.Up.Ratio", "Average.Packet.Size",
    "Avg.Fwd.Segment.Size", "Avg.Bwd.Segment.Size", "Fwd.Header.Length.1", "Fwd.Avg.Bytes.Bulk",
    "Fwd.Avg.Packets.Bulk", "Fwd.Avg.Bulk.Rate", "Bwd.Avg.Bytes.Bulk", "Bwd.Avg.Packets.Bulk",
    "Bwd.Avg.Bulk.Rate", "Subflow.Fwd.Packets", "Subflow.Fwd.Bytes", "Subflow.Bwd.Packets", "Subflow.Bwd.Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active.Mean", "Active.Std", "Active.Max", "Active.Min", "Idle.Mean", "Idle.Std", "Idle.Max", "Idle.Min", "Label"]

names_id = ["ID", "Cluster"]

df_raw = pd.read_csv('raw_data_remove1st.csv', names=names, low_memory=False)
df_cluster = pd.read_csv('cluster_remove.csv', names=names_id, low_memory=False)
df_cluster.drop('ID', axis=1, inplace=True)

print(df_raw.shape)
print(df_cluster.shape)
df_feature = pd.DataFrame()

df_raw['Flow.ID'] = df_raw['Flow.ID'].astype('category')
df_raw['Source.IP'] = df_raw['Source.IP'].astype('category')
df_raw['Destination.IP'] = df_raw['Destination.IP'].astype('category')
df_raw['Timestamp'] = df_raw['Timestamp'].astype('category')

cat_columns = df_raw.select_dtypes(['category']).columns
df_raw[cat_columns] = df_raw[cat_columns].apply(lambda x: x.cat.codes)

df_feature['Source.IP'] = df_raw['Source.IP']
df_feature['Source.Port'] = df_raw['Source.Port']
df_feature['Destination.IP'] = df_raw['Destination.IP']
df_feature['Destination.Port'] = df_raw['Destination.Port']
df_feature['Protocol'] = df_raw['Protocol']
df_feature['Init_Win_bytes_forward'] = df_raw['Init_Win_bytes_forward']
df_feature['Flow.ID'] = df_raw['Flow.ID']
df_feature['Protocol'] = df_raw['Protocol']
df_feature['min_seg_size_forward'] = df_raw['min_seg_size_forward']

print(df_raw.shape)

data = df_feature.to_numpy()
ansdata = df_cluster.to_numpy()
Y = ansdata[:,0]
X = data[:,0:9]
Y = Y.reshape(-1, 1)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sScaler = StandardScaler()
rescaleX = sScaler.fit_transform(X)
pca = PCA(n_components=2)
rescaleX = pca.fit_transform(rescaleX)
rescaleX = np.append(rescaleX, Y, axis=1)
principalDf = pd.DataFrame(data = rescaleX, columns = ['principal component 1', 'principal component 2', 'target'])

labels = ['one', 'two', 'three', 'four']

plt.clf()
plt.figure()

plt.title('KDD data set - Linear separability')
plt.xlabel('pc1')
plt.ylabel('pc2')

for i in range(len(labels)):
    bucket = principalDf[principalDf['target'] == i]
    bucket = bucket.iloc[:,[0,1]].values
    plt.scatter(bucket[:, 0], bucket[:, 1], label=labels[i]) 
plt.legend(loc='upper left',
           fontsize=8)

plt.show() 




