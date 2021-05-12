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

df_raw['Cluster'] = df_cluster['Cluster']
df_raw.drop('Label', axis=1, inplace=True)

df_raw['Flow.ID'] = df_raw['Flow.ID'].astype('category')
df_raw['Source.IP'] = df_raw['Source.IP'].astype('category')
df_raw['Destination.IP'] = df_raw['Destination.IP'].astype('category')
df_raw['Timestamp'] = df_raw['Timestamp'].astype('category')

df_raw.drop('Timestamp', axis=1, inplace=True)
df_raw.drop('Flow.ID', axis=1, inplace=True)
df_raw.drop('FIN.Flag.Count', axis=1, inplace=True)

cat_columns = df_raw.select_dtypes(['category']).columns
df_raw[cat_columns] = df_raw[cat_columns].apply(lambda x: x.cat.codes)

print(df_raw.shape)

data = df_raw.to_numpy()
Y = data[:,82]
X = data[:,0:5]
Y = Y.reshape(-1, 1)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sScaler = StandardScaler()
rescaleX = sScaler.fit_transform(X)
pca = PCA(n_components=2)
rescaleX = pca.fit_transform(rescaleX)
principalDf = pd.DataFrame(data = rescaleX, columns = ['principal component 1', 'principal component 2'])
data = principalDf.to_numpy()

plt.clf()
plt.figure()

plt.title('KDD data set - Linear separability')
plt.xlabel('pc1')
plt.ylabel('pc2')
#plt.scatter(principalDf.iloc[:,0], principalDf.iloc[:,1], s=50)
 
#plt.show()


kmeans = KMeans(n_clusters=4,random_state=0).fit(rescaleX)
y_kmeans = kmeans.predict(rescaleX)
plt.scatter(principalDf.iloc[:,0], principalDf.iloc[:,1], c=y_kmeans, s=50, cmap='viridis')


centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

from sklearn.metrics.cluster import adjusted_mutual_info_score
ans = df_cluster.to_numpy()
ans = ans.flatten()
kmeans_performance = adjusted_mutual_info_score ( ans,y_kmeans )
print("kmeans_performance: ")
print(kmeans_performance)
print("")


from sklearn.cluster import Birch
brc = Birch(n_clusters=4)
y_brc = brc.fit(rescaleX)
y_brc_ans = y_brc.predict(rescaleX)

birch_performance = adjusted_mutual_info_score ( ans,y_brc_ans )
print("Birch_performance: ")
print(birch_performance)
print("")


from sklearn.cluster import SpectralClustering

spectralA = SpectralClustering(n_clusters=4, affinity='rbf',
                           assign_labels='kmeans')
y_spectralA = spectralA.fit_predict(rescaleX)
performance2 = adjusted_mutual_info_score ( ans,y_spectralA )
print("SpectralA_performance: ")
print(performance2)
print("")


spectralA = SpectralClustering(n_clusters=4, affinity='rbf',
                           assign_labels='discretize')
y_spectralB = spectralA.fit_predict(rescaleX)
performance3 = adjusted_mutual_info_score ( ans,y_spectralB )
print("SpectralB_performance: ")
print(performance3)
print("")