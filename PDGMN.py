from utils import *
from scipy.sparse import csr_matrix,save_npz,load_npz
from scipy.stats import norm
import time
import os
import networkx as nx
import warnings
from tqdm import tqdm as tqdm
warnings.filterwarnings("ignore")
#********************************** Step 1 ***************************************************
# Load PPI and GGA networks from files
#*********************************************************************************************
cancer='LUAD' #indicating the name of folder of the cancer dataset to be processed
# netx_name indicating the name of folder,
# for example, './results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name,nety_name,cancer,netx_name,k_name)
netx_name='PPIcheng'
nety_name='regnetwork' # indicating the name of folder
netx,nodesx=load_network('./data/network/PPI_cheng.txt')
nety,nodesy=load_network('./data/network/GGA_regnetwork.txt')
nodes_xy,ref_net1,ref_net2=intersec_net_x_y(netx,nety,nodesx,nodesy)

#********************************** Step 2 ***************************************************
# Construct the sample-specific multiplex networks for each patient by CSN method.
#*********************************************************************************************
datafile='./data/cancer/%s/RNA-seq.txt' % cancer
data_df=pd.read_table(filepath_or_buffer=datafile,sep='\t',header=0,index_col=0)
data_df['genename']=data_df.index.tolist()
data_df=pd.merge(left=nodes_xy,right=data_df,left_on='nodes',right_on='genename',how='inner')
data_df.set_index(["nodes"], inplace=True)
data_df=data_df.drop(['genename'],axis=1)

data=np.array(data_df)
samp_lst=data_df.columns.tolist()
gene_lst=data_df.index.tolist()
pd.DataFrame({'gene':gene_lst}).to_csv(path_or_buf='./data/cancer/%s/gene_lst_%s_%s.txt'%(cancer,netx_name,nety_name),sep='\t',header=False,index=False)

# Generate the adj matrices for ref_net1 and ref_net2
ng=len(gene_lst)
mtx_ref1=np.zeros([ng,ng])
mtx_ref2=np.zeros([ng,ng])
print('Processing with the reference network...')
print('This may take a few minutes...')
for i in np.arange(0,ng):
    g=gene_lst[i]
    targ1 = ref_net1[ref_net1['source'] == g]['target'].values.tolist()
    targ2 = ref_net2[ref_net2['source'] == g]['target'].values.tolist()
    targ1_idx = [i for i in range(ng) if gene_lst[i] in targ1]
    targ2_idx = [i for i in range(ng) if gene_lst[i] in targ2]
    tmp1 = np.zeros(ng)
    tmp2 = np.zeros(ng)
    tmp1[targ1_idx] = 1
    tmp2[targ2_idx] = 1
    mtx_ref1[i, :] = tmp1
    mtx_ref2[i, :] = tmp2

mtx_ref1=mtx_ref1+mtx_ref1.transpose() #To undirected
mtx_ref2=mtx_ref2+mtx_ref2.transpose() #To undirected
for i in np.arange(0,mtx_ref1.shape[0]):
    mtx_ref1[i,i]=mtx_ref1[i,i]/2
    mtx_ref2[i, i] = mtx_ref2[i, i] / 2
print('Done.')

# The function performs the transformation from gene expression matrix to
# cell-specific network (csn).
# data: Gene expression matrix, rows = genes, columns = cells/samples
# c: Construct the CSNs for all cells, set c = [] (Default);
#    Construct the CSN for cell k, set  c = k
# alpha: Significant level (eg. 0.001, 0.01, 0.05 ...)
#        larger alpha leads to more edges, Default = 0.01
# boxsize: Size of neighborhood, Default = 0.1
# weighted: 1  edge is weighted
#           0  edge is not weighted (Default)
# csn: Cell-specific network, the kth CSN is in csn{k}
#      rows = genes, columns = genes
#  Too many cells or genes may lead to out of memory.
print('********************************************************************')
print('Starting to construct sample-specific multiplex networks for each sample')
print('Processing...')
print('********************************************************************')
weighted=0
boxsize=0.1
alpha=0.01

[n1,n2]=data.shape
c=np.arange(0,n2)

# Define the neighborhood of each plot
upper=np.zeros([n1,n2])
lower=np.zeros([n1,n2])

for i in np.arange(0,n1):
    s1=np.sort(data[i,:])
    s2=np.argsort(data[i,:])
    n3=n2-np.count_nonzero(s1)
    h=round(boxsize/2*np.count_nonzero(s1))
    k=1
    while k<=n2:
        s=0
        while (k+s+1 <= n2) and (s1[k+s+1-1] == s1[k-1]):
            s = s+1

        if s>=h:
            upper[i,s2[k-1:k+s]] = data[i,s2[k-1]]
            lower[i,s2[k-1:k+s]] = data[i,s2[k-1]]
        else:
            upper[i,s2[k-1:k+s]] = data[i,s2[min(n2-1,k+s+h-1)]]
            lower[i,s2[k-1:k+s]] = data[i,s2[max(n3*int(n3>h)+1-1,k-h-1)]]

        k=k+s+1

# Construction of cell-specific network
csn = {}
B=np.zeros([n1,n2])
p=-norm.ppf(alpha)

time_start = time.time()
for k in c:
    k_name = samp_lst[k]
    if os.path.exists('./results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name, nety_name, cancer, netx_name, k_name)) and os.path.exists('./results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name,nety_name,cancer,nety_name, k_name)):
        print('%s.npz already exists.' % (k_name))
        continue

    for j in np.arange(0,n2):
        B[:,j]=np.logical_and(data[:,j]<=upper[:,k] , data[:,j]>=lower[:,k]).astype(int)
    a = B.sum(axis=1)
    a.resize((len(a), 1))
    d = (np.dot(B,B.T)*n2-np.dot(a,a.T))/np.sqrt( np.dot(a,a.T)*np.dot(n2-a,(n2-a).T)/(n2-1) + np.spacing(1))#加上eps：np.spacing(1)防止分母为0
    d[np.diag_indices_from(d)]=0

    csn_k=(d > p).astype(int)
    spr_csn_k_ref1=csr_matrix(np.logical_and(csn_k,mtx_ref1).astype(int))
    spr_csn_k_ref2=csr_matrix(np.logical_and(csn_k,mtx_ref2).astype(int))

    save_npz('./results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name,nety_name,cancer,netx_name,k_name), spr_csn_k_ref1 )
    save_npz('./results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name,nety_name,cancer,nety_name, k_name), spr_csn_k_ref2)

    time_end = time.time()
    print('Sample %d is completed----time cost: %d s' % (k,time_end - time_start))

time_end = time.time()
print('Total time cost: %d s' % (time_end - time_start))

#********************************** Step 3 ***************************************************
# Integrate somatic mutation data to generate weighted sample-specific multiplex networks;
# Then, identify personalized cancer driver genes from each sample-specific multiplex network
# by applying the weighted MVC set identification algorithm.
#*********************************************************************************************
samp_df=pd.read_table(filepath_or_buffer='./data/cancer/%s/sample_lst.txt' % (cancer),sep='\t',header=None,index_col=None,names=['sample'])
samp_lst=samp_df['sample'].values.tolist()
mut_df=pd.read_table(filepath_or_buffer='./data/cancer/%s/SomaticMutation.txt' % cancer,sep='\t',header=0,index_col=0)
mut_df=mut_df[samp_lst]

ref_net1=netx_name
ref_net2=nety_name
gene_df=pd.read_table(filepath_or_buffer='./data/cancer/%s/gene_lst_%s_%s.txt' % (cancer,ref_net1,ref_net2),sep='\t',header=None,index_col=None,names=['nodes'])
mut_df['genename']=mut_df.index.tolist()
mut_df=pd.merge(left=gene_df,right=mut_df,left_on='nodes',right_on='genename',how='inner')
mut_df.set_index(['nodes'],inplace=True)
mut_df=mut_df.drop(['genename'],axis=1)
g_weig_df=calc_gene_weight(mut_df,gene_df)
lamd=0.1

elps_time_lst=[]
for i in np.arange(0,len(samp_lst)):
    starttime = time.perf_counter()
    samp_k=samp_lst[i]
    if os.path.exists('./results/SSMN_%s_%s_%s/driver/%s.txt' % (ref_net1, ref_net2, cancer, samp_k)):
        print("%d processing--%s done, elapsed time:%ds." % (i + 1, samp_k, time.perf_counter() - starttime))
        continue

    adj_s_k1=load_npz('./results/SSMN_%s_%s_%s/%s/%s.npz' % (ref_net1,ref_net2,cancer,ref_net1,samp_k))
    adj_s_k1=np.array(adj_s_k1.todense())
    adj_s_k2=load_npz('./results/SSMN_%s_%s_%s/%s/%s.npz' % (ref_net1,ref_net2,cancer,ref_net2,samp_k))
    adj_s_k2=np.array(adj_s_k2.todense())

    edgeset1=convert_adj_to_edgeset(adj_s_k1,gene_df)
    edgeset2=convert_adj_to_edgeset(adj_s_k2,gene_df)
    nodes1=get_nodeset_from_edgeset(edgeset1)
    nodes2=get_nodeset_from_edgeset(edgeset2)
    nodes_12,net_s_k1,net_s_k2=intersec_net_x_y(edgeset1,edgeset2,nodes1,nodes2)
    weig_nodes_12=pd.merge(left=nodes_12,right=g_weig_df,how='inner',on='nodes')

    d_s_k1=find_penalty_weighted_MCV_single_layer(net_s_k1,weig_nodes_12,lamd)

    graph2 = create_G_from_dataframe([tuple(v) for v in net_s_k2.values], False)
    # 从graph2中移除d_s_k1
    graph2.remove_nodes_from(d_s_k1)
    graph2.remove_nodes_from(list(nx.isolates(graph2)))
    graph2_edges = list(graph2.edges)
    net_s_k22 = pd.DataFrame({'source': [e[0] for e in graph2_edges], 'target': [e[1] for e in graph2_edges]})

    d_s_k2=find_penalty_weighted_MCV_single_layer(net_s_k22,weig_nodes_12,lamd)
    d_s_union12=list(set(d_s_k1+d_s_k2))

    d_s_union12=pd.DataFrame(d_s_union12)
    d_s_union12.to_csv(path_or_buf='./results/SSMN_%s_%s_%s/driver/%s.txt' %
                              (ref_net1, ref_net2, cancer, samp_k), sep='\t', header=False, index=False)
    elps_time_lst.append(time.perf_counter() - starttime)
    print("%d processing--%s done, elapsed time:%ds." % (i+1,samp_k, time.perf_counter() - starttime))

# Ranking mutated driver nodes by the sum of node degree and node score
samp_driver_rank_dic={}
print('Ranking for mutated driver nodes...')
if not os.path.exists('./results/SSMN_%s_%s_%s/score_degree_of_driver_nodes.pkl' % (netx_name, nety_name, cancer)):
    for i in tqdm(np.arange(0, len(samp_lst))):
        starttime = time.perf_counter()
        samp_k = samp_lst[i]
        d_s_k3 = pd.read_table(filepath_or_buffer='./results/SSMN_%s_%s_%s/driver/%s.txt' %  (ref_net1, ref_net2, cancer, samp_k), sep='\t', header=None, index_col=None)
        save_npz('./results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name, nety_name, cancer, netx_name, k_name), spr_csn_k_ref1)
        save_npz('./results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name, nety_name, cancer, nety_name, k_name), spr_csn_k_ref2)

        adj_s_k1 = load_npz('./results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name, nety_name, cancer, netx_name, samp_k))
        adj_s_k1 = np.array(adj_s_k1.todense())
        adj_s_k2 = load_npz('./results/SSMN_%s_%s_%s/%s/%s.npz' % (netx_name, nety_name, cancer, nety_name, samp_k))
        adj_s_k2 = np.array(adj_s_k2.todense())

        edgeset1 = convert_adj_to_edgeset(adj_s_k1, gene_df)  # 耗时操作
        edgeset2 = convert_adj_to_edgeset(adj_s_k2, gene_df)  # 耗时操作

        nodes1 = get_nodeset_from_edgeset(edgeset1)
        nodes2 = get_nodeset_from_edgeset(edgeset2)

        graph1 = create_G_from_dataframe([tuple(v) for v in edgeset1.values], False)
        graph2 = create_G_from_dataframe([tuple(v) for v in edgeset2.values], False)

        degree = []
        weight = []
        for n in d_s_k3[0].values.tolist():
            deg = 0
            if n in graph1.nodes:
                deg = deg + graph1.degree(n)
            if n in graph2.nodes:
                deg = deg + graph2.degree(n)
            degree.append(deg)
            weight.append(g_weig_df[g_weig_df['nodes'] == n].iloc[0, 1])

        samp_mut_rank = pd.DataFrame({'weight': weight, 'degree': degree})
        samp_mut_rank_norm=(samp_mut_rank - samp_mut_rank.min()) / (samp_mut_rank.max() - samp_mut_rank.min())
        samp_mut_rank_norm['name'] = d_s_k3[0].values.tolist()
        samp_mut_rank_norm['sum'] = samp_mut_rank_norm['weight']+samp_mut_rank_norm['degree']
        samp_mut_rank_norm.sort_values('sum', ascending=False, inplace=True)
        samp_driver_rank_dic[samp_k] = samp_mut_rank_norm

    save_obj(samp_driver_rank_dic,'./results/SSMN_%s_%s_%s/score_degree_of_driver_nodes.pkl' % (netx_name, nety_name, cancer))


mut = pd.read_table(filepath_or_buffer='./data/cancer/%s/SomaticMutation.txt' % cancer, sep='\t',
                        header=0, index_col=0)
mut.columns = ['-'.join(s.split('-')[0:3]) for s in mut.columns.values.tolist()]
cnv = pd.read_table(filepath_or_buffer='./data/cancer/%s/CNV.cgt' % cancer,
                    sep='\t', header=0, index_col=0)
cnv.columns = [s.replace('.', '-') for s in cnv.columns.values.tolist()]

samp_driver_rank_dic = load_obj('./results/SSMN_%s_%s_%s/score_degree_of_driver_nodes.pkl' % (netx_name, nety_name, cancer))

pdg_dic = dict()
for s in samp_driver_rank_dic.keys():
    s_df = samp_driver_rank_dic[s]
    s = '-'.join(s.split('-')[0:3])
    if s in mut.columns.values.tolist():
        if s in cnv.columns.values.tolist():
            s_mut_cnv = list(set(mut[s][mut[s] != 0].index.values.tolist() + cnv[s][cnv[s] != 0].index.values.tolist()))

    s_mut_cnv = list(set(s_mut_cnv) & set(s_df['name'].values.tolist()))
    if len(s_mut_cnv) < 5:
        continue
    s_df.set_index('name', inplace=True)
    s_df = s_df.loc[s_mut_cnv,:]
    s_df = s_df.sort_values('sum', ascending=False)
    s_df = s_df.iloc[0:5, :]
    pdg_dic[s] = s_df.index.values.tolist()

save_obj(pdg_dic,'./results/SSMN_%s_%s_%s/PDG_list.pkl' % (netx_name, nety_name, cancer))

print('PDGs list for each sample in %s have been obtained, which is sorted in the PDG_list.pkl file.'%cancer)


