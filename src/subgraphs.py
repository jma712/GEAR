

class Subgraph:
    # Class for subgraph extraction

    def __init__(self, x, edge_index, path, maxsize=50, n_order=10):
        self.x = x
        self.path = path
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize

        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])),
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)

        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}

    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def adjust_edge(self, idx):
        # Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i

        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            # edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def adjust_x(self, idx):
        # Generate node features for subgraphs
        return self.x[idx]

    def build(self):
        # Extract subgraphs for all nodes
        if os.path.isfile(self.path + '_subgraph') and os.stat(self.path + '_subgraph').st_size != 0:
            print("Exists subgraph file")
            self.subgraph = torch.load(self.path + '_subgraph')
            return

        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][:self.maxsize]
            x = self.adjust_x(nodes)
            edge = self.adjust_edge(nodes)
            self.subgraph[i] = Data(x, edge)
        torch.save(self.subgraph, self.path + '_subgraph')

    def search(self, node_list):
        # Extract subgraphs for nodes in the list
        batch = []
        index = []
        size = 0
        for node in node_list:
            batch.append(self.subgraph[node])
            index.append(size)
            size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)
        batch = Batch().from_data_list(batch)
        return batch, index
