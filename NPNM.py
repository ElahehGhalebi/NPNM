import subprocess
import numpy as np
import matplotlib
import multiprocessing as mp
from multiprocessing import Manager
from operator import itemgetter
from heapq import nlargest
matplotlib.use('Agg')
import networkx as nx
from collections import Counter, defaultdict
import time as t
from iteration_utilities import flatten
from sklearn.metrics.cluster import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score, v_measure_score,homogeneity_completeness_v_measure,homogeneity_score
import os
import community


class network_model:
    def __init__(self, alpha, gamma, tau, theta, K, net, dir, iters, cascs, curr_iter, weights
                 , c_nodes, com_path, com_file, seed, batch_len, o_com):
        """
        :param alpha: controlling number of clusters
        :param gamma: controlling number of nodes
        :param tau: controlling similarity between clusters
        :param theta: threshold of when to stop sampling
        :param K: initial number of clusters
        :param net: graph to inject to model
        :param dir: directory of files
        :param iters: number of iterations for gibbs-sampling
        :param cascs: input cascades
        :param curr_iter: iteration of updating cascades
        :param weights: occurrence of each link in the cascades
        :param c_nodes: nodes participating in each cascade
        :param com_path: the path of the community detection for nodes to compare
        :param seed: seed for randomly initialization
        :param batch_len: number of cascades for each batch
        :param o_com: output community path
        """
        """Assign parameters"""
        self.iter = curr_iter
        self.theta = theta
        self.K = K
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_orig = alpha
        self.dir_name = dir
        self.iterations = iters
        self.seed = seed

        """Best partition of nodes for comparing"""
        self.partition = []
        self.community_precision = 0
        self.community_code_path = com_path
        self.community_file = com_file
        self.algo = 2
        self.out_com_name = o_com

        """For Cascades"""
        self.num_of_cascades = batch_len
        self.cascades = cascs
        self.hit_time = defaultdict(lambda: [])
        self.roots = defaultdict(lambda: [])
        self.zero_probs_links = []

        """For Link Probabilities"""
        self.link_probablity = defaultdict(lambda: 0.0)
        self.probs_links = defaultdict(lambda:0.0)
        self.weights = weights
        self.G_inverse = []
        self.inferred_c_n = defaultdict(lambda: [])

        """For Graph"""
        self.graph = net
        self.all_links = []
        self.links = list(self.graph.edges())
        self.links_reverse = [(r, s) for (s, r) in self.links]
        self.nodes = list(self.graph.nodes())
        self.M = len(self.nodes)
        self.N = len(self.links)
        self.G_main = [0 for _ in range(self.N)]
        self.G = dict()
        self.directed = 0
        self.nodes_of_cascade = c_nodes
        for i in self.links:
            self.all_links.append(i)
            if self.directed == 0:
                self.all_links.append((i[1], i[0]))

        """For Clusters"""
        self.NodesClusters = defaultdict(lambda: [])
        self.beta_s = defaultdict(lambda: [])
        self.beta_r = defaultdict(lambda: [])
        self.N_in_k = defaultdict(lambda: [])
        self.N_out_k = defaultdict(lambda: [])

    def first_clusters(self):
        params = [self.alpha * (self.N / self.K) for _ in range(self.K)]
        dir = stats.dirichlet(params)
        d_k = dir.rvs(size=self.N)
        g_ij = [np.argmax(x) for x in d_k]

    def initialization_step(self):
        """
        initialize link to cluster assignments, G
        updating N_i_k and N_j_k
        updating beta_s and beta_r
        updating N_k
        :arg: initial clusters and K
        :return: vars initiated
        """
        for i in self.nodes:
            # self.NodesClusters[i] = -1
            self.beta_s[i] = [0 for _ in range(self.K)]
            self.beta_r[i] = [0 for _ in range(self.K)]
            self.N_in_k[i] = [0 for _ in range(self.K)]
            self.N_out_k[i] = [0 for _ in range(self.K)]

        self.N_k = [0 for _ in range(self.K)]
        self.beta_u_s = [0 for _ in range(self.K)]
        self.beta_u_r = [0 for _ in range(self.K)]

        for node in self.nodes:
            self.hit_time[node] = [100000 for _ in range(self.num_of_cascades)]
        for casc, pairs in self.cascades.items():
            for item in pairs:
                if self.hit_time[item[1]][casc] > item[0]:
                    self.hit_time[item[1]][casc] = item[0]
                if self.hit_time[item[1]][casc] == 0:
                    self.roots[casc] = item[1]
                if item[1] not in self.nodes_of_cascade[casc]:
                    self.nodes_of_cascade[casc].append(item[1])

        """assign each link to its cluster, and if undirected assign reverse link to same cluster as well"""
        for link in self.links:
            k = self.G[link]
            self.N_out_k[link[0]][k] += 1
            self.N_in_k[link[1]][k] += 1
            if self.directed == 0:
                self.G[(link[1], link[0])] = k
                self.N_out_k[link[1]][k] += 1
                self.N_in_k[link[0]][k] += 1

        """after initialization update variables"""
        """remove empty clusters"""
        to_remove = []
        for (k, count) in Counter(self.G.values()).items():
            self.N_k[k] = count
        for k in range(self.K):
            if self.N_k[k] == 0:
                to_remove.append(k)
        for k in range(len(to_remove)):
            self.remove_cluster(to_remove[k])
        """Update number of links for each node in each cluster, as a sender and as a receive
                and update cluster of each node based on its link clusters"""
        for node in self.nodes:
            self.NodesClusters[node] = self.cluster_node_by_links(node)

    def cluster_node_by_links(self, i=None):
        if i is None:
            for i in range(self.M):
                N_node = [self.N_in_k[i][k] + self.N_out_k[i][k] for k in range(self.K)]
                m = max(N_node)
                k_index = [i for i, j in enumerate(N_node) if j == m]
                if len(k_index) == 1:
                    return k_index[0]
                Betas = [(k, self.beta_s[i][k] + self.beta_r[i][k]) for k in k_index]
                maxVal = Betas[0][1]
                k_index = Betas[0][0]
                for i in range(len(Betas)):
                    if Betas[i][1] >= maxVal:
                        maxVal = Betas[i][1]
                        k_index = Betas[i][0]
                self.NodesClusters[i] = k_index
        else:
            #TODO only consider betas for clustering nodes
            N_node = [self.N_in_k[i][k] + self.N_out_k[i][k] for k in range(self.K)]
            m = max(N_node)
            k_index = [i for i, j in enumerate(N_node) if j == m]
            if len(k_index) == 1:
                return k_index[0]
            # Betas = [(self.beta_s[i][k]/(sum([self.beta_s[j][k] for j in self.nodes])) + self.beta_r[i][k]/(sum([self.beta_r[j][k] for j in self.nodes])))*100 for k in self.K]
            Betas = [(k, (self.N_in_k[i][k]+self.N_out_k[i][k])/(self.N_k[k]+1)) for k in k_index]
            # Betas = [(k, self.beta_s[i][k] + self.beta_r[i][k]) for k in k_index]
            maxVal = Betas[0][1]
            k_index = Betas[0][0]
            for i in range(len(Betas)):
                if Betas[i][1] >= maxVal:
                    maxVal = Betas[i][1]
                    k_index = Betas[i][0]
            return k_index

    def remove_cluster(self, k):
        """
        when a cluster is removed, the corresponding parameters of this cluster should be removed
        :param k: index of cluster
        """
        self.K -= 1
        # when a cluster is removed , other cluster ids is shifted to left, so update indices
        for link in self.G:
            if self.G[link] > k:
                self.G[link] -= 1

        self.N_k.pop(k)
        self.beta_u_r.pop(k)
        self.beta_u_s.pop(k)

        for node in self.nodes:
            self.N_in_k[node].pop(k)
            self.N_out_k[node].pop(k)
            self.NodesClusters[node] = self.cluster_node_by_links(node)
            self.beta_s[node].pop(k)
            self.beta_r[node].pop(k)

    def remove_from_cluster(self, link):
        """
        remove link from its current cluster, if number of links=0 remove cluster
        :param link:
        """
        k = self.G[link]  # cluster id of this link

        s = link[0]  # sender index of link
        r = link[1]  # receiver index of link
        # decrease counters
        self.N_out_k[s][k] -= 1
        self.N_in_k[r][k] -= 1
        self.N_k[k] -= 1

        if self.directed == 0:
            # reverse
            link_reverse = (r, s)
            k_r = self.G[link_reverse]  # cluster id of this link
            # decrease counters
            self.N_out_k[r][k_r] -= 1
            self.N_in_k[s][k_r] -= 1
            self.N_k[k_r] -= 1

    def add_new_cluster(self):
        """
        This is commonly used by sampling_g and sampling_beta
        :return: cluster id of new cluster that has been added
        """
        k_new = self.K
        self.N_k.append(0)
        self.beta_u_s.append(0)
        self.beta_u_r.append(0)

        for i in self.nodes:
            if i == -1:
                continue
            self.N_in_k[i].append(0)
            self.N_out_k[i].append(0)

            self.beta_s[i].append(0.0)
            self.beta_r[i].append(0.0)

        self.K += 1
        return k_new

    def assign_to_cluster(self, link, k_new):
        """
        assigns link_id to cluster k_new and updates parameters
        :param link:
        :param k_new:
        """
        self.G[link] = k_new
        g_id = self.links.index(link)
        self.G_main[g_id] = k_new
        self.N_k[k_new] += 1

        s = link[0]  # sender of link
        r = link[1]  # receiver of link

        self.N_out_k[s][k_new] += 1
        self.N_in_k[r][k_new] += 1

        self.NodesClusters[s] = self.cluster_node_by_links(s)
        self.NodesClusters[r] = self.cluster_node_by_links(r)

        if self.directed == 0:
            link_reverse = (r, s)
            self.G[link_reverse] = k_new
            self.N_k[k_new] += 1

            self.N_out_k[r][k_new] += 1
            self.N_in_k[s][k_new] += 1

            self.NodesClusters[r] = self.cluster_node_by_links(r)
            self.NodesClusters[s] = self.cluster_node_by_links(s)

    def calc_cluster_posterior(self, link):
        """for clustering links and nodes"""
        g_ij = [0] * self.K
        a_ij = [0] * self.K
        s = link[0]
        r = link[1]
        for k in range(self.K):
            s1_1 = self.alpha / (np.sum(self.N_k) + self.alpha)
            s1 = (float(self.N_k[k])) / (np.sum(self.N_k) + self.alpha)
            if self.beta_s[s][k] != 0 and self.beta_r[r][k] != 0:  ##s<=J and r<=J
                s2 = (float(self.N_out_k[s][k] + self.beta_s[s][k] * self.tau)) / (self.N_k[k] + self.tau)
                s3 = (float(self.N_in_k[r][k] + self.beta_r[r][k] * self.tau)) / (self.N_k[k] + self.tau)
                s2_1 = self.beta_s[s][k]
                s3_1 = self.beta_r[r][k]
            if self.beta_s[s][k] != 0 and self.beta_r[r][k] == 0:  ##s<=J and r>J
                s2 = (float(self.N_out_k[s][k] + self.beta_s[s][k] * self.tau)) / (self.N_k[k] + self.tau)
                s3 = (self.beta_u_r[k] * self.tau)  # / (self.N_k[k] + self.tau)
                s2_1 = self.beta_s[s][k]
                s3_1 = (self.beta_u_r[k] * self.tau) / (self.N_k[k] + self.tau)
            if self.beta_s[s][k] == 0 and self.beta_r[r][k] != 0:  ##s>J and r<=J
                s2 = (self.beta_u_r[k] * self.tau)  # / (self.N_k[k] + self.tau)
                s3 = (float(self.N_in_k[r][k] + self.beta_r[r][k] * self.tau)) / (self.N_k[k] + self.tau)
                s2_1 = (self.beta_u_s[k] * self.tau) / (self.N_k[k] + self.tau)
                s3_1 = self.beta_r[r][k]
            if self.beta_s[s][k] == 0 and self.beta_r[r][k] == 0:  ##s>J and r>J
                s2 = self.beta_u_s[k] * self.tau
                s3 = self.beta_u_r[k] * self.tau
                s2_1 = (self.beta_u_s[k] * self.tau) / (self.N_k[k] + self.tau)
                s3_1 = (self.beta_u_r[k] * self.tau) / (self.N_k[k] + self.tau)
            g_ij[k] = round(s1 * s2 * s3, 6)
            a_ij[k] = round(s1_1 * s2_1 * s3_1, 6)
        k_max = np.argmax(g_ij)
        if link not in self.weights.keys():
            self.link_probablity[link] = [g_ij[k_max] if g_ij[k_max] >= a_ij[k_max] else a_ij[k_max]][0]
        else:
            self.link_probablity[link] = [g_ij[k_max] if g_ij[k_max] >= a_ij[k_max] else a_ij[k_max]][0] * self.weights[
                link]
        if g_ij[k_max] >= a_ij[k_max]:
            return k_max
        else:
            return -1

    def community_similarity(self):
        nodes_best = defaultdict(list)
        nodes_cluster_best = dict()
        true_vals = []
        pred_vals = []
        for n, d in self.partition.items():
            nodes_best[d].append(n)
            nodes_cluster_best[n] = d
        for node in sorted(self.nodes):
            if node in nodes_cluster_best.keys():
                true_vals.append(nodes_cluster_best[node])
            else:
                true_vals.append(self.K+1)
            pred_vals.append(self.NodesClusters[node])
        return normalized_mutual_info_score(true_vals, pred_vals)

    def sample_beta(self):
        n_s_nodes_with_zero_links = [0 for _ in range(self.K + 1)]
        n_r_nodes_with_zero_links = [0 for _ in range(self.K + 1)]
        for k in range(self.K):
            for i in self.nodes:
                if i == -1:  ### sender if has link
                    continue
                if self.N_out_k[i][k] != 0:
                    self.beta_s[i][k] = float(self.N_out_k[i][k]) / (float(self.N_k[k]) + self.gamma)
                    n_s_nodes_with_zero_links[k] += 1
                else:
                    self.beta_s[i][k] = 0.0

                if self.N_in_k[i][k] != 0:  ### receiver if has link
                    self.beta_r[i][k] = float(self.N_in_k[i][k]) / (float(self.N_k[k]) + self.gamma)
                    n_r_nodes_with_zero_links[k] += 1
                else:
                    self.beta_r[i][k] = 0.0

        for k in range(self.K):
            if n_s_nodes_with_zero_links[k] == 0:
                n_s_nodes_with_zero_links[k] = 1
            if n_r_nodes_with_zero_links[k] == 0:
                n_r_nodes_with_zero_links[k] = 1
            self.beta_u_s[k] = (self.gamma / n_s_nodes_with_zero_links[k]) / (self.N_k[k] + self.gamma)
            self.beta_u_r[k] = (self.gamma / n_r_nodes_with_zero_links[k]) / (self.N_k[k] + self.gamma)

    def sample_g(self, link):
        """
        sample g based on current beta and data
        :param link_id: sample g for link n with link_id
        :return:
        """
        k_pre = self.G[link]
        self.remove_from_cluster(link)
        k_new = self.calc_cluster_posterior(link)

        if k_new == -1:
            k_new = self.add_new_cluster()
        self.assign_to_cluster(link, k_new)
        if self.N_k[k_pre] == 0:
            self.remove_cluster(k_pre)

    def inference(self):
        """
        Gibbs sampler
        samples beta and G in each step
        :param: initial parameters, alpha, gamma, tau and iter
        :return: Model
        """
        self.first_clusters()
        self.initialization_step()
        for sample_step in range(self.iterations):
            all_links_to_infer = list(self.links)
            dif_count = 0
            while all_links_to_infer:
                """randomly select a link from the sequence for edge exchangeable"""
                np.random.seed(self.seed)
                link_id = np.random.randint(0, len(all_links_to_infer))
                link = all_links_to_infer[link_id]
                self.alpha = self.alpha_orig / self.K
                pre_assignment = self.G[link]
                self.sample_beta()
                self.sample_g(link)
                if self.G[link] != pre_assignment:
                    dif_count += 1
                all_links_to_infer.remove(link)
            if (dif_count / self.N) <= self.theta:
                break
        self.community_precision = self.community_similarity()
        
    def calc_posterior_link_probs(self):
        for s in self.nodes:
            for r in self.nodes:
                if s == r :
                    continue
                link = (s, r)
                self.inferred_c_n[link] = self.calc_cluster_posterior(link)

    def max_spanning_tree_of_each_cascade(self, casc_id, w):
        tree = []
        E = w.keys()

        for i in self.nodes_of_cascade[casc_id]:
            if i == self.roots[casc_id]:
                continue
            possible_links = [(j, i) for j in self.nodes_of_cascade[casc_id] if (j, i) in E]
            if len(possible_links) == 0:
                continue
            index_j = np.argmax([w[possible_links[j]] for j in range(len(possible_links))])
            link = possible_links[index_j]
            tree.append(link)
        for link in E:
            if link not in self.probs_links.keys():
                self.probs_links[link] = w[link]
            else:
                self.probs_links[link] += w[link]
        return tree

    def max_spanning_tree_of_each_cascade_2(self, w, current_links):
        weights_not_in_current = defaultdict(lambda :[])
        for k, v in w.items():
            if k in current_links:
                continue
            weights_not_in_current[k] = v
        if len(weights_not_in_current) == 0:
            return [], None
        (index, highest_w) = nlargest(1, enumerate(weights_not_in_current.values()), itemgetter(1))[0]
        highest_link = list(weights_not_in_current.keys())[index]
        node_to_substitude = highest_link[1]
        for link in current_links:
            if link[1] == node_to_substitude:
                current_links.remove(link)
        current_links.append(highest_link)
        for link in current_links:
            if link not in self.probs_links.keys():
                self.probs_links[link] = w[link]
            else:
                self.probs_links[link] += w[link]
        return current_links, highest_link

    def max_spanning_tree_test(self, w, current_links, t_id):
        if len(w) == 0:
            return []
        w_sorted = sorted(w.items(), key=itemgetter(1), reverse=True)

        flag = False

        for link, weight in w_sorted:
            if link not in current_links:
                if t_id == 0:
                    selected_link = link
                    flag = True
                    break
                t_id -= 1
        if not flag:
            return []
        for link in current_links:
            if link[1] == selected_link[1]:
                current_links.remove(link)
                break
        current_links.append(selected_link)

        for link in current_links:
            if link not in self.probs_links.keys():
                self.probs_links[link] = w[link]
            else:
                self.probs_links[link] += w[link]

        return current_links

    def update_cascades_setZeroNotPossiblestoOne(self):
        self.calc_posterior_link_probs()
        self.probs_links.clear()

        links_all = sorted(self.link_probablity, key=self.link_probablity.get, reverse=True)
        casc_links = defaultdict(lambda: [])
        mpt = defaultdict(lambda: [])
        test_not_p = [defaultdict() for _ in range(self.num_of_cascades)]
        start_t = t.time()

        for casc_id in self.cascades.keys():
            link_weights = defaultdict(lambda: [])
            possible_casc_links = [l for l in links_all if
                                   l[0] in self.nodes_of_cascade[casc_id] and l[1] in self.nodes_of_cascade[
                                       casc_id]]
            for link in possible_casc_links:
                s = link[0]
                r = link[1]
                dt = self.hit_time[r][casc_id] - self.hit_time[s][casc_id]
                if dt <= 0:
                    link_weights[link] = 0
                    test_not_p[casc_id][link] = 0
                if dt > 0:
                    test_not_p[casc_id][link] = 1
                    if link not in casc_links[casc_id]:
                        casc_links[casc_id].append(link)
                        w = np.exp(-dt) + 1
                        link_weights[link] = self.link_probablity[link] * w
            mpt[casc_id] = self.max_spanning_tree_of_each_cascade(casc_id, link_weights)

        not_possible_links_for_me = []
        for link in links_all:
            for casc_id in self.cascades.keys():
                if link in test_not_p[casc_id]:
                    if test_not_p[casc_id][link] == 1:
                        for other_casc_id in self.cascades.keys():
                            if link in test_not_p[other_casc_id] and test_not_p[other_casc_id][link] == 0:
                                test_not_p[other_casc_id][link] = 1

            for casc_id in self.cascades.keys():
                if link in test_not_p[casc_id] and test_not_p[casc_id][link] == 0:
                    not_possible_links_for_me.append(link)

        for link in not_possible_links_for_me:
            self.probs_links[link] = 0

        for link, prob in self.link_probablity.items():
            if link not in self.probs_links.keys():
                self.probs_links[link] = prob
        return list(flatten((mpt.values())))

    def update_cascades_consider_as_trees_as_toInfer(self, toInfer):
        self.calc_posterior_link_probs()
        self.probs_links.clear()
        links_all = sorted(self.link_probablity, key=self.link_probablity.get, reverse=True)
        mpt = defaultdict(lambda: [])
        test_not_p = [defaultdict() for _ in range(self.num_of_cascades)]
        start_t = t.time()
        link_weights = [defaultdict(lambda: []) for _ in range(self.num_of_cascades)]

        for casc_id in self.cascades.keys():
            possible_casc_links = [l for l in links_all if
                                   l[0] in self.nodes_of_cascade[casc_id] and l[1] in self.nodes_of_cascade[
                                       casc_id]]
            for link in possible_casc_links:
                s = link[0]
                r = link[1]
                dt = self.hit_time[r][casc_id] - self.hit_time[s][casc_id]
                if dt <= 0:
                    link_weights[casc_id][link] = 0
                    test_not_p[casc_id][link] = 0
                if dt > 0:
                    test_not_p[casc_id][link] = 1
                    w = np.exp(-dt) + 1
                    link_weights[casc_id][link] = self.link_probablity[link] * w
            mpt[casc_id] = self.max_spanning_tree_of_each_cascade(casc_id, link_weights[casc_id])

        inferred_links=set()
        for link in flatten((mpt.values())):
            if (link[1],link[0]) not in inferred_links:
                inferred_links.add(link)
        count_casc = 0
        while len(inferred_links) < toInfer:
            for casc_id in self.cascades.keys():
                new_links,highest_link = self.max_spanning_tree_of_each_cascade_2(link_weights[casc_id], list(mpt[casc_id]))
                if len(new_links) > 0:
                    mpt[casc_id].append(highest_link)
                    if (highest_link[1],highest_link[0]) not in inferred_links:
                        inferred_links.add(highest_link)
                count_casc += 1
            if len(inferred_links) >= toInfer or count_casc>=len(self.cascades):
                break
        not_possible_links_for_me = []
        for link in links_all:
            count_Zero = 0
            count_One = 0
            for i in range(self.num_of_cascades):
                if link in test_not_p[i]:
                    if test_not_p[i][link] == 0:
                        count_Zero += 1
                    else:
                        count_One += 1
            if count_Zero > count_One:
                not_possible_links_for_me.append(link)

        for link in not_possible_links_for_me: self.probs_links[link] = 0

        for link, prob in self.link_probablity.items():
            if link not in self.probs_links.keys():
                self.probs_links[link] = prob
        return list(flatten((mpt.values())))

    def update_cascades_consider_n_mpt(self, num_trees):
        self.calc_posterior_link_probs()
        self.probs_links.clear()

        links_all = sorted(self.link_probablity, key=self.link_probablity.get, reverse=True)
        mpt = defaultdict(lambda: [])
        test_not_p = [defaultdict() for _ in range(self.num_of_cascades)]
        start_t = t.time()
        link_weights = [defaultdict(lambda: []) for _ in range(self.num_of_cascades)]

        for casc_id in self.cascades.keys():
            if casc_id==0: continue
            possible_casc_links = [l for l in links_all if
                                   l[0] in self.nodes_of_cascade[casc_id] and l[1] in self.nodes_of_cascade[
                                       casc_id]]
            for link in possible_casc_links:
                s = link[0]
                r = link[1]
                dt = self.hit_time[r][casc_id] - self.hit_time[s][casc_id]
                if dt <= 0:
                    link_weights[casc_id][link] = 0
                    test_not_p[casc_id][link] = 0
                if dt > 0:
                    test_not_p[casc_id][link] = 1
                    w = np.exp(-dt) + 1
                    link_weights[casc_id][link] = self.link_probablity[link] * w
            mpt[casc_id] = self.max_spanning_tree_of_each_cascade(casc_id, link_weights[casc_id])

        inferred_links=set()
        for link in flatten((mpt.values())):
            if (link[1],link[0]) not in inferred_links:
                inferred_links.add(link)
        new_links= []
        for t_id in range(num_trees):
            for casc_id in self.cascades.keys():
                new_links.append(self.max_spanning_tree_test(link_weights[casc_id], list(mpt[casc_id]), t_id))

        links = list(flatten((mpt.values()))) + list(flatten((new_links)))

        not_possible_links_for_me = []
        for link in links_all:
            count_Zero = 0
            count_One = 0
            for i in range(self.num_of_cascades):
                if link in test_not_p[i]:
                    if test_not_p[i][link] == 0:
                        count_Zero += 1
                    else:
                        count_One += 1
            if count_Zero > count_One:
                not_possible_links_for_me.append(link)

        for link in not_possible_links_for_me:
            self.probs_links[link] = 0

        for link, prob in self.link_probablity.items():
            if link not in self.probs_links.keys():
                self.probs_links[link] = prob
        return links

    def update_cascades_countZero_One(self):
        self.calc_posterior_link_probs()
        self.probs_links.clear()

        links_all = sorted(self.link_probablity, key=self.link_probablity.get, reverse=True)
        mpt = defaultdict(lambda: [])
        test_not_p = [defaultdict() for _ in range(self.num_of_cascades)]
        time_of_start = t.time()

        for casc_id in self.cascades.keys():
            if casc_id==0: continue
            possible_casc_links = [l for l in links_all if
                                   l[0] in self.nodes_of_cascade[casc_id] and l[1] in self.nodes_of_cascade[
                                       casc_id]]
            link_weights = defaultdict(lambda :[])
            for link in possible_casc_links:
                s = link[0]
                r = link[1]
                dt = self.hit_time[r][casc_id] - self.hit_time[s][casc_id]
                if dt <= 0:
                    link_weights[link] = 0
                    test_not_p[casc_id][link] = 0
                if dt > 0:
                    test_not_p[casc_id][link] = 1
                    w = np.exp(-dt) + 1
                    link_weights[link] = self.link_probablity[link] * w
            mpt[casc_id] = self.max_spanning_tree_of_each_cascade(casc_id, link_weights)

        inferred_links=list(flatten((mpt.values())))

        not_possible_links_for_me = []
        for link in links_all:
            count_Zero = 0
            count_One = 0
            for i in range(self.num_of_cascades):
                if link in test_not_p[i]:
                    if test_not_p[i][link] == 0:
                        count_Zero += 1
                    else:
                        count_One += 1
            if count_Zero > count_One:
                not_possible_links_for_me.append(link)

        for link in not_possible_links_for_me:
            self.probs_links[link] = 0

        for link, prob in self.link_probablity.items():
            if link not in self.probs_links.keys():
                self.probs_links[link] = prob
        return inferred_links


