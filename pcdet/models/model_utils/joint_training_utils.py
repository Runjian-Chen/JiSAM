import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class SeparateBN(nn.Module):
    """Separate Batch Normalization for Tensors.

    Note:
        This implementation is modified from nn.Norm1D

        use different running mean and variance for different domains
        use the same weight and bias for different domains
    """
    def __init__(self, num_features, domains, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(SeparateBN, self).__init__()
        self.domains = domains
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            for domain in self.domains:
                setattr(self, f'weight_{domain}', nn.Parameter(torch.Tensor(num_features)))
                setattr(self, f'bias_{domain}', nn.Parameter(torch.Tensor(num_features)))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            for domain in self.domains:
                self.register_buffer(f'running_mean_{domain}', torch.zeros(num_features))
                self.register_buffer(f'running_var_{domain}', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            for domain in self.domains:
                self.register_parameter(f'running_mean_{domain}', None)
                self.register_parameter(f'running_var_{domain}', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            for domain in self.domains:
                getattr(self, f'running_mean_{domain}').zero_()
                getattr(self, f'running_var_{domain}').fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            for domain in self.domains:
                getattr(self, f'weight_{domain}').data.fill_(1)
                getattr(self, f'bias_{domain}').data.zero_()

    def _check_input_dim(self, input, coors = None):
        return NotImplementedError

    def get_features_in_each_domain(self, input, domain_list=None,coors=None):
        raise NotImplementedError

    def forward(self, input, domain_list = None, coors = None):
        '''
            input: [N, C] or [N, C, H, W]
            domain_list: a list mapping batch index to domain name. For example, domain_list[0] is a string indicating the domain of the first sample in the batch.
            coors: if input is a 2-dim features from sparse tensor, coors is used to index the features in the same batch
        '''

        self._check_input_dim(input, coors)
        features_in_each_domain = self.get_features_in_each_domain(input, domain_list, coors)

        features_after_bn = []

        if self.training:
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum

            for domain in self.domains:
                if domain in features_in_each_domain.keys():
                    features = features_in_each_domain[domain]
                    features_after_bn.append(F.batch_norm(features, getattr(self, f'running_mean_{domain}'), getattr(self, f'running_var_{domain}'),
                                                                    getattr(self, f'weight_{domain}'), getattr(self, f'bias_{domain}'), True, exponential_average_factor, self.eps))

            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1

            ## check whether weight and bias update
            return torch.cat(features_after_bn, dim=0).squeeze(-1) # for 2-dim input, a new dim is added when getting features in each domain

        else:
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            for domain in self.domains:
                if domain in features_in_each_domain.keys():
                    features = features_in_each_domain[domain]
                    features_after_bn.append(F.batch_norm(features, getattr(self, f'running_mean_{domain}'), getattr(self, f'running_var_{domain}'),
                                                                    getattr(self, f'weight_{domain}'), getattr(self, f'bias_{domain}'), False, exponential_average_factor, self.eps))

            return torch.cat(features_after_bn, dim=0).squeeze(-1) # for 2-dim input, a new dim is added when getting features in each domain

class SeparateBN1d(SeparateBN):
    def _check_input_dim(self, input, coors=None):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        if coors is None:
            raise ValueError('expected coors information for sparse tensor')

    def get_features_in_each_domain(self, input, domain_list, coors):
        output_dict = {}
        for domain in self.domains:
            if domain in domain_list:
                index_in_domain = [idx for idx, domain_ in enumerate(domain_list) if domain_ == domain]
                features_in_domain = []
                for idx in index_in_domain:
                    features_in_domain.append(input[torch.where(coors[:,0]==idx)[0]])
                output_dict[domain] = torch.cat(features_in_domain, dim=0)

        return output_dict

class SeparateBN2d(SeparateBN):
    def _check_input_dim(self, input, coors=None):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def get_features_in_each_domain(self, input, domain_list, coors = None):
        assert coors is None
        output_dict = {}
        for domain in self.domains:
            if domain in domain_list:
                index_in_domain = [idx for idx, domain_ in enumerate(domain_list) if domain_ == domain]
                output_dict[domain] = input[index_in_domain]

        return output_dict


class ShapeContext(object):
    '''
        Adapted From: https://github.com/facebookresearch/ContrastiveSceneContexts/blob/83515bef4754b3d90fc3b3a437fa939e0e861af8/pretrain/contrastive_scene_contexts/lib/shape_context.py

    '''
    def __init__(self, r1=0.125, r2=2, nbins_xy=2, nbins_zy=2):
        # right-hand rule
        """
        nbins_xy >= 2
        nbins_zy >= 1
        """
        self.r1 = r1
        self.r2 = r2
        self.nbins_xy = nbins_xy
        self.nbins_zy = nbins_zy
        self.partitions = nbins_xy * nbins_zy * 2

    @staticmethod
    def pdist(rel_trans):
        D2 = torch.sum(rel_trans.pow(2), 2)
        return torch.sqrt(D2 + 1e-7)

    @staticmethod
    def compute_rel_trans(A, B):
        return A.unsqueeze(0) - B.unsqueeze(1)

    @staticmethod
    def hash(A, B, seed):
        '''
        seed = bins of B
        entry < 0 will be ignored
        '''
        mask = (A >= 0) & (B >= 0)
        C = torch.zeros_like(A) - 1
        C[mask] = A[mask] * seed + B[mask]
        return C

    @staticmethod
    def compute_angles(rel_trans):
        """ compute angles between a set of points """
        angles_xy = torch.atan2(rel_trans[:, :, 1], rel_trans[:, :, 0])
        # angles between 0, 2*pi
        angles_xy = torch.fmod(angles_xy + 2 * math.pi, 2 * math.pi)

        angles_zy = torch.atan2(rel_trans[:, :, 1], rel_trans[:, :, 2])
        # angles between 0, pi
        angles_zy = torch.fmod(angles_zy + 2 * math.pi, math.pi)

        return angles_xy, angles_zy

    def compute_partitions(self, xyz):
        rel_trans = ShapeContext.compute_rel_trans(xyz, xyz)

        # angles
        angles_xy, angles_zy = ShapeContext.compute_angles(rel_trans)
        angles_xy_bins = torch.floor(angles_xy / (2 * math.pi / self.nbins_xy))
        angles_zy_bins = torch.floor(angles_zy / (math.pi / self.nbins_zy))
        angles_bins = ShapeContext.hash(angles_xy_bins, angles_zy_bins, self.nbins_zy)

        # Compute residual angles
        res_angles_xy = angles_xy - angles_bins * (2 * math.pi / self.nbins_xy)

        # distances
        distance_matrix = ShapeContext.pdist(rel_trans)
        dist_bins = torch.zeros_like(angles_bins) - 1

        # partitions
        mask = (distance_matrix >= self.r1) & (distance_matrix < self.r2)
        dist_bins[mask] = 0
        mask = distance_matrix >= self.r2
        dist_bins[mask] = 1

        bins = ShapeContext.hash(dist_bins, angles_bins, self.nbins_xy * self.nbins_zy)
        return bins, res_angles_xy, distance_matrix

    def compute_partitions_fast(self, xyz):
        '''
        fast partitions:  axis-aligned partitions
        '''

        partition_matrix = torch.zeros((xyz.shape[0], xyz.shape[0]))
        partition_matrix = partition_matrix.cuda() - 1e9

        rel_trans = ShapeContext.compute_rel_trans(xyz, xyz)
        maskUp = rel_trans[:, :, 2] > 0.0
        maskDown = rel_trans[:, :, 2] < 0.0

        distance_matrix = ShapeContext.pdist(rel_trans)

        mask = (distance_matrix[:, :] > self.r1) & (distance_matrix[:, :] <= self.r2)
        partition_matrix[mask & maskUp] = 0
        partition_matrix[mask & maskDown] = 1

        mask = distance_matrix[:, :] > self.r2
        partition_matrix[mask & maskUp] = 2
        partition_matrix[mask & maskDown] = 3
        self.partitions = 4

        return partition_matrix


"""
Pareto optimal approximation utils from:
https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/min_norm_solvers.py
"""

class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn