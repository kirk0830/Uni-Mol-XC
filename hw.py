import torch

def sort_then_merge(index):
    '''sort the indices by the first order, then merge all
    the indices of the same structure together, like from
    [[0, 0], [0, 1], [0, 2], [1, 1]] to [[0, 1, 2], [1]]'''
    index = torch.tensor(index, dtype=torch.long)
    sorted_indices = torch.argsort(index[:, 0])
    index = index[sorted_indices]
    temp = index[:, 0]
    ifirst, _, counts = torch.unique(temp, return_inverse=True, return_counts=True)
    iaccum = torch.cat([torch.tensor([0]), torch.cumsum(counts, dim=0)])
    return [index[iaccum[i]:iaccum[i + 1], 1].tolist()
            for i in range(len(ifirst))], ifirst.tolist()
    
if __name__ == '__main__':
    # Example usage
    index = [[0, 0], [0, 1], [0, 2], [1, 1]]
    merged_index = sort_then_merge(index)
    print(merged_index)  # Output: [[0, 1, 2], [1]]
    
    index = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]
    merged_index = sort_then_merge(index)
    print(merged_index)  # Output: [[0, 1], [0, 1], [0]]