import torch
import torch.nn.functional as F

def find_local_maxima(tensor):
    # Ensure the tensor is 2D
    assert tensor.ndim == 2, "The input tensor must be 2D"
    
    N = tensor.shape[0]
    
    # Pad the tensor to handle edge cases
    padded_tensor = F.pad(tensor, (1, 1, 1, 1), mode='constant', value=float('-inf'))
    
    # Create shifts for comparison
    shifts = [
        (0, 1),   # right
        (0, -1),  # left
        (1, 0),   # down
        (-1, 0),  # up
        (1, 1),   # down-right
        (1, -1),  # down-left
        (-1, 1),  # up-right
        (-1, -1)  # up-left
    ]
    
    # Initialize the mask for local maxima
    maxima_mask = torch.ones_like(tensor, dtype=torch.bool)
    
    # Compare each element with its neighbors
    for shift in shifts:
        shifted_tensor = torch.roll(padded_tensor, shifts=shift, dims=(0, 1))[1:N+1, 1:N+1]
        maxima_mask &= tensor > shifted_tensor
    
    # Get the indices of local maxima
    local_maxima_indices = torch.nonzero(maxima_mask)
    
    return local_maxima_indices, tensor[local_maxima_indices[:, 0], local_maxima_indices[:, 1]]

# Example usage
N = 5
tensor = torch.randn(N, N)
local_maxima_indices, local_maxima_values = find_local_maxima(tensor)

print("Input Tensor:")
print(tensor)
print("\nLocal Maxima Indices:")
print(local_maxima_indices)
print("\nLocal Maxima Values:")
print(local_maxima_values)