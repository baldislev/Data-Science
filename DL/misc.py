import torch

#calc jacobian of Y with respect to X
def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)


def solve_ls(A: Tensor, b: Tensor, abs: float = 1e-6, rel: float = 1e-6) -> Tensor:
    # Solves the system A x = b in a least-squares sense using SVD, and returns x
    U, S, V = torch.svd(A)
    th = max(rel * S[0].item(), abs)
    # Clip singular values
    Sinv = torch.where(S >= th, 1.0 / S, torch.zeros_like(S))
    return V @ torch.diag(Sinv) @ (U.transpose(1, 0) @ b)


def flatten(*z: Tensor):
    # Flattens a sequence of tensors into one "long" tensor of shape (N,)
    # Note: cat & reshape maintain differentiability!
    flat_z = torch.cat([z_.reshape(-1) for z_ in z], dim=0)
    return flat_z


def unflatten_like(t_flat: Tensor, *z: Tensor):
    # Un-flattens a "long" tensor into a sequence of multiple tensors of arbitrary shape
    t_flat = t_flat.reshape(-1) # make sure it's 1d
    ts = []
    offset = 0
    for z_ in z:
        numel = z_.numel()
        ts.append(
            t_flat[offset:offset+numel].reshape_as(z_)
        )
        offset += numel
    assert offset == t_flat.numel()
    
    return tuple(ts)
	
	
def onehot(y: Tensor, n_classes: int) -> Tensor:
    """
    Encodes y of shape (N,) containing class labels in the range [0,C-1] as one-hot of shape (N,C).
    """
    y = y.reshape(-1, 1) # Reshape y to (N,1)
    zeros = torch.zeros(size=(len(y), n_classes), dtype=torch.float32) # (N,C)
    ones = torch.ones_like(y, dtype=torch.float32)
    
    # scatter: put items from 'src' into 'dest' at indices correspondnig to 'index' along 'dim'
    y_onehot = torch.scatter(zeros, dim=1, index=y, src=ones)
    
    return y_onehot # result has shape (N, C)