import torch
import torch.nn as nn
from fairseq.modules.layer_norm import LayerNorm
import queue
import numpy as np


def CreateLayerHistory(args, is_encoder):
    history_type = args.encoder_history_type if is_encoder else args.decoder_history_type

    normalize_before = args.encoder_normalize_before if is_encoder else args.decoder_normalize_before
    layer_num = args.encoder_layers if is_encoder else args.decoder_layers
    dim = args.encoder_embed_dim if is_encoder else args.decoder_embed_dim

    if history_type is None:
        return None
    elif history_type == "residual":
        return ResidualLayerHistory(normalize_before, layer_num, dim, is_encoder)
    elif history_type == "dense":
        integration_type = getattr(args, 'encoder_integration_type', 'avg') if is_encoder else \
            getattr(args, 'decoder_integration_type', 'avg')
        windows_size = getattr(args, 'encoder_windows_size', -1) if is_encoder else \
            getattr(args, 'decoder_windows_size', -1)
        return DenseLayerHistory(normalize_before, layer_num, dim, is_encoder, integration_type, windows_size)
    elif history_type == "learnable_dense":
        return LearnableDenseLayerHistory(normalize_before, layer_num, dim, is_encoder)
    elif history_type == "learnable_dense_mask":
        return LearnableDenseMaskLayerHistory(normalize_before, layer_num, dim, is_encoder)
    elif history_type == "learnable_dense_nonorm":
        return LearnableDenseNoNormLayerHistory(normalize_before, layer_num, dim, is_encoder)
    elif history_type == "gru":
        return GruLayerHistory(normalize_before, layer_num, dim, is_encoder)
    else:
        raise ValueError


class BaseLayerHistory(nn.Module):

    def __init__(self, normalize_before, layer_num, dim, is_encoder):
        super(BaseLayerHistory, self).__init__()
        self.is_encoder = is_encoder
        self.normalize_before = normalize_before

        # the first layer (aka. embedding layer) does not have layer normalization
        self.layer_norms = nn.ModuleList(LayerNorm(dim) for _ in range(layer_num))

    def add(self, layer):
        raise NotImplemented

    def pop(self):
        raise NotImplemented

    def clean(self):
        raise NotImplemented


class ResidualLayerHistory(BaseLayerHistory):
    """
    x_n = x_{n-1} + y_{n-1}
    """

    def __init__(self, normalize_before, layer_num, dim, is_encoder):
        super(ResidualLayerHistory, self).__init__(normalize_before, layer_num, dim, is_encoder)
        self.count = 0
        self.x = None
        self.y = None

    def add(self, layer):
        if self.x is None:
            self.x = layer
            self.count += 1
            return
        self.count += 1
        if self.normalize_before:
            self.y = self.layer_norms[self.count - 2](layer)
        else:
            self.y = layer

    def pop(self):
        assert self.x is not None
        if self.y is None:
            return self.x
        ret = self.x + self.y
        if not self.normalize_before:
            ret = self.layer_norms[self.count - 2](ret)
        self.x = ret
        return ret

    def clean(self):
        self.x = None
        self.y = None
        self.count = 0


class DenseLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, normalize_before, layer_num, dim, is_encoder, integration_type, windows_size):
        super(DenseLayerHistory, self).__init__(normalize_before, layer_num, dim, is_encoder)
        self.sum = None
        self.count = 0
        self.individuals = None  # store past individual value, used for windows_size > 0

        self.integration_type = integration_type
        # windows = 1 means not use residual connection
        self.windows_size = windows_size
        if self.windows_size > 0:
            assert self.windows_size <= 1 + layer_num
            self.individuals = queue.Queue(self.windows_size)

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            if self.individuals is not None:
                self.individuals.put(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.sum = self.sum + layer
        if self.windows_size != -1 and self.count > self.windows_size:
            self.sum = self.sum - self.individuals.get()

        if self.individuals is not None:
            self.individuals.put(layer)

    def pop(self):
        assert self.sum is not None
        if self.integration_type == 'sum':
            ret = self.sum
        else:
            if self.windows_size == -1:
                ret = self.sum / self.count
            else:
                ret = self.sum / min(self.count, self.windows_size)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        if self.individuals is not None:
            self.individuals.queue.clear()


class LearnableDenseLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, normalize_before, layer_num, dim, is_encoder):
        super(LearnableDenseLayerHistory, self).__init__(normalize_before, layer_num, dim, is_encoder)
        self.sum = None
        self.count = 0
        self.layer_num = 1 + layer_num
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)
        self.layers = []

    def extra_repr(self):
        return 'n_layers={layer_num}, '.format(**self.__dict__)

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0
        ret = (torch.stack(self.layers, 0) * self.weight[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        self.layers = []

    def get_loss(self):
        return (0.5 * (self.weight.sum(1) - 1.0) ** 2).mean()


class LearnableDenseMaskLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, normalize_before, layer_num, dim, is_encoder):
        super(LearnableDenseMaskLayerHistory, self).__init__(normalize_before, layer_num, dim, is_encoder)
        self.sum = None
        self.count = 0
        self.layer_num = 1 + layer_num
        if is_encoder:
            self.weight_mask = np.loadtxt("encoder_mask.txt", dtype=float, delimiter=' ')
        else:
            self.weight_mask = np.loadtxt("decoder_mask.txt", dtype=float, delimiter=' ')
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0
        ret = (torch.stack(self.layers, 0) * self.weight[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        self.layers = []

    def get_loss(self):
        return (0.5 * (self.weight.sum(1) - 1.0) ** 2).mean()


class LearnableDenseNoNormLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, normalize_before, layer_num, dim, is_encoder):
        super(LearnableDenseNoNormLayerHistory, self).__init__(normalize_before, layer_num, dim, is_encoder)
        self.sum = None
        self.count = 0
        self.layer_num = 1 + layer_num
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)
        self.layers = []
        self.layer_norms = None

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0

        ret = (torch.stack(self.layers, 0) * self.weight[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        self.layers = []


class GruLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, normalize_before, layer_num, dim, is_encoder):
        super(GruLayerHistory, self).__init__(normalize_before, layer_num, dim, is_encoder)
        self.count = 0
        self.gru = nn.GRUCell(dim)
        self.gru_cells = []
        self.layer_norms = nn.ModuleList(LayerNorm(dim) for _ in range(layer_num + 1))
        self.decoder_layers = layer_num

    def compute_gru(self, layer_output):
        if len(self.gru_cells) == 0:
            self.gru_cells.append(layer_output)
            return self.layer_norms[self.count](layer_output)

        self.count += 1
        prev_h = self.gru_cells[-1]
        L, B, H = layer_output.size()
        layer_output = torch.reshape(layer_output, (-1, H))
        prev_h = torch.reshape(prev_h, (-1, H))
        h = self.gru(layer_output, prev_h).view(L, B, H)
        self.gru_cells.append(h)
        if self.count != self.decoder_layers:
            return self.layer_norms[self.count](h)
        else:
            return None

    def clean(self):
        self.gru_cells = []
        self.count = 0
