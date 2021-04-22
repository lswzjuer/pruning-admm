# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import torch
import numpy as np
import numpy.linalg as LA
from collections import  OrderedDict
from patterns import pattern3x3_n4

# default pruning layer type
weighted_modules = [
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'Linear', 'Bilinear',
    'PReLU',
    'Embedding', 'EmbeddingBag',
]


_logger = logging.getLogger(__name__)


# custom layertype
class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__
        self._forward = None


class AdmmPruner(object):
    '''
    Unstructured pruning algorithm based on ADMM
    '''
    def __init__(self, model, pattern_config,connectivity_config,
                 trainer=None, num_iterations=30, training_epochs=5,
                 row1=1e-4, row2=1e-4, base_algo='l1'):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress,It must be a pre-trained model
        pattern_config : list
            the configurations that users specify for compression.
            config for pattern pruning.
        connectivity_config : list
            the configurations that users specify for compression.
            config for connectivity pruning.

        """
        #
        self._num_iterations = num_iterations
        self._trainer = trainer
        self._training_epochs = training_epochs
        self._row1 = row1
        self._row2 = row2

        self.bound_model = model
        self.pattern_config = pattern_config
        self.connect_config = connectivity_config
        self.all_patterns = pattern3x3_n4()
        self.pattern_pruning_layers,self.connect_pruning_layers = self.detect_modules_to_compress()
        self.pattern_num = None
        self.pattern_freq = self.init_model_patterns()
        self.best_patterns_index = self.get_best_patterns()

        # init admm mask
        self.init_admm_matrix()
        self.init_admm_matrix_changerate_log()



    def init_admm_matrix(self):
        '''
        Initializes the approximate matrices for pattern pruning and connect pruning
        :return:
        '''
        # pattern
        self.pattern_admm = OrderedDict()
        for name,info in self.pattern_pruning_layers.items():
            layer,config = info
            assert layer.name == name,"The layer name and configuration information do not correspond "
            index,mask,z = self.get_best_pattern(layer.module.weight.detach())
            sub_dict = {
                "index": index,
                "mask": mask,
                "z": z,
                "u": torch.zeros_like(z)
            }
            self.pattern_admm[name] = sub_dict

        # connectivity
        if len(self.connect_pruning_layers.values()) == 0 :
            _logger.info("Connectivity pruning is not performed !")
            self.connect_admm = None
        else:
            self.connect_admm = OrderedDict()
            for name,info in self.connect_pruning_layers.items():
                layer, config = info
                assert layer.name == name,"The layer name and configuration information do not correspond "
                prune_rate = self.get_current_prate(config,iters=0)
                index,mask,y = self.get_best_connect(layer.module.weight.detach(),prune_rate)
                sub_dict = {
                    "index": index,
                    "mask": mask,
                    "y": y,
                    "v": torch.zeros_like(y)
                }
                self.connect_admm[name] = sub_dict


    def init_admm_matrix_changerate_log(self):
        '''
        :return:
        '''
        self.pattern_change_rate = {}
        for name in self.pattern_admm.keys():
            self.pattern_change_rate[name] = []
        self.connect_change_rate = {}
        for name in self.connect_admm.keys():
            self.connect_change_rate[name] = []

    def get_admm_matrix(self):
        '''
        The latest ADMM approximate matrix is obtained to solve the sparse regularization term
        :return:
        '''
        return self.pattern_admm,self.connect_admm

    def get_admm_matrix_change_rate_log(self):
        '''
        :return:
        '''
        return self.pattern_change_rate,self.connect_change_rate


    def update_admm_matrix(self,iters):
        '''
        Update the approximation matrix according to the latest weight values
        :param iters:
        :return:
        '''
        pattern_count = 0
        connect_count = 0
        for name,module in self.bound_model.named_modules():
            if name in self.pattern_admm.keys():
                p_weight = module.weight.detach() + self.pattern_admm[name]["u"]
                pindex, pmask, z = self.get_best_pattern(p_weight)
                u = p_weight - z
                pnew_dict = {"index": pindex,
                            "mask": pmask,
                            "z": z,
                            "u": u}
                # logging info
                # The pattern change rate
                pold_index = self.pattern_admm[name]["index"]
                pequal = pindex == pold_index
                pchange_rate = (np.prod(pequal.shape) - np.count_nonzero(pequal.astype(np.int32)))\
                                                        / np.prod(pequal.shape)
                _logger.info("{}  Pattern change rate: {}".format(name,pchange_rate))
                self.pattern_change_rate[name].append(pchange_rate)
                self.pattern_admm.update({name : pnew_dict})
                pattern_count += 1

            if name in self.connect_admm.keys():
                c_weight = module.weight.detach() + self.connect_admm[name]["v"]
                prune_rate = self.get_current_prate(self.connect_pruning_layers[name][1],
                                                    iters=iters)
                cindex,cmask,y = self.get_best_connect(c_weight,prune_rate)
                v = c_weight - y
                cnew_dict = {"index": cindex,
                            "mask": cmask,
                            "y": y,
                            "v": v}
                cold_index = self.connect_admm[name]["index"]
                cequal = cindex == cold_index
                cchange_rate = (np.prod(cequal.shape) - np.count_nonzero(cequal.astype(np.int32)))\
                                                        / np.prod(cequal.shape)
                _logger.info("{}  Connect change rate: {}".format(name,cchange_rate))
                self.connect_change_rate[name].append(cchange_rate)
                self.connect_admm.update({name: cnew_dict})
                connect_count += 1
        assert pattern_count == len(self.pattern_admm.keys()) and \
               connect_count == len(self.connect_admm.keys()),"Incomplete update !"


    def get_best_pattern(self,weight):
        '''
        :param layer: 3x3 conv layer`s weight ndarray
        :return:
        '''
        shape = weight.size()
        filters,channels = shape[0],shape[1]
        index = np.zeros(shape=(filters,channels),dtype=np.int32)
        mask = torch.zeros_like(weight).to(weight.device)
        for filter_index in range(filters):
            for channel_index in range(channels):
                best_index = 0
                largest_value = 0
                for n in range(self.pattern_num):
                    pattern_index = self.best_patterns_index[n]
                    pattern = torch.from_numpy(self.all_patterns[pattern_index]).type_as(weight).to(weight.device)
                    value = torch.norm(weight[filter_index,channel_index,:,:] * pattern,p=2)
                    if value.item() > largest_value:
                        largest_value = value.item()
                        best_index = pattern_index
                index[filter_index,channel_index] = best_index
                mask[filter_index,channel_index,:,:] = \
                    torch.from_numpy(self.all_patterns[best_index]).type_as(weight).to(weight.device)
        return index, mask, weight * mask


    def get_best_connect(self,weight,prate):
        '''
        :param weight:
        :param prate: connectivity pruning rate
        :return:
        '''
        shape = weight.size()
        filters,channels = shape[0],shape[1]
        zeros_3x3 = torch.zeros((shape[2], shape[3])).to(weight.device)
        normal_val = np.zeros(shape=(filters,channels))
        mask = torch.ones_like(weight).to(weight.device)
        # np_weight = weight.cpu().numpy()

        for filter_index in range(filters):
            for channel_index in range(channels):
                normal_val[filter_index, channel_index] = torch.norm(weight[filter_index,channel_index,:,:],\
                                                                  p=2).item()

        pcen = np.percentile(normal_val, prate * 100)
        index = (normal_val > pcen).astype(np.int32)

        for filter_index in range(filters):
            for channel_index in range(channels):
                if index[filter_index, channel_index] == 0:
                    mask[filter_index,channel_index, :, :] = zeros_3x3
        mask = mask.type_as(weight)
        return index, mask, weight * mask



    def get_current_prate(self,config,iters):
        '''
        TODO: Dynamic pruning updates pruning rate
        :param config:
        :param iters:
        :return:
        '''
        assert 0 <= config["sparsity"] and config["sparsity"] < 1,"Incorrect pruning rate range"
        return config["sparsity"]



    def detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.
        The model will be instrumented and user should never edit it after calling this method.
        """
        pattern_pruning_layers = OrderedDict()
        connect_pruning_layers = OrderedDict()
        for name, module in self.bound_model.named_modules():
            layer = LayerInfo(name, module)
            # get this layer`s pruning config, if none, don`t pruning
            pconfig = self.select_pattern_config(layer)
            cconfig = self.select_connect_config(layer)
            # this layer add the modules_to_compress list
            if pconfig is not None:
                _logger.info("Name: {} Type:{} Pattern pruning".format(layer.name,
                                                                        layer.type))
                pattern_pruning_layers[layer.name] = (layer, pconfig)
            if cconfig is not None:
                _logger.info("Name: {} Type:{} Connect pruning".format(layer.name,
                                                                        layer.type))
                connect_pruning_layers[layer.name] = (layer, cconfig)

        return pattern_pruning_layers,connect_pruning_layers


    def select_pattern_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.pattern_config`

        Parameters
        ----------
        layer : LayerInfo
            one layer

        Returns
        -------
        config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
        for config in self.pattern_config:
            config = config.copy()
            # true pruning layer`s op types
            config['op_types'] = self._expand_config_op_types(config)
            if layer.type not in config['op_types']:
                continue
            # op_names may be is none, it means all the op_types`s op will be pruning
            # if op_names is not none, we just poruning the op in op_name list
            if config.get('op_names') and layer.name not in config['op_names']:
                continue
            if layer.module.weight.data.size(2) != 3 or \
               layer.module.weight.data.size(3) != 3:
                continue
            ret = config
        if ret is None or ret.get('exclude'):
            return None
        return ret

    def select_connect_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.pattern_config`

        Parameters
        ----------
        layer : LayerInfo
            one layer

        Returns
        -------
        config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
        for config in self.connect_config:
            config = config.copy()
            # true pruning layer`s op types
            config['op_types'] = self._expand_config_op_types(config)
            if layer.type not in config['op_types']:
                continue
            # op_names may be is none, it means all the op_types`s op will be pruning
            # if op_names is not none, we just poruning the op in op_name list
            if config.get('op_names') and layer.name not in config['op_names']:
                continue
            ret = config
        if ret is None or ret.get('exclude'):
            return None
        return ret


    def init_model_patterns(self):
        '''
        Initialize the Patterns information from the configuration file and model.
        Only the layers involved in pattern pruning are considered.
        '''
        assert len(self.pattern_pruning_layers.values()) != 0, \
                        "You should specify at least one valid pattern pruning layer"
        pattern_freq = {}
        for index in range(len(self.all_patterns)):
            pattern_freq[index] = 0
        for layer, config in self.pattern_pruning_layers.values():
            if self.pattern_num == None:
                self.pattern_num = config["pattern_num"]
            else:
                assert self.pattern_num == config["pattern_num"],\
                        "Currently, only the same amount of pattern is supported for all layers"
            weight = layer.module.weight.data.detach().cpu().numpy()
            shape = layer.module.weight.data.size()
            for filter_index in range(shape[0]):
                for channel_index in range(shape[1]):
                    best_pattern_index = 0
                    largest_norm = 0
                    for index in range(len(self.all_patterns)):
                        norm_val = LA.norm(weight[filter_index, channel_index, :, :] \
                                           * self.all_patterns[index])
                        if norm_val > largest_norm:
                            largest_norm = norm_val
                            best_pattern_index = index
                    pattern_freq[best_pattern_index] += 1
        return pattern_freq


    def get_best_patterns(self):
        assert self.pattern_num is not  None and self.pattern_num is not None, \
                    "The init_model_patterns function should be called before this function is called"

        self.pattern_freq = sorted(self.pattern_freq.items(),
                                   key= lambda  kv: (kv[1], kv[0]),reverse=True)
        best_patterns_index = []
        for i in range(self.pattern_num):
            best_patterns_index.append(self.pattern_freq[i][0])
        _logger.info("Best pattern indexs is :{}".format(best_patterns_index))
        return best_patterns_index


    def _expand_config_op_types(self, config):
        '''

        :param config:    if config inlcude default, we expand it as default_layers.weighted_modules
        :return:
        '''
        if config is None:
            return []
        expanded_op_types = []
        for op_type in config.get('op_types', []):
            if op_type == 'default':
                expanded_op_types.extend(weighted_modules)
            else:
                expanded_op_types.append(op_type)
        return expanded_op_types

    def compres(self):
        '''
        Perform the ADMM sparse optimization process
        :return:
        '''
        _logger.info('Starting ADMM Compression...')

        optimizer = torch.optim.Adam(
            self.bound_model.parameters(), lr=1e-3, weight_decay=5e-5)

        # Loss = cross_entropy +  l2 regulization + \Sum_{i=1}^N \row_i ||W_i - Z_i^k + U_i^k||^2
        criterion = torch.nn.CrossEntropyLoss()

        # callback function to do additonal optimization, refer to the deriatives of Formula (7)
        def callback():
            for name, module in self.bound_model.named_modules():
                if name in self.pattern_admm.keys():
                    module.weight.data -= self._row1* \
                    (module.weight.data- self.pattern_admm[name]["z"] + self.pattern_admm[name]["u"])
                if name in self.connect_admm.keys():
                    module.weight.data -= self._row2* \
                    (module.weight.data- self.connect_admm[name]["y"] + self.connect_admm[name]["v"])

        # optimization iteration
        for k in range(self._num_iterations):
            _logger.info('ADMM iteration : %d', k)

            # step 1: optimize W with AdamOptimizer
            for epoch in range(self._training_epochs):
                self._trainer(self.bound_model, optimizer=optimizer,
                              criterion=criterion, epoch=epoch, callback=callback)

            # step 2: update Z, U,Y V
            self.update_admm_matrix((k+1)*self._training_epochs)

        # apply prune
        _logger.info('Compression finished.')
        return self.bound_model


    def wrapper_model(self):
        '''
        Change the forward and back propagation functions
        according to the mask information of pruning
        :return:
        '''
        self.mask_dict= {}
        for name,module in self.bound_model.named_modules():
            flag1 = 0
            flag2 = 0
            if name in self.pattern_admm.keys():
                flag1 = 1
            if name in self.connect_admm.keys():
                flag2 = 1
            if flag1 + flag2 > 0:
                layer = LayerInfo(name, module)
                if flag1 == 1 and flag2 ==1:
                    mask = self.pattern_admm[name]["mask"] * self.connect_admm[name]["mask"]
                    mask_sum = mask.sum().item()
                    mask_num = mask.numel()
                    print("{} pattern + connect    -->rate:{}".format(name,1 - mask_sum / mask_num))
                    self._instrument_layer(layer,mask)
                    sub_dict = {
                        "index1": self.pattern_admm[name]["index"],
                        "index2": self.connect_admm[name]["index"],
                        "mask" : mask
                    }
                    self.mask_dict[name] = sub_dict

                elif flag1 == 1  and flag2 == 0:
                    mask = self.pattern_admm[name]["mask"]
                    mask_sum = mask.sum().item()
                    mask_num = mask.numel()
                    print("{} pattern     -->rate:{}".format(name,1 - mask_sum / mask_num))
                    self._instrument_layer(layer,mask)
                    sub_dict = {
                        "index1": self.pattern_admm[name]["index"],
                        "index2": None,
                        "mask" : mask
                    }
                    self.mask_dict[name] = sub_dict

                elif flag1 == 0 and flag2 == 1:
                    mask = self.connect_admm[name]["mask"]
                    mask_sum = mask.sum().item()
                    mask_num = mask.numel()
                    print("{} connect   -->rate:{}".format(name,1 - mask_sum / mask_num))
                    self._instrument_layer(layer,mask)
                    sub_dict = {
                        "index1": None,
                        "index2": self.connect_admm[name]["index"],
                        "mask" : mask
                    }
                    self.mask_dict[name] = sub_dict
                else:
                    ValueError("this layer should be pruned,but lack mask info !")
                    return

        del self.pattern_pruning_layers
        del self.connect_pruning_layers
        del self.pattern_admm
        del self.connect_admm

        return self.bound_model


    def _instrument_layer(self, layer, mask):
        """
        Create a wrapper forward function to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for generating the mask
        """
        assert layer._forward is None, 'Each model can only be compressed once'
        if not _check_weight(layer.module):
            _logger.warning('Module %s does not have parameter "weight"', layer.name)
            return
        layer._forward = layer.module.forward

        def new_forward(*inputs):
            # apply mask to weight
            old_weight = layer.module.weight.data
            layer.module.weight.data = old_weight.mul(mask)
            # calculate forward
            ret = layer._forward(*inputs)
            return ret

        # new forward func
        def grad_hook(grad):
            return grad.mul(mask)

        layer.module.forward = new_forward
        layer.module.weight.register_hook(grad_hook)


    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None):
        """
        Export pruned model weights, masks and onnx model(optional)

        Parameters
        ----------
        model_path : str
            path to save pruned model state_dict
        mask_path : str
            (optional) path to save mask dict
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        """

        assert model_path is not None, 'model_path must be specified'
        assert len(self.mask_dict.keys()) != 0, 'Do wrapper_model func first !  '
        for name, m in self.bound_model.named_modules():
            if name == "":
                continue
            mask = None
            if name in self.mask_dict.keys():
                mask = self.mask_dict[name]["mask"].clone()
                self.mask_dict[name]["mask"] = self.mask_dict[name]["mask"].cpu().numpy()
            if mask is not None:
                mask_sum = mask.sum().item()
                mask_num = mask.numel()
                _logger.info('Layer: {} Sparsity: {}'.format(name, 1 - mask_sum / mask_num))
                m.weight.data = m.weight.data.mul(mask)
        torch.save(self.bound_model.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)

        if mask_path is not None:
            mask_state={
                "mask_dict":self.mask_dict,
                "best_patterns_index" : self.best_patterns_index,
            }
            torch.save(mask_state, mask_path)
            _logger.info('Mask dict saved to %s', mask_path)

        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self.bound_model, input_data, onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)



def _check_weight(module):
    try:
        return isinstance(module.weight, torch.nn.Parameter) and isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False



if __name__ == "__main__":

    import sys
    sys.path.append("../")
    from models import get_models,init_weights
    model = get_models("ResNet18",num_classes=10)
    print(model)

    pattern_config = [{
        'pattern_num': 8,
        'op_types': ['Conv2d'],
        'op_names': [
                     'layer2.0.conv1']
    }]

    connect_config = [
        {
        'sparsity': 0.5,
        'op_types': ['Conv2d'],
        'op_names': ['layer1.0.conv1']
        }
    ]

    admm_pruner = AdmmPruner(model=model,
                            pattern_config=pattern_config,
                             connectivity_config=connect_config)


    init_weights(model, init_type='xavier', gain=0.02)

    admm_pruner.update_admm_matrix(0)
    rate1,rate2 = admm_pruner.get_admm_matrix_change_rate_log()

    model = admm_pruner.wrapper_model()

    admm_pruner.export_model("./test.pth", mask_path="./mask.pth")