# -*- coding: utf-8 -*-  
import sys

class Stack(object):
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def clear(self):
        self.items = []

    def invConnectNames(self, plus=""):
        connectedName = None
        for i in range(len(self.items)):
            if i == 0:
                connectedName = self.items[i]
            else:
                connectedName = connectedName + plus + self.items[i]
        return connectedName

class CCVariableScope:
    def __init__(self, cc, scope):
        self.scope_ = scope
        self.cc = cc

    def __enter__(self):
        self.cc.scope_stack.push(self.scope_)
        return self

    def __exit__(self, type, value, trace):
        self.cc.scope_stack.pop()

class CCNoScopeFlags:
    def __init__(self, cc):
        self.cc = cc

    def __enter__(self):
        self.cc.scope_flags_stack.push(False)
        return self

    def __exit__(self, type, value, trace):
        self.cc.scope_flags_stack.pop()


class CC:
    scope_stack = Stack()
    scope_flags_stack = Stack()
    top_blob = None
    layers = []
    scope_plus_char = "/"
    name_ = None
    conv_keys = None

    def name(self, name=None):
        if name is None:
            return self.name_

        self.name_ = name
        return self.name_

    def clean(self):
        self.scope_stack = Stack()
        self.scope_flags_stack = Stack()
        self.top_blob = None
        self.layers = []

    def scopePlusChar(self, newChar = None):
        if not name is None:
            self.scope_plus_char = newChar
        return self.scope_plus_char

    def top(self, name = None):
        if not name is None:
            self.top_blob = name
        return self.top_blob

    def __init__(self):
        pass

    def noScopeFlag(self):
        return not self.scope_flags_stack.is_empty()

    def scopeName(self, name=None):
        scope = self.scope_stack.invConnectNames(self.scope_plus_char)
        if self.noScopeFlag():
            scope = None

        if scope is None:
            return name
        elif name is None:
            return scope
        else:
            return scope + name

    def scope(self, name):
        return CCVariableScope(self, name)

    def noscope(self):
        return CCNoScopeFlags(self)

    def default_conv_param(self, **keys):
        self.conv_keys = keys

    def get_default_conv_param(self):
        return self.conv_keys

    def default_pooling_param(self, **keys):
        self.pooling_keys = keys

    def get_default_pooling_param(self):
        return self.pooling_keys

    def default_inner_product_param(self, **keys):
        self.inner_product_keys = keys

    def get_default_inner_product_param(self):
        return self.inner_product_keys

    def default_scale_param(self, **keys):
        self.scale_keys = keys

    def get_default_scale_param(self):
        return self.scale_keys

    def default_batchnorm_param(self, **keys):
        self.batch_norm_keys = keys

    def get_default_batchnorm_param(self):
        return self.batch_norm_keys

    def prototxt(self):
        p = self.name()
        if not p is None:
            p = "name: \"%s\"" % p

        for l in self.layers:
            if p is None:
                p = str(l)
            else:
                p = p + "\n" + str(l)
        return p

    def save_to_file(self, file_name, contents):
        fh = open(file_name, 'w')
        fh.write(contents)
        fh.close()

class Serializbale:
    prefix = Stack()

    def getPrefix(self):
        if self.prefix.is_empty():
            return ""
        else:
            return self.prefix.invConnectNames()

    def pushPrefix(self, p="  "):
        self.prefix.push(p)

    def popPrefix(self):
        self.prefix.pop()

    def selGeneralLayer(self, layer):
        proto = "layer {"
        itemval = ""
        self.pushPrefix()

        #firstProb = ["name", "type", "bottom", "top"]
        firstProb = ["bottom", "top", "name", "type"]
        pkeys = layer.prob_.keys()

        for i in range(len(firstProb)):
            if firstProb[i] in pkeys:
                item = firstProb[i]
                itemval = itemval + "\n" + self.selitem(item, layer.prob_[item])

        for item in layer.prob_:
            if not item in firstProb:
                itemval = itemval + "\n" + self.selitem(item, layer.prob_[item])
        self.popPrefix()
        proto = proto + itemval + "\n}"
        return proto

    def selInputlayer(self, layer):
        itemval = self.selitem("input", layer.prob_["input"])
        itemval = itemval + "\n" + self.selitem("input_shape", layer.prob_["input_shape"])
        return itemval

    def prototxt(self, message):
        stype = message.prob_["type"]

        if stype == "Input":
            return self.selInputlayer(message)
        else:
            return self.selGeneralLayer(message)

    def selList(self, name, value):
        proto = ""
        for i in range(len(value)):
            p = self.selitem(name, value[i])
            if i == 0:
                proto = p
            else:
                proto = proto + "\n" + p
        return proto

    def selDict(self, name, value, hasFlag = True):
        if value is None:
            return ""

        proto = ""
        if hasFlag:
            proto = "%s%s {" % (self.getPrefix(), name)
            self.pushPrefix()

        for item in value:
            if not value[item] is None:
                p = self.selitem(item, value[item])

                if proto == "":
                    proto = p
                else:
                    proto = proto + "\n" + p

        if hasFlag:
            self.popPrefix()
            proto = "%s\n%s}" % (proto, self.getPrefix())
        return proto

    def selitem(self, name, value):
        
        if type(value) is list:
            return self.selList(name, value)
        elif type(value) is dict:
            return self.selDict(name, value)
        elif type(value) is str:
            return "%s%s: \"%s\"" % (self.getPrefix(), name, value)
        elif type(value) is Enum:
            return "%s%s: %s" % (self.getPrefix(), name, str(value))
        elif type(value) is bool:
            return "%s%s: %s" % (self.getPrefix(), name, "true" if value else "false")
        else:
            return "%s%s: %s" % (self.getPrefix(), name, str(value))

class Enum:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

class Layer(Serializbale):
    prob_ = {}

    def __init__(self, **keys):
        self.prob_ = keys
        if "phase" in keys:
            phase = keys["phase"]
            del keys["phase"]
            keys["include"] = {"phase": phase}

    def prob(self):
        return self.prob_
        
    def __str__(self):
        return self.prototxt(self)

    def __repr__(self):
        return str(self)

class Layers:
    def __init__(self, cc):
        self.cc = cc

    def getAndRemoveParam(self, dic, key, strongvalue=None):
        if key in dic:
            r = dic[key] if strongvalue is None else strongvalue
            del dic[key]
            return r
        else:
            return strongvalue

    def conv(self, name="conv", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()

        conv_keys = self.cc.get_default_conv_param()
        if not conv_keys is None:
            for k in conv_keys.keys():
                if k not in keys.keys():
                    keys[k] = conv_keys[k]

        param = self.getAndRemoveParam(keys, "param")
        if param is None:
            layer = Layer(top=scope, bottom=top, type="Convolution", name=scope, 
                convolution_param=keys)
        else:
            layer = Layer(top=scope, bottom=top, type="Convolution", name=scope, 
                param=param, convolution_param=keys)
        
        self.cc.layers.append(layer)
        self.cc.top(scope)

    def relu(self, name="relu"):
        scope = self.cc.scopeName(name)
        top = self.cc.top()

        layer = Layer(top=top, bottom=top, type="ReLU", name=scope)
        self.cc.layers.append(layer)

    def pooling(self, name="pool", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()

        pooling_keys = self.cc.get_default_pooling_param()
        
        for k in pooling_keys.keys():
            if k not in keys.keys():
                keys[k] = pooling_keys[k]

        layer = Layer(top=scope, bottom=top, type="Pooling", name=scope, pooling_param=keys)
        self.cc.layers.append(layer)
        self.cc.top(scope)

    def Data(self, name="Data", phase=Enum("TRAIN"), **keys):
        scope = self.cc.scopeName(name)
        top0 = "data"
        top1 = "label"
        layer = Layer(top=[top0, top1], type="Data", name=scope, include={"phase": phase}, **keys)
        self.cc.layers.append(layer)
        self.cc.top(top0)
        return top0, top1

    def LMDBData(self, source, name="Data", batch_size=8, backend=Enum("LMDB"), phase=Enum("TRAIN"), **keys):
        scope = self.cc.scopeName(name)
        top0 = "data"
        top1 = "label"
        layer = Layer(top=[top0, top1], type="Data", name=scope, include={"phase": phase},
            data_param={"source": source, "batch_size": batch_size, "backend": backend}, **keys)
        self.cc.layers.append(layer)
        self.cc.top(top0)
        return top0, top1

    def InputLayer(self, name="data", dims=[]):
        scope = self.cc.scopeName(name)
        layer = Layer(input=scope, type="Input", input_shape={"dim": dims})
        self.cc.layers.append(layer)
        self.cc.top(scope)

    def SoftmaxWithLoss(self, label, name="loss", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()
        self.cc.layers.append(Layer(top=scope, bottom=[top, label], type="SoftmaxWithLoss", name=scope, **keys))

    def Accuracy(self, label, name="accuracy", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()
        self.cc.layers.append(Layer(top=scope, bottom=[top, label], type="Accuracy", name=scope, **keys))

    def softmax(self, name="prob", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()
        self.cc.layers.append(Layer(top=scope, bottom=[top], type="Softmax", name=scope, **keys))
        self.cc.top(scope)

    def BatchNorm(self, name="bn", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()

        batch_norm_keys = self.cc.get_default_batchnorm_param()
        
        for k in batch_norm_keys.keys():
            if k not in keys.keys():
                keys[k] = batch_norm_keys[k]

        layer = Layer(top=top, bottom=top, type="BatchNorm", name=scope, **keys)
        self.cc.layers.append(layer)

    def inner_product(self, name, num_output=None, **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()

        inner_product_keys = self.cc.get_default_inner_product_param()
        
        for k in inner_product_keys.keys():
            if k not in keys.keys():
                keys[k] = inner_product_keys[k]

        def_inner_product_param = self.getAndRemoveParam(keys, "inner_product_param")
        inner_product_param = def_inner_product_param if num_output is None else {"num_output": num_output}

        layer = Layer(top=scope, bottom=top, type="InnerProduct", name=scope, 
            inner_product_param=inner_product_param, **keys)
        self.cc.layers.append(layer)
        self.cc.top(scope)

    def concat(self, bottoms, name="concat", **keys):
        scope = self.cc.scopeName(name)
        layer = Layer(top=scope, bottom=bottoms, type="Concat", name=scope, **keys)
        self.cc.layers.append(layer)
        self.cc.top(scope)

    def scale(self, name="scale", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()

        scale_keys = self.cc.get_default_scale_param()
        
        for k in scale_keys.keys():
            if k not in keys.keys():
                keys[k] = scale_keys[k]

        layer = Layer(top=top, bottom=top, type="Scale", name=scope, **keys)
        self.cc.layers.append(layer)

    def dropout(self, name="dropout", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()

        layer = Layer(top=top, bottom=top, type="Dropout", name=scope, **keys)
        self.cc.layers.append(layer)

    def Eltwise(self, bottoms, name="", **keys):
        scope = self.cc.scopeName(name)
        layer = Layer(top=scope, bottom=bottoms, type="Eltwise", name=scope, **keys)
        self.cc.layers.append(layer)
        self.cc.top(scope)

    def custom_layer(self, stype, inplace,  name="custom_layer", **keys):
        scope = self.cc.scopeName(name)
        top = self.cc.top()

        if not inplace:
            layer = Layer(top=scope, bottom=top, type=stype, name=scope, **keys)
            self.cc.layers.append(layer)
            self.cc.top(scope)
        else:
            layer = Layer(top=scope, bottom=top, type=stype, name=scope, **keys)
            self.cc.layers.append(layer)
       
class Solver(Serializbale):
    def __init__(self):
        self.net = None
        self.train_net = None
        self.test_net = None
        self.train_state = Enum("TRAIN")
        self.test_state = Enum("TEST")
        self.eval_type = None               #"classification"
        self.ap_version = None              #"Integral"
        self.show_per_class_result = False
        self.test_iter = None
        self.test_interval = 0
        self.test_compute_loss = False
        self.test_initialization = True
        self.base_lr = 0.0
        self.display = None
        self.average_loss = 1
        self.max_iter = None
        self.iter_size = 1
        self.lr_policy = None
        self.gamma = None
        self.power = None
        self.momentum = None
        self.weight_decay = None
        self.regularization_type = "L2"
        self.stepsize = None
        self.stepvalue = None
        self.plateau_winsize = None 
        self.clip_gradients = -1
        self.snapshot = 0
        self.snapshot_prefix = None
        self.snapshot_diff = False
        self.snapshot_format = None         #Enum("BINARYPROTO")
        self.solver_mode = Enum("GPU")
        self.device_id = 0
        self.random_seed = -1
        self.type = Enum("SGD")
        self.delta = 1e-8
        self.momentum2 = 0.999
        self.rms_decay = 0.99
        self.debug_info = False
        self.snapshot_after_train = True
        self.solver_type = Enum("SGD")
        self.save_loss = True
        self.show_realtime_loss = False

    def __str__(self):
        return str(self.selDict(name=None, value=vars(self), hasFlag = False))
       
cc = CC()
L = Layers(cc)