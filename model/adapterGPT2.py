from transformers import GPT2Model, GPT2Tokenizer, GPT2PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
from transformers.activations import get_activation


projected_task_embedding_dim= 6
# task_hidden_dim = 128
task_hidden_dim = 64
task_embedding_dim = 1
tasks = ["sgd_travel", "sgd_payment", "TMA_restaurant", "TMB_music", "sgd_ridesharing", "TMA_auto", "sgd_music", "sgd_buses", 'TMB_restaurant', 'MWOZ_attraction'\
         ,'TMB_sport', 'sgd_movies', 'sgd_homes', 'TMA_coffee', 'sgd_restaurants', 'sgd_hotels', 'sgd_weather', 'sgd_trains', 'MWOZ_train', 'sgd_flights', \
         'sgd_media', 'MWOZ_taxi', 'sgd_alarm', 'TMA_movie', 'sgd_banks', 'TMA_pizza', 'TMB_flight', 'sgd_rentalcars', 'TMB_movie', 'sgd_events'\
         ,'MWOZ_restaurant', 'sgd_services', 'sgd_calendar', 'TMB_food-ordering', 'MWOZ_hotel', 'TMA_uber', 'TMB_hotel']


train_task_embeddings = False
conditional_layer_norm = True
add_layer_norm_after_adapter = False
add_layer_norm_before_adapter = True
activation_type= 'gelu_new'
input_dim = 768
reduction_factor = 32
hidden_dim = 128

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a
    regular python dictionary.
    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape
            :obj:`(batch_size, 1, hidden_size)` is output.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Adapter(nn.Module):
    def __init__(self, config, bottleneck):
        super(Adapter, self).__init__()
        nx = config.n_embd
        self.ln = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.project_down = nn.Linear(nx, bottleneck)
        self.relu = nn.ReLU()
        self.project_up = nn.Linear(bottleneck, nx)
    def forward(self, x):
        x_ = self.ln(x)
        x_ = self.project_down(x_)
        x_ = self.relu(x_)
        x_ = self.project_up(x_)
        x  = x + x_ #residual connection
        return x


class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""
    def __init__(self, config):
        super(LayerNormHyperNet, self).__init__()
        self.input_dim = input_dim
        self.task_embedding_dim = projected_task_embedding_dim \
            if train_task_embeddings else task_embedding_dim

        self.weight_generator = linear_layer(self.task_embedding_dim, self.input_dim)
        self.bias_generator = linear_layer(self.task_embedding_dim, self.input_dim)
    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)



class HyperAdapter(nn.Module):
    def __init__(self, config, bottleneck):
        super(HyperAdapter, self).__init__()

        self.reduction_factor=reduction_factor
        self.input_dim = input_dim
        self.down_sample_size = self.input_dim // self.reduction_factor
        self.meta_up_sampler = AdapterHyperNet(config, self.input_dim, self.down_sample_size) #521--->16
        self.meta_down_sampler = AdapterHyperNet(config, self.down_sample_size, self.input_dim) #16--->512
        self.activation_type = activation_type
        self.conditional_layer_norm = conditional_layer_norm
        self.train_task_embeddings = train_task_embeddings
        self.add_layer_norm_before_adapter = add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = add_layer_norm_after_adapter

        if self.add_layer_norm_after_adapter:
            if self.conditional_layer_norm:
                self.post_layernorm_hypernet = LayerNormHyperNet(config)
            else:
                self.post_layer_norm = nn.LayerNorm(self.input_dim)
        if self.add_layer_norm_before_adapter:
            if self.conditional_layer_norm:
                self.pre_layernorm_hypernet = LayerNormHyperNet(config)
            else:
                self.pre_layer_norm = nn.LayerNorm(self.input_dim)

        # self.pre_layernorm_hypernet = LayerNormHyperNet(config)
        # self.post_layernorm_hypernet = LayerNormHyperNet(config)
        # self.post_layer_norm = nn.LayerNorm(self.input_dim)

    def call_adapter(self, inputs, task_embedding):
        weight_up, bias_up = self.meta_up_sampler(task_embedding)
        weight_down, bias_down = self.meta_down_sampler(task_embedding)
        down = F.linear(inputs, weight=weight_down, bias=bias_down)
        middle = get_activation(self.activation_type)(down)
        output = F.linear(middle, weight=weight_up, bias=bias_up)
        return output

    def apply_pre_layer_norm(self, inputs, task_embeddings):
        """Applies pre layer norm to the inputs."""
        if self.conditional_layer_norm:
            weight, bias = self.pre_layernorm_hypernet(task_embeddings)
            return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
        else:
            return self.pre_layer_norm(inputs)

    def apply_post_layer_norm(self, inputs, task_embeddings):
        """Applies post layer norm to the inputs."""
        if self.conditional_layer_norm:
            weight, bias = self.post_layernorm_hypernet(task_embeddings)
            return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
        else:
            return self.post_layer_norm(inputs)

    def forward(self, task_embedding, inputs):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        z = self.apply_pre_layer_norm(inputs, task_embedding) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, task_embedding)
        # if self.add_layer_norm_after_adapter:
        # outputs = self.apply_post_layer_norm(outputs, task_embedding)
        outputs = outputs + inputs
        return outputs


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial feeded task embeddings."""

    def __init__(self, config):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = task_hidden_dim
        self.projected_task_embedding_dim = projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(
            linear_layer(config.task_embedding_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embeding_generator(task_embedding).view(-1)



class TaskEmbeddingController(nn.Module):
    """Main module controlling task embeddings."""

    def __init__(self, config):
        super(TaskEmbeddingController, self).__init__()
        # self.device = config.device
        self.task_embedding_dim = task_embedding_dim
        self.tasks = tasks
        self.task_to_task_embeddings = {task: task for task in self.tasks}
        self.set_task_embeddings(self.tasks)
        self.train_task_embeddings = train_task_embeddings
        if self.train_task_embeddings:
            self.task_hyper_net = TaskHyperNet(config)

    def get_task(self, task):
        return self.task_to_task_embeddings[task]

    def set_task_embeddings(self, tasks):
        self.task_to_embeddings = nn.ParameterDict(dict())
        for task in tasks:
            task_embedding = torch.Tensor(torch.randn(self.task_embedding_dim)).cuda(1)
            self.task_to_embeddings[task] = nn.Parameter(task_embedding)

    def forward(self, task):
        task_mapped = self.get_task(task)
        task_embedding = self.task_to_embeddings[task_mapped]
        if self.train_task_embeddings:
            return self.task_hyper_net(task_embedding)
        return task_embedding








class MixAdapter(nn.Module):
    def __init__(self, config, bottleneck_size=400, adapter_num=25):
        super(MixAdapter, self).__init__()
        # 20 adapters with task_id 0--19, when task_id==-1 means dont use adapter
        # self.mixadapter = nn.ModuleList([Adapter(config, bottleneck_size) for _ in range(adapter_num)])

        self.mixadapter = nn.ModuleList([HyperAdapter(config, bottleneck_size) for _ in range(adapter_num)])

        self.task_embedding_controller = TaskEmbeddingController(config)

    def forward(self, x, task_id=-1,task_string=None):
        if task_id==-1:
            return x
        else:
            # task_embeddings=
            # return self.mixadapter[task_id](x)
            task_embedding=self.task_embedding_controller(task_string)
            return self.mixadapter[task_id](task_embedding,x)

# class MixAdapter_zm(nn.Module):
#     def __init__(self, config, bottleneck_size=400, adapter_num=25):
#         super(MixAdapter_zm, self).__init__()
#         # 20 adapters with task_id 0--19, when task_id==-1 means dont use adapter
#         self.mixadapter = nn.ModuleList([Adapter(config, bottleneck_size) for _ in range(adapter_num)])
#         self.mlp = nn.Sequential(
#             nn.Linear(config.n_embd, adapter_num),  # 输出大小为 3，每个适配器一个维度
#             nn.Sigmoid()  # 为每个适配器的权重生成一个介于 0 和 1 之间的值
#         )
#     def forward(self, x, task_id=-1):
#         if task_id==-1:
#             return x
#         else:
#             #config.n_embd: 768
#             # adapter_output =  [adapter(x, task_id) for adapter in self.mixadapter] #dim=2
#             adapter_output =  [adapter(x) for adapter in self.mixadapter] #dim=2
#             # adapter_weights = [mlp(x) for mlp in self.mlp]  # 计算适配器权重, dim=1
#             adapter_weights = self.mlp(x)  # 计算适配器权重，dim=1
#             normalized_adapter_weights=F.normalize(adapter_weights,dim=2)
#             # weighted_output = [ao * aw for ao, aw in zip(adapter_output, adapter_weights)]
#             # mixed_output = torch.stack(weighted_output, dim=0).sum(dim=0)
#             weighted_output = sum([normalized_adapter_weights[:, :, i:i+1]*adapter_output[i] for i in range(len(adapter_output))])
#             # print("mixed_output:", mixed_output.shape, type(mixed_output))
#             return weighted_output

# class MixAdapter_zzq(nn.Module):
#     def __init__(self, config, bottleneck_size=400, adapter_num=25):
#         super(MixAdapter_zzq, self).__init__()
#         # 20 adapters with task_id 0--19, when task_id==-1 means dont use adapter
#         self.mixadapter = nn.ModuleList([Adapter_zzq(config, bottleneck_size) for _ in range(adapter_num)])
#         self.mlp = nn.Sequential(
#             nn.Linear(config.n_embd, adapter_num),  # 输出大小为 3，每个适配器一个维度
#             nn.Sigmoid()  # 为每个适配器的权重生成一个介于 0 和 1 之间的值
#         )
#     def forward(self, x, task_id=-1):
#         if task_id==-1:
#             return x
#         else:
#             #config.n_embd: 768
#             adapter_output =  [adapter(x,task_id) for adapter in self.mixadapter] #dim=2
#             # adapter_weights = [mlp(x) for mlp in self.mlp]  # 计算适配器权重, dim=1
#             adapter_weights = self.mlp(x)  # 计算适配器权重，dim=1
#             # weighted_output = [ao * aw for ao, aw in zip(adapter_output, adapter_weights)]
#             # mixed_output = torch.stack(weighted_output, dim=0).sum(dim=0)
#             weighted_output = sum([adapter_weights[:, :, i:i+1]*adapter_output[i] for i in range(len(adapter_output))])
#             # print("mixed_output:", mixed_output.shape, type(mixed_output))
#             return weighted_output

def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)

def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear

class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = False
        self.task_embedding_dim=task_embedding_dim
        self.weight_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        weight = self.weight_generator(task_embedding).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(task_embedding).view(-1)
        return weight, bias




class GPT2Adapter(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        self.config = config


        
    def get_output_embeddings(self):
        return self.lm_head

    def add_adapters(self,bottleneck_size=100,adapter_num=40):
        # original implementation
        self.adapter_blocks = nn.ModuleList([MixAdapter(self.config,bottleneck_size,adapter_num) for _ in range(self.config.n_layer)])
        # self.adapter_blocks = nn.ModuleList([MixAdapter_zzq(self.config, bottleneck_size, adapter_num) for _ in range(self.config.n_layer)])


    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create postion_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id = -1,
            task_string=None,
            **kwargs,
        ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # transformer_outputs = self.transformer(
        #     input_ids,
        #     past_key_values=past_key_values,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_attention_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        ### OVERLOADING THIS FUNCTION 
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        output_attentions = output_attentions if output_attentions is not None else self.transformer.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.transformer.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.transformer.config.use_cache
        return_dict = return_dict if return_dict is not None else self.transformer.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.transformer.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.transformer.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.transformer.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.transformer.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.transformer.get_head_mask(head_mask, self.transformer.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.transformer.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.transformer.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if(complex):
            img_part = torch.zeros_like(hidden_states)
        for i, (block, layer_past, adapter) in enumerate(zip(self.transformer.h, past_key_values, self.adapter_blocks)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if getattr(self.transformer.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
            outputs[0] = adapter(outputs[0], task_id=task_id, task_string=task_string)

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.transformer.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            transformer_outputs = tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)
        else:
            transformer_outputs = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )


        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

