import tensorflow as tf
from tensorflow import keras
import pandas as pd
from collections import OrderedDict
from tensorflow.keras.layers import Embedding, Concatenate, Flatten, Dense, Add
from .tool import *


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_sklearn_transform_features(feature_dict, use_features, transformer, Tout):
    input_features = tf.concat([feature_dict[ft] for ft in use_features], axis=1)
    output_features = tf.py_function(func=lambda arr: transformer.transform(arr),
                                     inp=[input_features],
                                     Tout=Tout)
    for i, ft in enumerate(use_features):
        feature_dict[ft] = tf.gather(output_features, indices=[i], axis=1)


def lookup_value(query, ids, values=None, use_value=False):
    if tf.rank(query) != tf.rank(ids):
        query = tf.expand_dims(query, -1)
    result = tf.reduce_max(
        tf.where(
            tf.math.logical_and(
                tf.equal(query, ids),
                tf.greater(query, 0)),
            values if use_value else 1,
            0),
        axis=-1,
        keepdims=True)
    return result


def get_context_preference_fromlist(query, l, multi=True, reduce_type='max', maxsplit=3, l_maxsplit=50):
    ids = tf.strings.split(input=l, sep=',', maxsplit=l_maxsplit).to_tensor(default_value='0',
                                                                            shape=l.shape + [l_maxsplit])
    ids = tf.strings.to_number(ids, out_type=tf.int32)
    if multi:
        # 先对query是mulithot时，先对query进行分解，然后分别获取偏好，最后求平均
        query = tf.strings.split(input=query, sep=',', maxsplit=maxsplit).to_tensor(default_value='0',
                                                                                    shape=query.shape + [maxsplit])
        query = tf.strings.to_number(query, out_type=tf.int32)
        context_fav_list = [lookup_value(tf.gather(query, i, axis=-1), ids) for i in range(maxsplit)]
        context_fav = tf.concat(context_fav_list, axis=-1)
        if reduce_type == 'max':
            return tf.reduce_max(context_fav, axis=-1)
        elif reduce_type == 'sum':
            return tf.reduce_sum(context_fav, axis=-1)
    else:
        return tf.squeeze(lookup_value(query, ids), axis=-1)


def get_context_preference_fromkv(query, kv, multi=False, reduce_type='max', maxsplit=5, l_maxsplit=50, use_value=True):
    splied = tf.strings.split(input=kv, sep=',', maxsplit=l_maxsplit).to_tensor(default_value='0:0',
                                                                                shape=kv.shape + [maxsplit])
    splied = tf.strings.split(splied, sep=':', maxsplit=2).to_tensor(default_value='0',
                                                                     shape=kv.shape + [maxsplit] + [2])
    ids, values = tf.split(splied, 2, axis=-1)
    # splied = tf.strings.split(input=kv, sep=',')
    # splied = tf.strings.split(splied, sep=':').to_tensor(default_value='0')
    # ids, values = tf.split(splied, 2, axis=-1)
    ids = tf.squeeze(tf.strings.to_number(ids, out_type=tf.int32), axis=-1)
    values = tf.squeeze(tf.strings.to_number(values, out_type=tf.float32), axis=-1)
    if multi:
        # 先对query是mulithot时，先对query进行分解，然后分别获取偏好，最后reduce
        query = tf.strings.split(input=query, sep=',', maxsplit=maxsplit).to_tensor(default_value='0',
                                                                                    shape=query.shape + [maxsplit])
        query = tf.strings.to_number(query, out_type=tf.int32)
        context_fav_list = [lookup_value(tf.gather(query, i, axis=-1), ids, values, use_value) for i in range(maxsplit)]
        context_fav = tf.concat(context_fav_list, axis=-1)
        if reduce_type == 'max':
            return tf.reduce_max(context_fav, axis=-1)
        elif reduce_type == 'mean':
            zeros = tf.zeros_like(context_fav, dtype=tf.float32)
            ones = tf.ones_like(context_fav, dtype=tf.float32)
            ct = tf.where(tf.greater(context_fav, zeros), ones, zeros)
            return tf.math.divide_no_nan(
                tf.reduce_sum(context_fav, axis=-1),
                tf.reduce_sum(ct, axis=-1))
    else:
        return tf.squeeze(lookup_value(query, ids, values, use_value), axis=-1)


def get_emb_and_reduce(params, ids, hashtable=None, reducetype='sum', values=None, name=None):
    # mask
    mask = tf.not_equal(ids, -1)
    float_mask = tf.cast(mask, tf.float32)
    # 防止使用-1查询embeddlookuptable
    ids = tf.where(mask, ids, tf.zeros_like(ids, dtype=ids.dtype))
    # hashtable映射
    ids = ids if hashtable is None else hashtable.lookup(ids)
    # embedding lookup
    emb = tf.nn.embedding_lookup(params=params, ids=ids)
    # mask score
    scores = float_mask if values is None else float_mask * values
    emb = tf.matmul(tf.expand_dims(scores, axis=-2), emb)
    name = name if name is None else name + '_sum'
    emb = tf.squeeze(emb, axis=-2, name=name)
    if reducetype == 'mean':
        ct = tf.reduce_sum(float_mask, axis=-1, keepdims=True)
        name = name if name is None else name + '_mean'
        return tf.divide(emb, ct, name=name)
    elif reducetype == 'sum':
        return emb


def split_kv_features(inp, sep=',', kv_sep=':', maxsplit=30):
    if maxsplit > 0:
        splied = tf.strings.split(input=inp, sep=sep, maxsplit=maxsplit).to_tensor(default_value='-1:-1',
                                                                                   shape=inp.shape + [maxsplit])
        splied = tf.strings.split(splied, sep=kv_sep, maxsplit=2).to_tensor(default_value='-1',
                                                                            shape=inp.shape + [maxsplit] + [2])
    else:
        splied = tf.strings.split(input=inp, sep=sep).to_tensor(default_value='-1:-1', )
        splied = tf.strings.split(input=splied, sep=kv_sep).to_tensor(default_value='-1')
    ids, values = tf.split(splied, 2, axis=-1)
    ids = tf.squeeze(tf.strings.to_number(ids, out_type=tf.int32), axis=-1)
    values = tf.squeeze(tf.strings.to_number(values, out_type=tf.float32), axis=-1)
    return ids, values


def split_multihot_features(inp, sep=',', maxsplit=5):
    if maxsplit > 0:
        ids = tf.strings.split(input=inp, sep=sep, maxsplit=maxsplit).to_tensor(default_value='-1',
                                                                                shape=inp.shape + [maxsplit])
    else:
        ids = tf.strings.split(input=inp, sep=sep).to_tensor(default_value='-1')
    return tf.strings.to_number(ids, out_type=tf.int32)


class Vocabulary(keras.layers.Layer):
    def __init__(self, keys_str, is_bucket, **kwargs):
        super(Vocabulary, self).__init__(**kwargs)
        self.keys_str = keys_str
        self.is_bucket = is_bucket
        if is_bucket:
            keys = list(range(len(keys_str.split(', ')) + 1))
            keys = [-1] + keys
            vals = tf.constant(list(range(len(keys))), dtype=tf.int32)
            keys = tf.constant(keys, tf.int32)
        else:
            df = pd.read_csv(keys_str, header=None)
            keys = list(df.values[:, 0])
            is_str = type(keys[0]) == str
            if is_str:
                list_safe_remove(['-1', '0'], keys)
                keys = ['-1', '0'] + keys
            else:
                list_safe_remove([-1, 0], keys)
                keys = [-1, 0] + keys
            vals = tf.constant(list(range(len(keys))), dtype=tf.int32)
            if is_str:
                keys = tf.constant(keys, tf.string)
            else:
                keys = tf.constant(keys, tf.int32)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals)
            , 1)

    def call(self, inputs, **kwargs):
        return self.table.lookup(inputs)

    def size(self):
        return self.table.size().numpy()

    def get_config(self):
        config = super(Vocabulary, self).get_config()
        config.update({"keys_str": self.keys_str,
                       "is_bucket": self.is_bucket})
        return config


class SumPooling(keras.layers.Layer):
    def __init__(self, reducetype='sum', **kwargs):
        super(SumPooling, self).__init__(**kwargs)
        self.reducetype = reducetype

    def call(self, emb, mask=None):
        float_mask = tf.cast(mask, "float32")
        float_mask = tf.expand_dims(float_mask, axis=-2)
        emb = tf.matmul(float_mask, emb)
        if self.reducetype == 'mean':
            ct = tf.reduce_sum(float_mask, axis=-1, keepdims=True)
            return tf.divide(emb, ct)
        return emb

    def get_config(self):
        config = super(SumPooling, self).get_config()
        config.update({"reducetype": self.reducetype})
        return config


class WeightedSumPooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedSumPooling, self).__init__(**kwargs)

    def call(self, emb, weights, mask=None):
        float_mask = tf.cast(mask, "float32")
        weights = weights * float_mask
        emb = tf.matmul(tf.expand_dims(weights, axis=-2), emb)
        return emb

    def get_config(self):
        config = super(WeightedSumPooling, self).get_config()
        return config


class CrossDiscreteFeatures(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CrossDiscreteFeatures, self).__init__(**kwargs)

    def call(self, inputs_list, **kwargs):
        return tf.strings.join(
            [tf.strings.as_string(x) for x in inputs_list]
            , separator=' ')

    def get_config(self):
        config = super(CrossDiscreteFeatures, self).get_config()
        return config


def get_tf_default_values(s):
    if s.type == 'int':
        return tf.constant([int(s.default_value)], dtype=tf.int32)
    elif s.type == 'float':
        return tf.constant([float(s.default_value)], dtype=tf.float32)
    elif s.type == 'str':
        return tf.constant([str(s.default_value)], dtype=tf.string)


def generate_inputs_from_config(use_config_model_input, config_model_input):
    # 从feature字段中提取
    use_features_list = list(
        use_config_model_input.feature.str.replace('cross_|cos_', '').map(lambda x: x.split('_x_')))
    # 从weights字段中提取
    use_features_list += [[x] for x in
                          list(use_config_model_input.loc[(use_config_model_input.weights != ''), 'weights'])]
    # 去重
    use_features_df = pd.DataFrame({'feature': list(set(sum(use_features_list, [])))})
    # 关联获取完整信息
    input_feature_df = pd.merge(use_features_df, config_model_input)
    input_feature_df['type'] = input_feature_df['type'].map(lambda x: tf.int32 if x == 'int' else tf.float32)
    # 排序与reindex
    input_feature_df.sort_values(['feature'], inplace=True)
    input_feature_df.reset_index(drop=True, inplace=True)

    # 获取input
    input_feature_dict = OrderedDict()
    for _, row in input_feature_df.iterrows():
        name = row['feature']
        if row['multi'] == 1:
            input_feature_dict[name] = keras.Input([None], dtype=row['type'], name=name)
        elif row['discrete'] == 1:
            input_feature_dict[name] = keras.Input([1], dtype=row['type'], name=name)
        else:
            input_feature_dict[name] = keras.Input([row['dense']], dtype=row['type'], name=name)

    return input_feature_dict


def generate_voc_layer_from_config(use_config_model_input, config_csv_input, vocabulary_dir='./vocabulary'):
    voc_df = pd.merge(use_config_model_input,
                      config_csv_input[['use_name', 'boundaries']].rename(columns={'use_name': 'feature'}),
                      how='left', on='feature')
    voc_df.fillna('', inplace=True)
    voc_df = voc_df.loc[
        voc_df.feature == voc_df.voc_use, ['feature', 'boundaries', 'type']]

    voc_layer_dict = {}
    for _, row in voc_df.iterrows():
        k = row.feature
        if row.boundaries != '':
            voc_layer_dict[row.feature] = Vocabulary(keys_str=row.boundaries, is_bucket=True, name='Vocabulary_' + k)
        else:
            voc_layer_dict[row.feature] = Vocabulary(f'{vocabulary_dir}/{row.feature}.tsv', is_bucket=False,
                                                     name='Vocabulary_' + k)
    return voc_layer_dict


def generate_emb_layer_from_config(use_config_model_input, voc_layer_dict):
    # 初始化embedding层
    emb_df = pd.DataFrame({'feature': use_config_model_input.emb_share.unique()})
    emb_df = pd.merge(emb_df, use_config_model_input[['feature', 'emb', 'voc_use']])

    emb_layer_dict = {}
    for _, row in emb_df.iterrows():
        k = row.feature
        emb_layer_dict[k] = Embedding(input_dim=voc_layer_dict[row.voc_use].size(),
                                      output_dim=row.emb,
                                      embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.),
                                      mask_zero=True,
                                      name='Embedding_' + k)
    return emb_layer_dict


def generate_discrete_layer_from_config(use_config_model_input, voc_layer_dict):
    discrete_df = use_config_model_input.loc[
        (use_config_model_input.discrete == 1) & (use_config_model_input.wide_feature == 1)]

    discrete_layer_dict = {}
    for _, row in discrete_df.iterrows():
        k = row.feature
        discrete_layer_dict[k] = Embedding(input_dim=voc_layer_dict[row.voc_use].size(),
                                           output_dim=1,
                                           embeddings_initializer=tf.keras.initializers.Zeros(),
                                           # embeddings_regularizer=tf.keras.regularizers.L1(l1=0.01),
                                           mask_zero=True,
                                           name='OneHot_' + k)
    return discrete_layer_dict


def srs_get_dnn_out(use_config_model_input, input_feature_dict, voc_layer_dict, emb_layer_dict, dnn_structure_units,
                    kernel_regularizer=None):
    deep_feature_df = use_config_model_input.loc[use_config_model_input.deep_feature == 1].copy()
    deep_feature_dict = OrderedDict()
    for _, row in deep_feature_df.iterrows():
        k = row.feature
        #     print(k)
        data = input_feature_dict[k]
        # lookup table转换
        if row.voc_use != '':
            data = voc_layer_dict[row.voc_use](data)
        # 获取embedding
        if row.emb_share != '':
            data = emb_layer_dict[row.emb_share](data)
        if row.multi != 0:
            if row.weights == '':
                data = SumPooling(reducetype='mean', name='Pooling_' + k)(data)
            else:
                k = k[:-2]  # kv形式特征后缀冗余
                data_weight = input_feature_dict[row.weights]
                data = WeightedSumPooling(name='Pooling_' + k)(data, data_weight)
        #     print(data)
        if row.dense == 1:
            data = tf.expand_dims(data, axis=-2, name='expand_dims_' + k)
            if row.type == 'int':
                data = tf.cast(data, tf.float32, name='cast_' + k)
        deep_feature_dict[k] = data
    li = list(deep_feature_dict.values())
    if len(li) == 1:
        data = li[0]
    else:
        data = Concatenate(name='Concatenate_deep_feature')(li)
    dnn_input = Flatten(name='dnn_input')(data)
    dnn_hidden = Dense(units=dnn_structure_units[0], activation='relu', name=f'deep_features_hidden_1')(
        dnn_input)
    if len(dnn_structure_units) > 1:
        for i, units in enumerate(dnn_structure_units):
            if i > 0:
                dnn_hidden = Dense(units=units, activation='relu', name=f'deep_features_hidden_{i + 1}')(
                    dnn_hidden)
    dnn_out = Dense(units=1, name='dnn_out', use_bias=False, kernel_regularizer=kernel_regularizer, )(dnn_hidden)
    return dnn_out, deep_feature_dict


def srs_get_wide_out(use_config_model_input, input_feature_dict, voc_layer_dict, discrete_layer_dict,
                     kernel_regularizer=None):
    # wide dense部分
    wide_feature_df = use_config_model_input.loc[use_config_model_input.wide_feature == 1].copy()
    wide_feature_dict_float = OrderedDict()
    for _, row in wide_feature_df.loc[((wide_feature_df.dense == 1) & (wide_feature_df.type == 'float'))].iterrows():
        k = row.feature
        if 'cos_' in row.feature:
            wide_feature_dict_float[k] = Flatten(name='Flatten_' + k)(input_feature_dict[k])
        else:
            wide_feature_dict_float[k] = input_feature_dict[k]
    li = list(wide_feature_dict_float.values())
    if len(li) == 1:
        wide_dense_input_float = li[0]
    else:
        wide_dense_input_float = Concatenate(name='Concatenate_wide_float_feature')(li)

    wide_dense_out_float = Dense(units=1,
                                 kernel_regularizer=kernel_regularizer,
                                 use_bias=True,
                                 name='wide_dense_out_float')(wide_dense_input_float)

    wide_feature_dict_int = OrderedDict()
    for _, row in wide_feature_df.loc[((wide_feature_df.dense == 1) & (wide_feature_df.type == 'int'))].iterrows():
        k = row.feature
        wide_feature_dict_int[k] = input_feature_dict[k]
    li = list(wide_feature_dict_int.values())
    if len(li) == 1:
        wide_dense_input_int = li[0]
    else:
        wide_dense_input_int = Concatenate(name='Concatenate_wide_int_feature')(li)

    wide_dense_out_int = Dense(units=1,
                               kernel_regularizer=kernel_regularizer,
                               use_bias=False,
                               name='wide_dense_out_int')(wide_dense_input_int)

    discrete_feature_dict = OrderedDict()
    for _, row in wide_feature_df.loc[(wide_feature_df.discrete == 1)].iterrows():
        k = row.feature
        #     print(k)
        data = input_feature_dict[k]
        if row.voc_use != '':
            # 查表转换为index
            data = voc_layer_dict[row.voc_use](data)
        data = discrete_layer_dict[row.voc_use](data)
        if row.multi == 1:
            data = SumPooling(reducetype='mean', name='SumMultiHot_' + k)(data)
        #     print(data)
        discrete_feature_dict[k] = data

    data = Add(name='Add_discrete_feature')(list(discrete_feature_dict.values()))
    discrete_out = Flatten(name='discrete_out')(data)

    return wide_dense_out_float, wide_dense_out_int, discrete_out


def prepare_for_make_csv_dataset(head_file, csv_input_file):
    head_df = pd.read_csv(head_file, sep='\t', nrows=2)
    config_csv_input = pd.read_excel(csv_input_file, engine='openpyxl')
    config_csv_input.fillna('', inplace=True)
    config_csv_input = config_csv_input.loc[config_csv_input.feature.isin(head_df.columns)]
    config_csv_input['indices'] = [head_df.columns.get_loc(x) for x in config_csv_input['feature']]
    config_csv_input.sort_values('indices', inplace=True)
    column_defaults = config_csv_input.apply(get_tf_default_values, axis=1).values
    return config_csv_input, column_defaults


# 根据config_csv_input进行特征变换
def feature_transform_general(config_csv_input, return_label=True, return_weight=True, keep_raw_features=''):
    def feature_transform(feature_dict):
        feature_dict_new = {}
        label_list = []
        weight_list = []

        # 补缺失字段
        tensor = list(feature_dict.values())[0]
        for feature in ['entity_dislike_7d', 'entity_trigger']:
            feature_dict_new[feature] = split_multihot_features(
                inp=tf.strings.as_string(tf.zeros_like(tensor, dtype=tf.int32)),
                sep=',',
                maxsplit=1)
        for feature in ['c_entity_dislike_7d', 'c_entity_trigger']:
            feature_dict_new[feature] = tf.expand_dims(tf.zeros_like(tensor, dtype=tf.int32), axis=1)

        # 保留原始值
        keep_raw_features_list = keep_raw_features.split(',')
        if len(keep_raw_features_list) > 0:
            for x in keep_raw_features_list:
                if x != '':
                    name = x + '_raw'
                    feature_dict_new[name] = tf.expand_dims(feature_dict[x], axis=1)

        # 特征变换与解析
        for i in range(len(config_csv_input)):
            s = config_csv_input.iloc[i]
            use_name = s.use_name
            # 修复数据问题
            if s.feature == 'topic_reply_top3':
                feature_dict[s.feature] = tf.strings.regex_replace(feature_dict[s.feature], '0:0', '0')

            # 从kv特征中生成context ctr
            if s.context_query != '':
                name = 'c_' + use_name
                if use_name == 'major_topic':
                    name = 'c_major_topic_id'
                is_multi = config_csv_input.loc[config_csv_input.feature == s.context_query, 'type'].iloc[0] == 'str'
                if s.default_value == '0:0':
                    feature_dict_new[name] = get_context_preference_fromkv(
                        query=feature_dict[s.context_query],
                        kv=feature_dict[s.feature],
                        multi=is_multi,
                        reduce_type='max',
                        maxsplit=5,
                        l_maxsplit=50,
                        use_value=True
                    )
                else:
                    feature_dict_new[name] = get_context_preference_fromlist(
                        query=feature_dict[s.context_query],
                        l=feature_dict[s.feature],
                        multi=is_multi,
                        reduce_type='max',
                        maxsplit=5,
                        l_maxsplit=50
                    )
                # 扩展维度，与input保持一致
                feature_dict_new[name] = tf.expand_dims(feature_dict_new[name], axis=1)

            # 提取label
            if s.feature_type == 'l':
                label_list.append(tf.expand_dims(feature_dict[s.feature], axis=1))
            # 提取weight
            elif s.feature_type == 'w':
                weight_list.append(tf.expand_dims(feature_dict[s.feature], axis=1))
            # 分桶离散化
            elif s.boundaries != '':
                feature_dict_new[use_name] = tf.raw_ops.Bucketize(
                    input=feature_dict[s.feature],
                    boundaries=[float(x) for x in s.boundaries.split(', ')]
                )
                feature_dict_new[use_name] = tf.expand_dims(feature_dict_new[use_name], axis=1)
            # 解析kv变量与multihot变量为list
            elif s.islist == 1:
                # 用户multihot特征保留更长序列
                if s.feature_type == 'u':
                    feature_dict_new[use_name] = split_multihot_features(inp=feature_dict[s.feature], sep=',',
                                                                         maxsplit=50)
                elif s.feature_type == 'i':
                    feature_dict_new[use_name] = split_multihot_features(inp=feature_dict[s.feature], sep=',',
                                                                         maxsplit=5)
            # kv型特征分裂为两个list
            elif s.default_value == '0:0':
                feature_dict_new[use_name + '_k'], feature_dict_new[use_name + '_v'] = split_kv_features(
                    inp=feature_dict[s.feature], sep=',', kv_sep=':', maxsplit=30)
            else:
                feature_dict_new[use_name] = feature_dict[s.feature]
                feature_dict_new[use_name] = tf.expand_dims(feature_dict_new[use_name], axis=1)
        if return_label:
            if return_weight:
                return feature_dict_new, label_list[0], weight_list[0]
            else:
                return feature_dict_new, label_list[0]
        else:
            return feature_dict_new

    return feature_transform
