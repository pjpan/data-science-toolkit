from .tf_utils import *
from tensorflow import keras
import numpy as np


class Dice(keras.layers.Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = keras.layers.BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, inputs, **kwargs):
        x_normed = self.bn(inputs)
        x_p = tf.sigmoid(x_normed)
        return self.alpha * (1.0 - x_p) * inputs + x_p * inputs


class DinAttention(keras.layers.Layer):
    def __init__(self,
                 units,
                 should_softmax=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):
        super(DinAttention, self).__init__(kwargs)
        self.should_softmax = should_softmax
        self.dense_hide = keras.layers.Dense(units=units,
                                             activation=Dice() if activation == 'dice' else activation,
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=kernel_regularizer,
                                             activity_regularizer=activity_regularizer,
                                             name=self.name + '_dense_hide')
        self.dense_out = keras.layers.Dense(units=1,
                                            activation=None,
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            name=self.name + '_dense_out')
        self.scores = None

    def call(self, inputs, **kwargs):
        query = inputs[0]  # [B, H]
        keys = inputs[1]  # [B, T, H]
        mask = inputs[2]
        keys_length = keys.shape[-2]
        # if tf.rank(query) != tf.rank(keys):
        #     query = tf.expand_dims(query, axis=-2)   # 会报错
        query = tf.expand_dims(query, axis=-2)
        tile_multiples = np.ones_like(keys.shape)
        tile_multiples[-2] = keys_length
        queries = tf.tile(query, tile_multiples)  # [B, T, H]
        concated = tf.concat([queries, keys, queries * keys], axis=-1)
        dnn1 = self.dense_hide(concated)
        dnn_out = self.dense_out(dnn1)
        # scores = tf.reshape(dnn_out, [-1, 1, keys_length])  # [B, 1, T]
        # 只转置最后两维
        scores = tf.transpose(dnn_out, perm=[0, 1, 3, 2])
        float_mask = tf.expand_dims(tf.cast(mask, "float32"), 1)  # [B, 1, T]
        if self.should_softmax:
            scores = tf.nn.softmax(scores)  # [B, 1, T]
        # paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        # scores = tf.where(keys_mask, scores, paddings, name=self.name+'_scores')
        scores = tf.multiply(float_mask, scores, name=self.name + '_scores')  # [B, 1, T]
        self.scores = scores
        output = tf.matmul(scores, keys)  # [B, 1, H]
        return tf.squeeze(output, axis=-2, name=self.name)


class WeightedLookup(keras.layers.Layer):
    def __init__(self,
                 params,
                 hashtable=None,
                 reducetype='sum',
                 **kwargs):
        super(WeightedLookup, self).__init__(**kwargs)
        self.params = params
        self.hashtable = hashtable
        self.reducetype = reducetype

    def call(self, ids, values=None, **kwargs):
        return get_emb_and_reduce(params=self.params, ids=ids, hashtable=self.hashtable,
                                  reducetype=self.reducetype, values=values, name=self.name)


class StdContinuous(keras.layers.Layer):
    def __init__(self,
                 params,
                 reducetype='sum',
                 hashtable=None,
                 **kwargs):
        super(StdContinuous, self).__init__(**kwargs)
        self.params = params
        self.reducetype = reducetype
        self.weightedLookup = WeightedLookup(params=self.params, hashtable=hashtable, reducetype=self.reducetype)

    def call(self, inputs, **kwargs):
        ids = tf.zeros_like(inputs, dtype=tf.int32)
        values = inputs
        result = self.weightedLookup(ids, values, name=self.name)
        if len(result.shape) == 2:
            return tf.expand_dims(result, axis=-2, name=self.name + '_expand')
        return result


class StdOneHot(keras.layers.Layer):
    def __init__(self,
                 params,
                 hashtable=None,
                 reducetype='sum',
                 **kwargs):
        super(StdOneHot, self).__init__(**kwargs)
        self.params = params
        self.reducetype = reducetype
        self.hashtable = hashtable

    def call(self, inputs, **kwargs):
        ids = inputs
        ids = ids if self.hashtable is None else self.hashtable.lookup(ids)
        result = tf.nn.embedding_lookup(params=self.params, ids=ids)
        if len(result.shape) == 2:
            return tf.expand_dims(result, axis=-2, name=self.name + '_expand')
        return result


class StdMultiHot(keras.layers.Layer):
    def __init__(self,
                 params,
                 hashtable=None,
                 reducetype='sum',
                 **kwargs):
        super(StdMultiHot, self).__init__(**kwargs)
        self.params = params
        self.hashtable = hashtable
        self.reducetype = reducetype
        self.weightedLookup = WeightedLookup(params=self.params, hashtable=self.hashtable, reducetype=self.reducetype)

    def call(self, inputs, **kwargs):
        ids = inputs
        result = self.weightedLookup(ids)
        if len(result.shape) == 2:
            return tf.expand_dims(result, axis=-2, name=self.name + '_expand')
        return result


class StdPreferenceKV(keras.layers.Layer):
    def __init__(self,
                 params,
                 hashtable=None,
                 reducetype='sum',
                 **kwargs):
        super(StdPreferenceKV, self).__init__(**kwargs)
        self.params = params
        self.hashtable = hashtable
        self.reducetype = reducetype
        self.weightedLookup = WeightedLookup(params=self.params, hashtable=self.hashtable, reducetype=self.reducetype)

    def call(self, inputs, **kwargs):
        ids = inputs[0]
        values = inputs[1]
        result = self.weightedLookup(ids, values)
        if len(result.shape) == 2:
            return tf.expand_dims(result, axis=-2, name=self.name + '_expand')
        return result


class StdSequence(keras.layers.Layer):
    def __init__(self,
                 seq_features,
                 mapping_share_embedding,
                 discrete_features_multi,
                 params_dict,
                 hashtable_dict,
                 embeded_values,
                 dinAttention,
                 reducetype='sum',
                 sep=',',
                 maxsplit=50,
                 input_array=False,
                 **kwargs):
        super(StdSequence, self).__init__(**kwargs)
        self.params_dict = params_dict
        self.hashtable_dict = hashtable_dict
        self.embeded_values = embeded_values
        self.reducetype = reducetype
        self.sep = sep
        self.maxsplit = maxsplit
        self.dinAttention = dinAttention
        self.input_array = input_array
        self.seq_features = seq_features
        self.discrete_features_multi = discrete_features_multi
        self.mapping_share_embedding = mapping_share_embedding

    def call(self, inputs, **kwargs):
        seq_emb_list = []
        mask_list = []
        query_emb_list = []
        for i, ft in enumerate(self.seq_features):
            use_ft = self.mapping_share_embedding[ft]
            hashtable = self.hashtable_dict[use_ft] if use_ft in self.hashtable_dict else None
            result = self.handle_one_feature(inputs[i],
                                             params=self.params_dict[use_ft],
                                             hashtable=hashtable,
                                             is_multi=use_ft in self.discrete_features_multi)
            seq_emb_list.append(result[0])
            mask_list.append(result[1])
            query_emb_list.append(self.embeded_values[use_ft])
        seq_emb = tf.concat(seq_emb_list, axis=-1)
        mask = mask_list[0]
        query_emb = tf.concat(query_emb_list, axis=-1)
        return self.dinAttention([query_emb, seq_emb, mask])

    def handle_one_feature(self, inputs, params, hashtable, is_multi):
        splied = inputs
        if not self.input_array:
            splied = tf.strings.split(input=inputs, sep=self.sep, maxsplit=self.maxsplit). \
                to_tensor(default_value='-1', shape=inputs.shape + [self.maxsplit])
        zeros = tf.zeros_like(splied, dtype=tf.int32)
        minus_one = zeros - 1
        if is_multi:
            minus_one = tf.strings.as_string(minus_one)
        mask = tf.not_equal(splied, minus_one)
        if is_multi:
            seq_emb = StdMultiHot(params, hashtable, self.reducetype)(splied)
        else:
            seq_emb = StdOneHot(params, hashtable, self.reducetype)(splied)
        return seq_emb, mask


class SrsRecModel(keras.Model):
    def __init__(self,
                 HASHTABLE,
                 VOC_SIZE,
                 MAPPING_SHARE_EMBEDDING,
                 MAPPING_KV_FEATURE,
                 embedding_dim,
                 reduce_type,
                 attention_units,
                 activation,
                 reducetype='sum',
                 l1=0.01,
                 l2=0,
                 dropout_rate=0.5,
                 dnn_structure_units=(),
                 features_dnn=(),
                 features_wide=(),
                 features_fm=(),
                 use_features=(),
                 continuous_features=(),
                 discrete_features=(),
                 discrete_features_multi=(),
                 kv_features=(),
                 seq_features=(),
                 embedding_features=(),
                 bucket_dict=(),
                 **kwargs):
        super(SrsRecModel, self).__init__(kwargs)
        # 特征分类
        self.weighted_values = {}
        self.embeded_values = {}
        self.feature_dict = {}
        self.use_features = use_features
        self.continuous_features = continuous_features
        self.discrete_features = discrete_features
        self.discrete_features_multi = discrete_features_multi
        self.kv_features = kv_features
        self.seq_features = seq_features
        self.embedding_features = embedding_features
        self.features_wide = features_wide
        self.features_fm = features_fm
        self.features_dnn = features_dnn
        self.bucket_dict = bucket_dict
        # 通用变量
        self.HASHTABLE = HASHTABLE
        self.VOC_SIZE = VOC_SIZE
        self.MAPPING_SHARE_EMBEDDING = MAPPING_SHARE_EMBEDDING
        self.MAPPING_KV_FEATURE = MAPPING_KV_FEATURE
        self.reducetype = reducetype
        # 超参数
        self.l1 = l1
        self.l2 = l2
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.reduce_type = reduce_type
        self.attention_units = attention_units
        self.dnn_structure_units = dnn_structure_units
        self.activation = activation
        # 变量初始化
        self.EMBEDDINGS = {}
        if len(self.features_dnn + self.features_fm) > 0:
            for ft, voc_size in VOC_SIZE.items():
                self.EMBEDDINGS[ft] = tf.Variable(
                    initial_value=tf.random.truncated_normal(shape=(voc_size, embedding_dim)),
                    trainable=True,
                    name='EMBEDDING_' + ft)
        # 初始化一阶权重
        self.WEIGHTS = {}
        for ft in self.features_wide:
            voc_size = VOC_SIZE[MAPPING_SHARE_EMBEDDING[ft]]
            self.WEIGHTS[ft] = tf.Variable(
                initial_value=tf.random.truncated_normal(shape=(voc_size, 1)),
                trainable=True,
                name='WEIGHT_' + ft)

        self.bias = tf.Variable(initial_value=[[0]], dtype=tf.float32, trainable=True, name='wide_bias')
        self.dnn_input_dropout = keras.layers.Dropout(self.dropout_rate, name='dnn_input_dropout')
        if len(dnn_structure_units) > 0:
            self.dnn_layers = [keras.layers.Dense(
                units=unit,
                activation=Dice() if activation == 'dice' else activation,
                name='dnn_hidden_' + str(i),
                kernel_regularizer=keras.regularizers.L1L2(l1=self.l1, l2=self.l2)
            ) for i, unit in enumerate(dnn_structure_units)]
        self.dnn_out = keras.layers.Dense(units=1, activation=None, name='dnn_out',
                                          kernel_regularizer=keras.regularizers.L1L2(l1=self.l1, l2=self.l2))
        self.dinAttention = DinAttention(units=attention_units
                                         , activation=activation
                                         , name='din_attention'
                                         , kernel_regularizer=keras.regularizers.L1L2(l1=self.l1, l2=self.l2))
        # 为了方便查看中间值
        self.values_wide = self.y_wide = \
            self.values_fm = self.y_fm = \
            self.values_dnn = self.values_dnn_hidden = self.y_dnn = \
            self.y = None

    def call(self, inputs, training=None, **kwargs):
        # 使用dict管理input
        self.feature_dict = {ft: inputs[i] for i, ft in enumerate(self.use_features)}

        # dropout embedding
        # if training:
        # for ft, emb in self.EMBEDDINGS.items():
        #     self.EMBEDDINGS[ft] = tf.nn.dropout(emb, rate=self.dropout_rate)

        # 分桶
        if len(self.bucket_dict) > 0:
            for ft in self.bucket_dict:
                self.feature_dict[ft] = tf.raw_ops.Bucketize(
                    input=self.feature_dict[ft],
                    boundaries=self.bucket_dict[ft])

        if len(self.continuous_features) > 0:
            for ft in self.continuous_features:
                use_ft = self.MAPPING_SHARE_EMBEDDING[ft]
                hashtable = self.HASHTABLE[use_ft] if use_ft in self.HASHTABLE else None
                self.weighted_values[ft] = StdContinuous(params=self.WEIGHTS[ft],
                                                         hashtable=hashtable,
                                                         reducetype=self.reducetype
                                                         )(self.feature_dict[ft])
                if len(self.features_dnn + self.features_fm) > 0:
                    self.embeded_values[ft] = StdContinuous(params=self.EMBEDDINGS[use_ft],
                                                            hashtable=hashtable,
                                                            reducetype=self.reducetype
                                                            )(self.feature_dict[ft])

        if len(self.discrete_features) > 0:
            for ft in self.discrete_features:
                use_ft = self.MAPPING_SHARE_EMBEDDING[ft]
                hashtable = self.HASHTABLE[use_ft] if use_ft in self.HASHTABLE else None
                self.weighted_values[ft] = StdOneHot(params=self.WEIGHTS[ft],
                                                     hashtable=hashtable,
                                                     reducetype=self.reducetype,
                                                     )(self.feature_dict[ft])
                if len(self.features_dnn + self.features_fm) > 0:
                    self.embeded_values[ft] = StdOneHot(params=self.EMBEDDINGS[use_ft],
                                                        hashtable=hashtable,
                                                        reducetype=self.reducetype
                                                        )(self.feature_dict[ft])

        if len(self.discrete_features_multi) > 0:
            for ft in self.discrete_features_multi:
                use_ft = self.MAPPING_SHARE_EMBEDDING[ft]
                hashtable = self.HASHTABLE[use_ft] if use_ft in self.HASHTABLE else None
                self.weighted_values[ft] = StdMultiHot(params=self.WEIGHTS[ft],
                                                       hashtable=hashtable,
                                                       reducetype=self.reducetype
                                                       )(self.feature_dict[ft])
                if len(self.features_dnn + self.features_fm) > 0:
                    self.embeded_values[ft] = StdMultiHot(params=self.EMBEDDINGS[use_ft],
                                                          hashtable=hashtable,
                                                          reducetype=self.reducetype,
                                                          )(self.feature_dict[ft])

        if len(self.kv_features) > 0:
            for ft in self.kv_features:
                use_ft = self.MAPPING_SHARE_EMBEDDING[ft]
                hashtable = self.HASHTABLE[use_ft] if use_ft in self.HASHTABLE else None
                k, v = self.MAPPING_KV_FEATURE[ft]
                self.weighted_values[ft] = StdPreferenceKV(params=self.WEIGHTS[use_ft],
                                                           hashtable=hashtable,
                                                           reducetype=self.reducetype
                                                           )([self.feature_dict[k], self.feature_dict[v]])
                if len(self.features_dnn + self.features_fm) > 0:
                    self.embeded_values[ft] = StdPreferenceKV(params=self.EMBEDDINGS[use_ft],
                                                              hashtable=hashtable,
                                                              reducetype=self.reducetype
                                                              )([self.feature_dict[k], self.feature_dict[v]])

        if len(self.seq_features) > 0:
            inputs = [self.feature_dict[ft] for ft in self.seq_features]
            self.embeded_values['din'] = StdSequence(seq_features=self.seq_features,
                                                     mapping_share_embedding=self.MAPPING_SHARE_EMBEDDING,
                                                     discrete_features_multi=self.discrete_features_multi,
                                                     params_dict=self.EMBEDDINGS,
                                                     hashtable_dict=self.HASHTABLE,
                                                     embeded_values=self.embeded_values,
                                                     dinAttention=self.dinAttention,
                                                     reducetype='sum',
                                                     sep=',',
                                                     maxsplit=50,
                                                     input_array=True,
                                                     )(inputs)

        if len(self.embedding_features) > 0:
            for ft in self.embedding_features:
                self.embeded_values[ft] = tf.expand_dims(self.feature_dict[ft], axis=-2)

        # Wide
        y_wide = None
        if len(self.features_wide) > 0:
            bias_values = tf.ones_like(self.feature_dict[self.features_wide[0]], dtype=tf.float32)
            bias_values = tf.expand_dims(bias_values * self.bias, axis=-2)

            values_wide = tf.concat([self.weighted_values[ft] for ft in self.features_wide] + [bias_values], axis=-1)
            y_wide = tf.reduce_sum(values_wide, axis=-1, name='wide_out')
            self.values_wide, self.y_wide = values_wide, y_wide
            # l1 loss
            weights = tf.concat([self.WEIGHTS[ft] for ft in self.features_wide if ft in self.WEIGHTS] + [self.bias],
                                axis=0)
            y_wide_loss = keras.regularizers.L1L2(l1=self.l1, l2=self.l2)(weights)
            self.add_loss(y_wide_loss)

        # FM
        y_fm = None
        values_fm = None
        if len(self.features_fm) > 0:
            for i, features in enumerate(self.features_fm):
                if len(features) == 2:
                    normalize_a = tf.nn.l2_normalize(self.embeded_values[features[0]], -1)
                    normalize_b = tf.nn.l2_normalize(self.embeded_values[features[1]], -1)
                    result = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1, keepdims=True,
                                           name='fm_out_' + str(i))
                else:
                    values_fm = tf.concat([self.embeded_values[ft] for ft in features], axis=-2)
                    # sum_square part
                    summed_emb = tf.reduce_sum(values_fm, -2)  # None * K
                    summed_emb_square = tf.square(summed_emb)  # None * K
                    # square_sum part
                    squared_emb = tf.square(values_fm)
                    squared_emb_sum = tf.reduce_sum(squared_emb, -2)  # None * K
                    result = 0.5 * tf.reduce_sum(summed_emb_square - squared_emb_sum, axis=-1, keepdims=True,
                                                 name='fm_out_' + str(i))
                if i == 0:
                    y_fm = result
                else:
                    y_fm += result
            self.values_fm, self.y_fm = values_fm, y_fm

        # DNN
        y_dnn = None
        if len(self.features_dnn) > 0:
            values_dnn = tf.concat([self.embeded_values[ft] for ft in self.features_dnn], axis=-1)
            values_dnn = self.dnn_input_dropout(tf.squeeze(values_dnn, axis=-2))
            values_dnn_hidden = values_dnn
            if len(self.dnn_structure_units) > 0:
                for i, layer in enumerate(self.dnn_layers):
                    if i == 0:
                        values_dnn_hidden = layer(values_dnn)
                    else:
                        values_dnn_hidden = layer(values_dnn_hidden)
            y_dnn = self.dnn_out(values_dnn_hidden)
            self.values_dnn, self.values_dnn_hidden, self.y_dnn = values_dnn, values_dnn_hidden, y_dnn
            # embedding loss
            # embeddings = tf.concat([self.EMBEDDINGS[ft] for ft in self.features_dnn if ft in self.EMBEDDINGS],
            #                     axis=0)
            # embeddings_loss = keras.regularizers.L1L2(l1=self.l1, l2=self.l2)(embeddings)
            # self.add_loss(embeddings_loss)

        y = 0
        if y_wide is not None:
            y += y_wide
        if y_fm is not None:
            y += y_fm
        if y_dnn is not None:
            y += y_dnn
        self.y = y
        return tf.nn.sigmoid(y, name='srs_output_ctr')

    def get_config(self):
        config = super(SrsRecModel, self).get_config()
        config.update({
            'embedding_dim':
                self.embedding_dim
        })
        return config
