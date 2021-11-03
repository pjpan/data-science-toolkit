import time
import lxml.html
import re
# import emoji
import pandas as pd
import numpy as np
import requests

space_regex = re.compile('[\xa0\u3000\n\r]')
INDEX = 'srs_thread_v1'
TAG_FIELD_NAME = 'tags_all'


def get_pre_n_day_timestamp(n):
    return time.time() - n * 86400


def unix_timestamp(dt):
    if len(dt) == 10:
        dt = dt + ' 00:00:00'
    return time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))


def from_unixtime(t):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def split_list(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


def remove_html(x):
    try:
        x = lxml.html.document_fromstring(x).text_content()
    except:
        x = ''
    return x


# 文本清理
def clean_title(x):
    try:
        # emoji替换
        # x = emoji.demojize(x, delimiters=('_', '_'))
        # 去除特殊空格
        x = space_regex.sub(' ', x)
    except:
        x = ''
    return x


def clean_content(x):
    x = clean_title(x)
    return remove_html(x)


def es_index_to_df(index, use_fields, es, doc=None):
    if doc is None:
        doc = {
            'size': 10000,
            'query': {
                'match_all': {}
            }
        }
    hits_list = []
    res = es.search(index=index, body=doc, scroll='1m')
    scroll = res['_scroll_id']
    hits_list.append(res['hits']['hits'])
    while len(res['hits']['hits']) > 0:
        res = es.scroll(scroll_id=scroll, scroll='1m')
        hits_list.append(res['hits']['hits'])
    hits = sum(hits_list, [])
    dic = {x: [] for x in use_fields}
    for hit in hits:
        for x in use_fields:
            try:
                dic[x].append(hit['_source'][x])
            except:
                dic[x].append(np.nan)
    data = pd.DataFrame(dic)
    data['es_index'] = index
    return data


# 修改tags_all字段逻辑
def update_tags_bulk_item(doc, tag_type, use_tag):
    if TAG_FIELD_NAME in doc['_source']:
        if tag_type in doc['_source'][TAG_FIELD_NAME]:
            # insert tag
            if type(doc['_source'][TAG_FIELD_NAME][tag_type]) == int:
                doc['_source'][TAG_FIELD_NAME][tag_type] = [doc['_source'][TAG_FIELD_NAME][tag_type]]
            doc['_source'][TAG_FIELD_NAME][tag_type].append(use_tag)
            doc['_source'][TAG_FIELD_NAME][tag_type] = list(set(doc['_source'][TAG_FIELD_NAME][tag_type]))
        else:
            doc['_source'][TAG_FIELD_NAME][tag_type] = [use_tag]
    else:
        doc['_source'][TAG_FIELD_NAME] = {}
        doc['_source'][TAG_FIELD_NAME][tag_type] = [use_tag]
    return {
        '_index': doc['_index'],
        '_id': doc['_id'],
        '_type': doc['_type'],
        '_op_type': 'update',
        '_source': {
            'doc': {TAG_FIELD_NAME: doc['_source'][TAG_FIELD_NAME]}
        }
    }


def srs_update_tag_body(tag_type, use_tag, ids, op="APPEND"):
    tag_dicts = [{tag_type: [use_tag]} for _ in range(len(ids))]
    return {
        "op": op,
        "data": dict(zip(ids, tag_dicts))
    }


# 为读取csv dataset将特征与默认值整理成dataframe（按照csv表头排序）
def make_use_features_df(use_features, default_values, head):
    use_features_indices = [head.columns.get_loc(x) for x in use_features]
    use_features_df = pd.DataFrame({'default_values': default_values,
                                    'use_features': use_features,
                                    'use_features_indices': use_features_indices})
    use_features_df.sort_values('use_features_indices', inplace=True)
    use_features_df.reset_index(drop=True, inplace=True)
    return use_features_df


def dingtalk_remind(content, atMobiles='', isatall=False):
    url = 'https://oapi.dingtalk.com/robot/send?access_token=8f917ea594e9e7f685745376799cf59bf1ed186b8222c2f0b8f90b95bccdf4bd'
    if atMobiles == '':
        atmobiles = []
    else:
        atmobiles = atMobiles.split(',')
    data = {
        "at": {
            "atMobiles": atmobiles,
            "isAtAll": isatall
        },
        "text": {
            "content": content
        },
        "msgtype": "text"
    }
    r = requests.post(url, json=data)
    return r.status_code


def list_safe_remove(l, keys):
    for x in l:
        if x in keys:
            keys.remove(x)
