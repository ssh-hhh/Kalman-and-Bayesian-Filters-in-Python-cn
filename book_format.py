# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.


Code supporting the book

Kalman and Bayesian Filters in Python
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


This is licensed under an MIT license. See the LICENSE.txt file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from contextlib import contextmanager
from IPython.core.display import HTML
import json
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import warnings
from kf_book.book_plots import set_figsize, reset_figsize


def _patch_filterpy_numpy2():
    """让 filterpy 1.4.5 兼容 NumPy 2.x。

    filterpy 1.4.5 的 ``stats.multivariate_gaussian`` 内部使用了
    ``np.array(x, copy=False, ndmin=1)``。自 NumPy 2.0 起 ``copy=False``
    的含义变为"绝不复制，需要复制就报错"，而把 list 转成 ndarray 必然
    需要复制，于是抛出::

        ValueError: Unable to avoid copy while creating an array as requested.

    这里用一个等价的实现替换该函数（改用 np.asarray），并同时替换
    ``filterpy.stats`` 与 ``filterpy.stats.stats`` 两个命名空间上的引用，
    这样无论 notebook 是 ``from filterpy.stats import multivariate_gaussian``
    还是 ``import filterpy.stats as stats; stats.multivariate_gaussian(...)``
    都能生效。本函数在 import book_format 时自动调用，因此每个章节开头
    只需照常 ``import book_format`` 即可。
    """
    import math

    def multivariate_gaussian(x, mu, cov):
        x = np.asarray(x, dtype=float).flatten()
        mu = np.asarray(mu, dtype=float).flatten()

        nx = len(mu)
        if np.isscalar(cov):
            if cov < 0:
                raise ValueError('covariance must be > 0')
            cov = np.eye(nx) * cov
        else:
            cov = np.atleast_2d(np.asarray(cov, dtype=float))
            np.linalg.cholesky(cov)   # 校验正定（与 filterpy 原行为一致）

        norm_coeff = nx * math.log(2 * math.pi) + np.linalg.slogdet(cov)[1]
        err = x - mu
        numerator = np.linalg.solve(cov, err).T.dot(err)
        return math.exp(-0.5 * (norm_coeff + numerator))

    try:
        import filterpy.stats as _stats
        _stats.multivariate_gaussian = multivariate_gaussian
        try:
            import filterpy.stats.stats as _stats_mod
            _stats_mod.multivariate_gaussian = multivariate_gaussian
        except Exception:
            pass
    except Exception:
        # filterpy 未安装等情况下静默跳过，不影响其它功能
        pass


_patch_filterpy_numpy2()

# version 1.4.3 of matplotlib has a bug that makes
# it issue a spurious warning on every plot that
# clutters the notebook output
if matplotlib.__version__ == '1.4.3':
    warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    matplotlib.style.use('default')
except:
    pass

def test_filterpy_version():

    import filterpy
    from distutils.version import LooseVersion

    v = filterpy.__version__
    min_version = "1.4.4"
    if LooseVersion(v) < LooseVersion(min_version):
       raise Exception("Minimum FilterPy version supported is {}.\n"
                       "Please install a more recent version.\n"
                       "   ex: pip install filterpy --upgrade".format(
             min_version))


# ensure that we have the correct filterpy loaded. This is
# called when this module is imported at the top of each book
# chapter so the reader can see that they need to update FilterPy.
test_filterpy_version()

pylab.rcParams['figure.max_open_warning'] = 50


@contextmanager
def numpy_precision(precision):
    old = np.get_printoptions()['precision']
    np.set_printoptions(precision=precision)
    yield
    np.set_printoptions(old)

@contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv

def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv


def set_style():
    version = [int(version_no) for version_no in matplotlib.__version__.split('.')]

    try:
        if sys.version_info[0] >= 3:
            style = json.load(open("./kf_book/538.json"))
        else:
            style = json.load(open(".//kf_book/538.json"), object_hook=_decode_dict)
        plt.rcParams.update(style)
    except:
        pass
    np.set_printoptions(suppress=True, precision=3, 
                        threshold=10000., linewidth=70,
                        formatter={'float':lambda x:' {:.3}'.format(x)})

    # I don't know why I have to do this, but I have to call
    # with suppress a second time or the notebook doesn't suppress
    # exponents
    np.set_printoptions(suppress=True)
    reset_figsize()

    style = '''
        <style>
        .output_wrapper, .output {
            height:auto !important;
            max-height:100000px; 
        }
        .output_scroll {
            box-shadow:none !important;
            webkit-box-shadow:none !important;
        }
        </style>
    '''
    return HTML(style)
