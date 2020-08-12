import pytest

from probgf.random_fields import RandomField
from probgf.helpers import cleanup


def test_mrf_valid_config():
    for stop in ['0.1r', '10t']:
        mrf_config = '{},chain,noprior:noprior,0.1,map,6'.format(stop)
        RandomField(mrf_config, [1, 2, 3], None, 2)
    for shape in ['cross3', 'chain', 'cross5']:
        mrf_config = '0.1r,{},noprior:noprior,0.1,map,6'.format(shape)
        RandomField(mrf_config, [1, 2, 3], None, 2)
    for prior in ['noprior', 'es', 'l2', 'tp']:
        mrf_config = '0.1r,cross3,{}:noprior,0.1,map,6'.format(prior)
        RandomField(mrf_config, [1, 2, 3], None, 2)
        mrf_config = '0.1r,cross3,noprior:{},0.1,map,6'.format(prior)
        RandomField(mrf_config, [1, 2, 3], None, 2)
    for lam in ['0.1', '0.001']:
        mrf_config = '0.1r,cross3,noprior:noprior,{},map,6'.format(lam)
        RandomField(mrf_config, [1, 2, 3], None, 2)
    for pred in ['gibbs', 'map', 'sup']:
        mrf_config = '0.1r,cross3,noprior:noprior,0.1,{},6'.format(pred)
        RandomField(mrf_config, [1, 2, 3], None, 2)
    cleanup()


def test_mrf_invalid_config():
    for stop in ['foo', 'r', '0.1t', '']:
        mrf_config = '{},chain,noprior:noprior,0.1,map,6'.format(stop)
        with pytest.raises(RuntimeError):
            RandomField(mrf_config, [1, 2, 3], None, 2)
    for shape in ['foo', 'cross', 'chain3', '']:
        mrf_config = '0.1r,{},noprior:noprior,0.1,map,6'.format(shape)
        with pytest.raises(RuntimeError):
            RandomField(mrf_config, [1, 2, 3], None, 2)
    for prior in ['foo', '']:
        mrf_config = '0.1r,cross3,{}:noprior,0.1,map,6'.format(prior)
        with pytest.raises(RuntimeError):
            RandomField(mrf_config, [1, 2, 3], None, 2)
        mrf_config = '0.1r,cross3,noprior:{},0.1,map,6'.format(prior)
        with pytest.raises(RuntimeError):
            RandomField(mrf_config, [1, 2, 3], None, 2)
    mrf_config = '0.1r,cross3,noprior,0.1,map,6'
    with pytest.raises(RuntimeError):
        RandomField(mrf_config, [1, 2, 3], None, 2)
    for lam in ['foo', '0a', '']:
        mrf_config = '0.1r,cross3,noprior:noprior,{},map,6'.format(lam)
        with pytest.raises(RuntimeError):
            RandomField(mrf_config, [1, 2, 3], None, 2)
    for pred in ['foo', 'mapsups']:
        mrf_config = '0.1r,cross3,noprior:noprior,0.1,{},6'.format(pred)
        with pytest.raises(RuntimeError):
            RandomField(mrf_config, [1, 2, 3], None, 2)
    cleanup()
