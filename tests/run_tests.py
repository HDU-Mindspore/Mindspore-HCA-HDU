import mindspore
import mindspore as ms
import mindspore.context as context
import pytest

context.set_context(mode=context.PYNATIVE_MODE)
ms.set_device('CPU')

if __name__ == '__main__':
    pytest.main(['tests/st/mint/test_acos.py'])
    pytest.main(['tests/st/mint/test_copy_.py'])
    #pytest.main(['tests/st/mint/test_perf_acos.py'])
    #pytest.main(['tests/st/mint/test_perf_copy_.py'])
