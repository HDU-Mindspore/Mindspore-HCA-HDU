import mindspore
import mindspore as ms
import mindspore.context as context
import pytest

context.set_context(mode=context.PYNATIVE_MODE)
ms.set_device('CPU')

if __name__ == '__main__':
    pytest.main(['tests/st/mint/test_asin.py'])
    pytest.main(['tests/st/mint/test_acos.py'])
    pytest.main(['tests/st/mint/test_asinh.py'])
    pytest.main(['tests/st/mint/test_acosh.py'])
    pytest.main(['tests/st/mint/test_cumsum.py'])
    pytest.main(['tests/st/mint/test_copy_.py'])
    pytest.main(['tests/st/mint/test_sin.py'])
    pytest.main(['tests/st/mint/test_cos.py'])
    pytest.main(['tests/st/mint/test_atan.py'])
    # FIXME:relu_算子，由于当前框架CPU后端不支持原地更新算子的输入输出共用同一个Tensor，会导致反向精度不正确，因此不执行反向测试用例
    pytest.main(['tests/st/mint/test_relu_.py'])
    # FIXME:stack算子，由于框架不支持Tuple(Tensor)类型输入，因此不执行stack测试用例
    # pytest.main(['tests/st/mint/test_stack.py'])
    pytest.main(['tests/st/mint/test_clone.py'])
    pytest.main(['tests/st/mint/test_logical_and.py'])
    pytest.main(['tests/st/mint/test_logical_not.py'])
    pytest.main(['tests/st/mint/test_exp.py'])
    pytest.main(['tests/st/mint/test_zeros_like.py'])
    # FIXME:index_select算子，反向依赖zeros_like和index_add_接入，暂时不执行反向测试用例
    pytest.main(['tests/st/mint/test_index_select.py'])

    # FIXME:执行性能用例时，需要把MS日志级别设置为ERROR级别，否则太多Warning日志会影响性能。
    pytest.main(['tests/st/mint/test_perf_acos.py'])
    pytest.main(['tests/st/mint/test_perf_copy_.py'])
    # FIXME:sin 算子走内置算子，因此性能不达标。
    # pytest.main(['tests/st/mint/test_perf_sin.py'])
    # FIXME:atan 算子走内置算子，因此性能不达标。
    pytest.main(['tests/st/mint/test_perf_atan.py'])
    # FIXME: relu_ 算子性能不达标，怀疑是由于框架多申请了一个输出Tensor导致。
    pytest.main(['tests/st/mint/test_perf_relu_.py'])
    pytest.main(['tests/st/mint/test_perf_stack.py'])
    pytest.main(['tests/st/mint/test_perf_clone.py'])
    # FIXME: index 算子性能不达标，原因是torch走View，MS暂时不支持。
    pytest.main(['tests/st/mint/test_perf_index.py'])
    pytest.main(['tests/st/mint/test_perf_logical_and.py'])
    pytest.main(['tests/st/mint/test_perf_logical_not.py'])
    pytest.main(['tests/st/mint/test_perf_index_select.py'])
    pytest.main(['tests/st/mint/test_perf_acosh.py'])
    pytest.main(['tests/st/mint/test_perf_asinh.py'])
    # FIXME: cumsum 算子走内置算子，因此性能不达标
    # pytest.main(['tests/st/mint/test_perf_cumsum.py'])
