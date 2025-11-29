# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from verl.workers.rollout.partial_buffer import PartialBuffer, pop_first


def custom_tail_filter(buffer, num_samples: int):
    """Pop from the tail to verify custom filter loading."""
    num = min(len(buffer), num_samples)
    samples = buffer[-num:]
    del buffer[-num:]
    return samples


def test_partial_buffer_fifo_basic():
    buf = PartialBuffer()
    buf.add_samples([1, 2, 3])
    out = buf.get_samples(2)
    assert out == [1, 2]
    assert len(buf) == 1
    assert buf.get_samples(5) == [3]
    assert len(buf) == 0


def test_partial_buffer_request_more_than_size():
    buf = PartialBuffer()
    buf.add_samples([1, 2])
    out = buf.get_samples(10)
    assert out == [1, 2]
    assert len(buf) == 0


def test_partial_buffer_noop_when_empty_or_zero():
    buf = PartialBuffer()
    assert buf.get_samples(0) == []
    assert buf.get_samples(5) == []
    buf.add_samples([])
    assert len(buf) == 0


def test_partial_buffer_custom_filter_path():
    # use dotted path to load the custom filter in this module
    filter_path = f"{__name__}:custom_tail_filter"
    buf = PartialBuffer(filter_path)
    buf.add_samples([1, 2, 3, 4])
    out = buf.get_samples(3)
    # custom filter pops from tail
    assert out == [2, 3, 4]
    assert len(buf) == 1
    assert buf.get_samples(2) == [1]


def test_pop_first_standalone():
    data = [1, 2, 3]
    out = pop_first(data, 2)
    assert out == [1, 2]
    assert data == [3]


def test_partial_buffer_filter_path_invalid():
    buf = PartialBuffer()
    buf.add_samples([1])
    # invalid module/function path should raise
    try:
        PartialBuffer("non.existent:fn")
    except Exception as e:  # noqa: PIE786
        assert isinstance(e, (ValueError, ModuleNotFoundError))
    else:
        raise AssertionError("Invalid filter path did not raise")


def test_partial_buffer_filter_not_callable(monkeypatch):
    # monkeypatch _load_filter to return non-callable
    import importlib

    module_name = __name__
    mod = importlib.import_module(module_name)
    setattr(mod, "not_callable", 123)
    try:
        try:
            PartialBuffer(f"{module_name}:not_callable")
        except Exception as e:  # noqa: PIE786
            assert isinstance(e, TypeError)
        else:
            raise AssertionError("Non-callable filter did not raise TypeError")
    finally:
        delattr(mod, "not_callable")


def test_partial_buffer_filter_returns_less():
    def filter_skip_one(buffer, num_samples):
        # return half and leave half in buffer
        num = min(len(buffer), num_samples // 2)
        samples = buffer[:num]
        del buffer[:num]
        return samples

    buf = PartialBuffer()
    buf._filter = filter_skip_one  # inject for test
    buf.add_samples([1, 2, 3, 4])
    out = buf.get_samples(4)
    # only half popped
    assert out == [1, 2]
    assert buf.get_samples(10) == [3, 4]
