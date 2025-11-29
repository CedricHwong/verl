# Copyright 2023-2024 Kingdee Cosmic AI Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Simple partial rollout buffer to support APRIL-style recycle/resume across rollout steps.

The buffer stores arbitrary sample payloads (could be request objects or token-level
states) and exposes FIFO pop by default. Filter can be customized via dotted-path
function.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional


def _load_filter(filter_path: Optional[str]) -> Optional[Callable[[list, int], list]]:
    """Load a filter function from a dotted path like `pkg.module.fn`."""
    if not filter_path:
        return None
    if ":" in filter_path:
        module_path, fn_name = filter_path.split(":", maxsplit=1)
    elif "." in filter_path:
        module_path, fn_name = filter_path.rsplit(".", maxsplit=1)
    else:
        raise ValueError(f"Invalid filter path: {filter_path}")
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Filter {filter_path} is not callable.")
    return fn


def pop_first(buffer: list, num_samples: int) -> list:
    """Default FIFO pop."""
    num = min(len(buffer), num_samples)
    samples = buffer[:num]
    del buffer[:num]
    return samples


class PartialBuffer:
    """In-memory partial rollout buffer."""

    def __init__(self, buffer_filter_path: Optional[str] = None):
        self._buffer: list[Any] = []
        self._filter = _load_filter(buffer_filter_path) or pop_first

    def __len__(self) -> int:
        return len(self._buffer)

    def add_samples(self, samples: list[Any]) -> None:
        if not samples:
            return
        self._buffer.extend(samples)

    def get_samples(self, num_samples: int) -> list[Any]:
        if num_samples <= 0 or len(self._buffer) == 0:
            return []
        return self._filter(self._buffer, num_samples)
