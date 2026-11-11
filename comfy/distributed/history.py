from __future__ import annotations

import copy
import collections
import typing
from itertools import islice

from ..component_model.queue_types import HistoryEntry, QueueItem, ExecutionStatus, MAXIMUM_HISTORY_SIZE, public_queue_tuple


class History:
    def __init__(self, maximum_history_size=MAXIMUM_HISTORY_SIZE):
        self.history: typing.OrderedDict[str, HistoryEntry] = collections.OrderedDict()
        self.maximum_history_size = maximum_history_size

    def put(self, queue_item: QueueItem, outputs: dict, status: ExecutionStatus, meta: dict | None = None):
        self.history[queue_item.prompt_id] = HistoryEntry(prompt=public_queue_tuple(queue_item.queue_tuple),
                                                          outputs=outputs,
                                                          status=ExecutionStatus(*status).as_dict(),
                                                          meta={} if meta is None else meta)
        while len(self.history) > self.maximum_history_size:
            self.history.popitem(last=False)

    def copy(self, prompt_id: typing.Optional[str | int] = None, max_items: typing.Optional[int] = None,
             offset: typing.Optional[int] = None) -> dict[str, HistoryEntry]:
        if prompt_id is not None:
            return {prompt_id: copy.deepcopy(self.history[prompt_id])} if prompt_id in self.history else {}
        if offset is None or offset < 0:
            offset = max(len(self.history) - max_items, 0) if max_items is not None else 0
        stop = None if max_items is None else offset + max_items
        return collections.OrderedDict((key, copy.deepcopy(self.history[key]))
                                       for key in islice(self.history, offset, stop))

    def get_status(self, prompt_id: str) -> ExecutionStatus | None:
        entry = self.history.get(prompt_id)
        return ExecutionStatus(**copy.deepcopy(entry['status'])) if entry else None

    def clear(self):
        self.history.clear()

    def pop(self, key: str):
        self.history.pop(key, None)
