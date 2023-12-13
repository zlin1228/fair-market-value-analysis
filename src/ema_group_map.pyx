from libcpp.unordered_map cimport unordered_map

cdef class EmaGroupMap:
    cdef unordered_map[int, unordered_map[int, double]] ema_map
    cdef double alpha

    def __cinit__(self, alpha: float):
        self.ema_map = unordered_map[int, unordered_map[int, double]]()
        self.alpha = alpha

    cpdef update_ema(self, group: int, group_id: int, value: float):
        cdef float prev_ema
        if not self.has_group(group):
            self.ema_map[group] = unordered_map[int, double]()
        if not self.has_group_id(group, group_id):
            self.ema_map[group][group_id] = value
        else:
            prev_ema = self.ema_map[group][group_id]
            self.ema_map[group][group_id] = (self.alpha * value) + ((1 - self.alpha) * prev_ema)

    cpdef bint has_group(self, group: int):
        it = self.ema_map.find(group)
        return it != self.ema_map.end()

    cpdef bint has_group_id(self, group: int, group_id: int):
        if not self.has_group(group):
            return False
        it = self.ema_map[group].find(group_id)
        return it != self.ema_map[group].end()

    cdef float _get_ema(self, group: int, group_id: int) except? -1:
        if not self.has_group(group):
            raise ValueError(f"Group {group} not found.")
        if not self.has_group_id(group, group_id):
            raise ValueError(f"Group ID {group_id} not found in group {group}.")
        return self.ema_map[group][group_id]

    cpdef get_ema(self, group: int, group_id: int):
        return self._get_ema(group, group_id)