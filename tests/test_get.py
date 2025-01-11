class signal:
    def __init__(self, _):
        self._name = None
        self._obj = None

    @property
    def obj(self):
        return self._obj

    @property
    def name(self):
        return self._name

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        # assert isinstance(
        #     obj, EventLoopObject
        # ), f"signals can only be added to {EventLoopObject.__name__}, not {type(obj)}"
        self._obj = obj
        return self

    def connect(self, other, slot: str = None):
        self._obj.connect(self._name, other, slot)

    def disconnect(self, other, slot: str = None):
        self._obj.disconnect(self._name, other, slot)

    def emit(self, *args):
        self._obj.emit(self._name, *args)

    def emit_many(self, list_of_args):
        self._obj.emit_many(self._name, list_of_args)

    def broadcast_on(self, event_loop):
        self._obj.register_broadcast(self._name, event_loop)


class Abs:
    def __init__(self):
        pass

    @signal
    def connect(self):
        ...


if __name__ == '__main__':
    abs = Abs()
    cn = abs.connect
    print(cn)
