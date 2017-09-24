import threading


class Event(object):
    """
    You can pass these into `AnyEvent` or `AllEvent` below to join
    these together. Beyond that, they act like `threading.Event` objects.
    """
    def __init__(self):
        self._event = threading.Event()
        # We have a lock to make set() and clear() atomic. It is re-entrant so
        # that we can create a callback that can atomically check various
        # conditions before running set or clear, without deadlocking (used in
        # AnyEvent and AllEvent, below).
        self._lock = threading.RLock()
        # We keep a mapping from other Event-like objects to pairs of (set,
        # clear) nullary functions.
        self._dependents = {}

    def set(self):
        with self._lock:
            self._event.set()
            # Note that the graph of all Event objects and their dependents is
            # a DAG, and that setting or clearing any Event will only need to
            # acquire the locks of the descendents of that Event. Consequently,
            # we cannot have a deadlock: that would require two Events that are
            # each others' descendents, and that cannot happen in a DAG.
            for set_function, clear_function in self._dependents.values():
                set_function()

    def clear(self):
        with self._lock:
            self._event.clear()
            # Similar to the implementation of set, we cannot have a deadlock
            # here because the Events form a DAG.
            for set_function, clear_function in self._dependents.values():
                clear_function()

    def is_set(self):
        return self._event.is_set()

    def wait(self, *args, **kws):
        return self._event.wait(*args, **kws)

    def _register(self, registrant, set_function, clear_function):
        with self._lock:
            if registrant in self._dependents:
                raise AssertionError("Cannot register an event twice")
            self._dependents[registrant] = (set_function, clear_function)

    def _unregister(self, registrant):
        with self._lock:
            if registrant not in self._dependents:
                raise AssertionError("Cannot unregister an event we never saw")
            self._dependents.pop(registrant)


    def destruct(self):  # Call this when you want the Event to go out of scope.
        raise NotImplementedError


class UsageError(Exception):
    """
    This is raised when you're trying to do something with an event that
    isn't allowed.
    """
    pass


class InverseEvent(Event):
    """
    This event is the inverse of the event passed in on initialization.
    If the base event is cleared, this is set and vice versa.

    Note: this type of event DOES NOT support being set and cleared itself. It
    just mirrors what happens to the base event we're inverse of.
    """
    def __init__(self, base_event):
        super(InverseEvent, self).__init__()
        self._base_event = base_event
        self._base_event._register(self, self._set_callback, self._clear_callback)
        if not self._base_event.is_set():
            self._event.set()

    def set(self):
        raise UsageError("Cannot set an inverse event!")

    def clear(self):
        raise UsageError("Cannot clear an inverse event!")

    def _set_callback(self):
        super(InverseEvent, self).clear()

    def _clear_callback(self):
        super(InverseEvent, self).set()

    def destruct(self):
        self._base_event._unregister(self)


class _ComboEvent(Event):
    def __init__(self, *events):
        """
        We combine all the input events together when creating this one, using
        the callbacks defined in subclasses.
        """
        super(_ComboEvent, self).__init__()
        self._ancestors = events

        with self._lock:
            for event in self._ancestors:
                event._register(self, self._set_callback, self._clear_callback)

            self._initialize()

    def destruct(self):
        # Before we can go out of scope, we need to remove ourselves from all
        # our ancestors, so that they don't hold references to us.
        for event in self._ancestors:
            event._unregister(self)

    def _initialize(self):  # Called to initialize the state at the beginning
        raise NotImplementedError

    def _set_callback(self):  # Called when one of our ancestors is set
        raise NotImplementedError

    def _clear_callback(self):  # Called when one of our ancestors is cleared
        raise NotImplementedError

    # We also act as a context manager to support `with` statements.
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        # Ignore any exceptions raised; they'll get reraised outside our
        # context anyway. Just clean up our references.
        self.destruct()


class AnyEvent(_ComboEvent):
    """
    This Event gets set whenever any event in the constructor list is set, and
    gets cleared when they're all cleared.
    """
    def _set_callback(self):
        with self._lock:
            self.set()

    def _clear_callback(self):
        with self._lock:
            if not any(event.is_set() for event in self._ancestors):
                self.clear()

    def _initialize(self):
        if any(event.is_set() for event in self._ancestors):
            self._event.set()


class AllEvent(_ComboEvent):
    """
    This Event gets set whenever all the events in the constructor list are
    set, and gets cleared when any of them are cleared.
    """
    def _set_callback(self):
        with self._lock:
            if all(event.is_set() for event in self._ancestors):
                self.set()

    def _clear_callback(self):
        with self._lock:
            self.clear()

    def _initialize(self):
        if all(event.is_set() for event in self._ancestors):
            self._event.set()
