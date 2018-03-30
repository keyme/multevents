from functools import wraps
import threading


class UsageError(Exception):
    """
    This is raised when you're trying to do something with an event that
    isn't allowed.
    """
    pass


def _atomic(func):
    """
    A decorator to acquire an object's lock for the entirety of a function.
    """
    @wraps(func)
    def inner(self, *args, **kws):
        with self._lock:
            return func(self, *args, **kws)
    return inner


class _ContextManagerMixin(object):  # Inherit object for Python2 compatibility
    """
    Inherit from this for Events that are created within a context manager.
    """
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        # Ignore any exceptions raised; they'll get reraised outside our
        # context anyway. Just clean up our references.
        self.destruct()

    def destruct(self):  # Call this when you want the Event to go out of scope.
        pass


class Event(_ContextManagerMixin):
    """
    You can pass these into `AnyEvent` or `AllEvent` below to join
    these together. Beyond that, they act like `threading.Event` objects.
    """
    def __init__(self):
        self._event = threading.Event()
        # We have a lock to make _set() and _clear() atomic. It is re-entrant
        # so that we can create a callback that can atomically check various
        # conditions before running set or clear, without deadlocking (used in
        # AnyEvent and AllEvent, below).
        self._lock = threading.RLock()
        # We keep a mapping from other Event-like objects to pairs of (set,
        # clear) nullary functions.
        self._dependents = {}

    def set(self):
        self._set()

    def clear(self):
        self._clear()

    @_atomic
    def _set(self):
        self._event.set()
        # Note that the graph of all Event objects and their dependents is
        # a DAG, and that setting or clearing any Event will only need to
        # acquire the locks of the descendents of that Event. Consequently,
        # we cannot have a deadlock: that would require two Events that are
        # each others' descendents, and that cannot happen in a DAG.
        for set_function, clear_function in self._dependents.values():
            set_function()

    @_atomic
    def _clear(self):
        self._event.clear()
        # Similar to the implementation of set, we cannot have a deadlock
        # here because the Events form a DAG.
        for set_function, clear_function in self._dependents.values():
            clear_function()

    def is_set(self):
        return self._event.is_set()

    def wait(self, *args, **kws):
        return self._event.wait(*args, **kws)

    @_atomic
    def _register(self, registrant, set_function, clear_function):
        if registrant in self._dependents:
            raise UsageError("Cannot register an event twice")
        self._dependents[registrant] = (set_function, clear_function)

    @_atomic
    def _unregister(self, registrant):
        if registrant not in self._dependents:
            raise UsageError("Cannot unregister an event we never saw")
        self._dependents.pop(registrant)


class _ComboEvent(Event):
    def __init__(self, *events):
        """
        We combine all the input events together when creating this one, using
        the callbacks defined in subclasses. Think of this as an abstract base
        class.
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

    def set(self):
        raise UsageError("Don't set combination events directly.")

    def clear(self):
        raise UsageError("Don't clear combination events directly.")

    def _initialize(self):  # Called to initialize the state at the beginning
        raise NotImplementedError

    def _set_callback(self):  # Called when one of our ancestors is set
        raise NotImplementedError

    def _clear_callback(self):  # Called when one of our ancestors is cleared
        raise NotImplementedError


class InverseEvent(_ComboEvent):
    """
    This event is the inverse of the event passed in on initialization.  If the
    base event is cleared, this is set and vice versa.
    """
    def __init__(self, event):
        # The difference between this __init__ and _ComboEvent.__init__ is that
        # this one only accepts a single parent Event, whereas _ComboEvent can
        # have arbitrarily many.
        super(InverseEvent, self).__init__(event)

    def _initialize(self):
        if all(not event.is_set() for event in self._ancestors):
            # There is just the one ancestor, but we look for "all" of them to
            # simplify the statement.
            self._set()

    @_atomic
    def _set_callback(self):
        self._clear()

    @_atomic
    def _clear_callback(self):
        self._set()


class AnyEvent(_ComboEvent):
    """
    This Event gets set whenever any event in the constructor list is set, and
    gets cleared when they're all cleared.
    """
    @_atomic
    def _set_callback(self):
        self._set()

    @_atomic
    def _clear_callback(self):
        if not any(event.is_set() for event in self._ancestors):
            self._clear()

    def _initialize(self):
        if any(event.is_set() for event in self._ancestors):
            self._event.set()


class AllEvent(_ComboEvent):
    """
    This Event gets set whenever all the events in the constructor list are
    set, and gets cleared when any of them are cleared.
    """
    @_atomic
    def _set_callback(self):
        if all(event.is_set() for event in self._ancestors):
            self._set()

    @_atomic
    def _clear_callback(self):
        self._clear()

    def _initialize(self):
        if all(event.is_set() for event in self._ancestors):
            self._event.set()
