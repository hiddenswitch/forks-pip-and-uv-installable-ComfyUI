#include <Python.h>
#include <stdatomic.h>

static void *(*allocate_callback)(size_t, int, void *);
static void (*free_callback)(void *, size_t, int, void *);
static atomic_bool enabled;

void managed_bridge_start(void *(*allocate)(size_t, int, void *),
                          void (*release)(void *, size_t, int, void *))
{
    allocate_callback = allocate;
    free_callback = release;
    atomic_store(&enabled, 1);
}

void managed_bridge_stop(void)
{
    atomic_store(&enabled, 0);
}

static int can_call_python(void)
{
    if (!atomic_load(&enabled) || !Py_IsInitialized()) {
        return 0;
    }
#if PY_VERSION_HEX >= 0x030D0000
    return !Py_IsFinalizing();
#else
    return !_Py_IsFinalizing();
#endif
}

static void *call_python(void *pointer, size_t size, int device, void *stream, int releasing)
{
    if (!can_call_python()) {
        return NULL;
    }
    PyGILState_STATE gil = PyGILState_Ensure();
    void *result = NULL;
    if (can_call_python()) {
        /* A tensor can be freed while its caller is propagating a Python exception. */
#if PY_VERSION_HEX >= 0x030C0000
        PyObject *error = PyErr_GetRaisedException();
#else
        PyObject *type, *error, *traceback;
        PyErr_Fetch(&type, &error, &traceback);
#endif
        if (releasing) {
            free_callback(pointer, size, device, stream);
        } else {
            result = allocate_callback(size, device, stream);
        }
#if PY_VERSION_HEX >= 0x030C0000
        PyErr_SetRaisedException(error);
#else
        PyErr_Restore(type, error, traceback);
#endif
    }
    PyGILState_Release(gil);
    return result;
}

void *managed_bridge_alloc(size_t size, int device, void *stream)
{
    return call_python(NULL, size, device, stream, 0);
}

void managed_bridge_free(void *pointer, size_t size, int device, void *stream)
{
    call_python(pointer, size, device, stream, 1);
}
