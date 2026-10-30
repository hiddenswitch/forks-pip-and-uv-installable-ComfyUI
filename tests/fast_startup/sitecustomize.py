"""Socket policy inherited by Python workers in the fast test profile."""
import ipaddress
import os
import socket


def _local(host):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host.split("%")[0]).is_loopback:
            return
    except ValueError:
        pass
    raise RuntimeError("Fast tests prohibit external network access: " + str(host))


if os.environ.get("COMFYUI_FAST_OFFLINE") == "1":
    _resolve = socket.getaddrinfo
    _connect = socket.socket.connect
    _connect_ex = socket.socket.connect_ex
    _sendto = socket.socket.sendto

    def getaddrinfo(host, *args, **kwargs):
        if host is not None:
            _local(host)
        return _resolve(host, *args, **kwargs)

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _local(address[0])
        return _connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _local(address[0])
        return _connect_ex(self, address)

    def sendto(self, data, *args):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _local(args[-1][0])
        return _sendto(self, data, *args)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
