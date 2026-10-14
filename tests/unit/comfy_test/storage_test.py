import os
import platform
from unittest import mock

import pytest

from comfy import storage


def test_fast_nvme_link_thresholds(sysfs):
    cases = [
        ("8.0 GT/s PCIe", "4", True),
        ("16.0 GT/s PCIe", "4", True),
        ("32.0 GT/s PCIe", "2", True),
        ("8.0 GT/s PCIe", "2", False),
    ]
    for i, (speed, width, expected) in enumerate(cases):
        sysfs.nvme(f"nvme{i}n1", controllers={f"nvme{i}": (speed, width)})
        assert storage._fast_nvme(f"nvme{i}n1") is expected


def test_non_nvme_is_not_fast():
    assert storage._fast_nvme("sda") is False


def test_every_model_file_must_be_on_fast_storage():
    with mock.patch.object(storage, "fast_storage", side_effect=[True, False]):
        assert storage.model_fast_disk(["first", "second"]) is False
    with mock.patch.object(storage, "fast_storage", side_effect=[True, True]):
        assert storage.model_fast_disk(["first", "second"]) is True


class FakeSysfs:
    """A /sys tree and /proc/self/mountinfo with just the nodes storage reads."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "sys"
        self.mountinfo = tmp_path / "mountinfo"
        (self.root / "dev" / "block").mkdir(parents=True)
        (self.root / "class" / "block").mkdir(parents=True)
        (self.root / "fs" / "btrfs").mkdir(parents=True)
        monkeypatch.setattr(storage, "SYSFS", str(self.root), raising=False)
        monkeypatch.setattr(storage, "PROC_SELF_MOUNTINFO", str(self.mountinfo), raising=False)
        storage._linux_fast_storage.cache_clear()

    def controller(self, name, speed, width):
        pci = self.root / "devices" / "pci0000:00" / f"0000:00:{name[4:]:0>2}.0"
        (pci / "nvme" / name).mkdir(parents=True, exist_ok=True)
        (pci / "current_link_speed").write_text(f"{speed}\n")
        (pci / "current_link_width").write_text(f"{width}\n")
        (self.root / "class" / "nvme").mkdir(parents=True, exist_ok=True)
        link = self.root / "class" / "nvme" / name
        if not link.exists():
            link.symlink_to(pci / "nvme" / name)
            (pci / "nvme" / name / "device").symlink_to(pci)
        return pci / "nvme" / name

    def nvme(self, namespace, controllers, partitions=(), multipath=False):
        """A namespace block device; with multipath the kernel names it after
        the subsystem, and each path device hangs off a controller."""
        head = self.root / "devices" / "virtual" / "nvme-subsystem" / namespace
        head.mkdir(parents=True)
        (self.root / "class" / "block" / namespace).symlink_to(head)
        controller_dirs = [self.controller(name, *link) for name, link in controllers.items()]
        if multipath:
            (head / "multipath").mkdir()
            for i, controller in enumerate(controller_dirs):
                path_device = controller / f"{namespace[:-2]}c{i}{namespace[-2:]}"
                path_device.mkdir()
                (head / "multipath" / path_device.name).symlink_to(path_device)
        else:
            (head / "device").symlink_to(controller_dirs[0])
        for partition in partitions:
            (head / partition).mkdir()
            (head / partition / "partition").write_text("1\n")
            (self.root / "class" / "block" / partition).symlink_to(head / partition)
            (head / partition / "slaves").mkdir()

    def btrfs(self, uuid, *devices):
        directory = self.root / "fs" / "btrfs" / uuid / "devices"
        directory.mkdir(parents=True)
        for device in devices:
            (directory / device).symlink_to(self.root / "class" / "block" / device)

    def mounts(self, *lines):
        self.mountinfo.write_text("".join(line + "\n" for line in lines))


def mount(mount_id, mount_point, fstype, source, dev="0:0"):
    # 0:0 is never a file's st_dev, so these resolve by mount point
    escaped = mount_point.replace(" ", "\\040")
    return f"{mount_id} 1 {dev} / {escaped} rw,noatime shared:1 - {fstype} {source} rw"


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    if platform.system() != "Linux":
        pytest.skip("Linux mount and sysfs resolution")
    return FakeSysfs(tmp_path, monkeypatch)


def model_file(tmp_path, name="dir with space"):
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "model.safetensors"
    path.write_bytes(b"x")
    return str(path)


def test_btrfs_subvolume_resolves_through_mountinfo(sysfs, tmp_path):
    # btrfs reports an anonymous st_dev (0:31 on appmana-001) that has no
    # /sys/dev/block entry; the backing device comes from mountinfo
    path = model_file(tmp_path)
    sysfs.nvme("nvme9n1", {"nvme9": ("16.0 GT/s PCIe", "4")}, partitions=["nvme9n1p3"])
    sysfs.nvme("nvme8n1", {"nvme8": ("2.5 GT/s PCIe", "1")}, partitions=["nvme8n1p1"])
    sysfs.btrfs("c22b34df", "nvme9n1p3")
    sysfs.mounts(
        mount(35, "/", "btrfs", "/dev/nvme8n1p1"),
        mount(36, os.path.dirname(path), "btrfs", "/dev/nvme9n1p3"),
        mount(37, os.path.dirname(path) + "-sibling", "ext4", "/dev/nvme8n1p1"),
    )
    assert storage.fast_storage(path) is True


def test_the_last_mount_on_a_mount_point_wins(sysfs, tmp_path):
    path = model_file(tmp_path)
    sysfs.nvme("nvme9n1", {"nvme9": ("16.0 GT/s PCIe", "4")}, partitions=["nvme9n1p3"])
    sysfs.nvme("nvme8n1", {"nvme8": ("2.5 GT/s PCIe", "1")}, partitions=["nvme8n1p1"])
    sysfs.mounts(
        mount(36, os.path.dirname(path), "ext4", "/dev/nvme9n1p3"),
        mount(37, os.path.dirname(path), "ext4", "/dev/nvme8n1p1"),
    )
    assert storage.fast_storage(path) is False


def test_multi_device_btrfs_needs_every_device_fast(sysfs, tmp_path):
    path = model_file(tmp_path)
    sysfs.nvme("nvme9n1", {"nvme9": ("16.0 GT/s PCIe", "4")}, partitions=["nvme9n1p3"])
    sysfs.nvme("nvme8n1", {"nvme8": ("2.5 GT/s PCIe", "1")}, partitions=["nvme8n1p1"])
    sysfs.btrfs("c22b34df", "nvme9n1p3", "nvme8n1p1")
    sysfs.mounts(mount(36, "/", "btrfs", "/dev/nvme9n1p3"))
    assert storage.fast_storage(path) is False


def test_nvme_multipath_namespace_uses_its_path_controllers(sysfs, tmp_path):
    # with native NVMe multipath the namespace is named after the subsystem
    # (nvme1n1) while the controller carrying it can be nvme3
    path = model_file(tmp_path)
    sysfs.nvme("nvme1n1", {"nvme3": ("16.0 GT/s PCIe", "4")}, partitions=["nvme1n1p3"], multipath=True)
    sysfs.controller("nvme1", "2.5 GT/s PCIe", "1")
    sysfs.mounts(mount(36, "/", "btrfs", "/dev/nvme1n1p3"))
    assert storage.fast_storage(path) is True


def test_network_and_virtual_filesystems_are_not_fast(sysfs, tmp_path):
    path = model_file(tmp_path)
    sysfs.mounts(mount(36, "/", "fuse.seaweedfs", "seaweedfs:8888:/buckets"))
    assert not storage.fast_storage(path)


def test_this_hosts_btrfs_nvme_resolves(tmp_path):
    """On a host whose files live on btrfs over NVMe (appmana-001), the
    anonymous st_dev must not make fast-disk detection give up."""
    if platform.system() != "Linux":
        pytest.skip("Linux only")
    path = os.path.realpath(__file__)
    device = os.stat(path).st_dev
    if os.path.exists(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}"):
        pytest.skip("st_dev resolves directly on this host")
    mounts = []
    with open("/proc/self/mountinfo", encoding="utf-8") as f:
        for line in f:
            head, tail = line.split(" - ", 1)
            mount_point = storage._unescape_mountinfo(head.split()[4])
            if os.path.commonpath([path, mount_point]) == mount_point:
                mounts.append((mount_point, *tail.split()[:2]))
    _, fstype, source = max(mounts, key=lambda mount: len(mount[0]))
    if fstype != "btrfs" or not source.startswith("/dev/nvme"):
        pytest.skip("test file is not on btrfs over NVMe")
    if not os.path.exists(f"/sys/class/block/{os.path.basename(source)}"):
        pytest.skip("backing block device is not exposed in this namespace")
    storage._linux_fast_storage.cache_clear()
    assert storage.fast_storage(path) is not None
